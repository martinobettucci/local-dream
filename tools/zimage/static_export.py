"""Fixed-shape, split-capable wrapper around ZImageTransformer2DModel.

The stock `forward` takes `list[Tensor]` and packs variable-length sequences,
which cannot become a static ONNX graph (`aten::pad_sequence`). This rebuilds it
for the one case the runner needs — batch 1, a fixed caption slot count, a fixed
canvas — so every length is a compile-time constant.

It also cuts the model into pieces, since 6B parameters do not fit one HTP
context. Each piece is its own nn.Module and exports separately:

    part 1     : (sample, timestep, context, pos_ids, attn_mask, cap_pad_mask)
                     -> (hidden, emb)      [or out_sample if it is also last]
    part 1<k<N : (hidden, emb, pos_ids, attn_mask) -> hidden
    part N     : (hidden, emb, pos_ids, attn_mask) -> out_sample

Faithfulness rests on three things that are easy to get wrong; see
docs/zimage.md section 9 and verify_static_equivalence.py:

  * sequence order is [image, caption], image first;
  * `attn_mask` must be applied to the caption refiner AS WELL AS the main
    blocks, masking past cap_len = round-up(true_len, 32) — the reference needs
    no mask at all at batch 1, and copying that omission is silently wrong;
  * image tokens are positioned at t = cap_len + 1, so `pos_ids` is an input.
"""

import math

import torch
import torch.nn as nn


def caption_len(true_len: int, seq_multiple: int = 32) -> int:
    """Where the reference's caption block ends, for a prompt of `true_len`."""
    return math.ceil(true_len / seq_multiple) * seq_multiple


def build_positions(true_len, cap_slots, grid_h, grid_w, seq_multiple=32):
    """Host-side pos_ids / attn_mask / cap_pad_mask, mirroring PipelineZImage.

    Returns (pos_ids [T,3] int32, attn_mask [T] float, cap_pad_mask [S] float)
    with T = grid_h*grid_w + cap_slots, image tokens first.
    """
    cap_len = min(caption_len(true_len, seq_multiple), cap_slots)
    n_img = grid_h * grid_w
    total = n_img + cap_slots

    pos = torch.zeros(total, 3, dtype=torch.int32)
    attn = torch.zeros(total, dtype=torch.float32)

    rows = torch.arange(grid_h, dtype=torch.int32).repeat_interleave(grid_w)
    cols = torch.arange(grid_w, dtype=torch.int32).repeat(grid_h)
    pos[:n_img, 0] = cap_len + 1
    pos[:n_img, 1] = rows
    pos[:n_img, 2] = cols
    attn[:n_img] = 1.0

    pos[n_img:n_img + cap_len, 0] = torch.arange(1, cap_len + 1, dtype=torch.int32)
    attn[n_img:n_img + cap_len] = 1.0

    cap_pad = torch.zeros(cap_slots, dtype=torch.float32)
    cap_pad[true_len:] = 1.0
    return pos, attn, cap_pad


class StaticZImageDiT(nn.Module):
    """One exportable piece of the DiT.

    `first` runs the embedders and refiners and emits the residual stream;
    `last` runs the final layer and unpatchifies. Both may be true (N = 1).
    """

    def __init__(self, model, cap_slots, latent_h, latent_w,
                 block_start=0, block_end=None, patch=2, f_patch=1,
                 first=None, last=None):
        super().__init__()
        self.m = model
        self.key = f"{patch}-{f_patch}"
        if self.key not in model.all_x_embedder:
            raise KeyError(f"model has no {self.key} embedder; has {list(model.all_x_embedder)}")
        self.cap_slots = int(cap_slots)
        self.p, self.fp = int(patch), int(f_patch)
        self.C = int(model.config.in_channels)
        self.H, self.W = int(latent_h), int(latent_w)
        self.grid_h, self.grid_w = self.H // self.p, self.W // self.p
        self.n_img = self.grid_h * self.grid_w

        # The image block carries no padding only when it lands on a whole
        # number of 32-token groups; the runner's canvases all do, and relying
        # on it lets the graph skip the x_pad_token substitution entirely.
        if self.n_img % 32 != 0:
            raise ValueError(f"{self.n_img} image tokens is not a multiple of 32")

        n_layers = len(model.layers)
        self.block_start = int(block_start)
        self.block_end = n_layers if block_end is None else int(block_end)
        if not 0 <= self.block_start < self.block_end <= n_layers:
            raise ValueError(f"bad block range [{block_start}, {block_end}) of {n_layers}")
        # Derived from the block range for a whole model, but overridable: the
        # 6B model never fits in RAM at once, so each part is built as a REDUCED
        # ZImageTransformer2DModel holding only its own blocks (renumbered from
        # 0). Such a piece looks like [0, n) — i.e. both first and last — when it
        # is really neither, so the caller states which it is.
        self.first = (self.block_start == 0) if first is None else bool(first)
        self.last = (self.block_end == n_layers) if last is None else bool(last)

    # -- pieces ------------------------------------------------------------
    # The obvious transcription of _patchify_image / unpatchify is a 7-D view
    # plus a fully interleaved permute. QNN rejects that outright:
    #     Failed to resolve 6D tensor by merging consecutive axes for Transpose
    #     with permutation [6, 0, 3, 1, 4, 2, 5]
    # It caps tensors at 5-D and can only reduce rank by merging axes that stay
    # adjacent, which an interleaved permutation never allows. pixel_unshuffle /
    # pixel_shuffle express exactly the same rearrangement in <=4-D and lower to
    # SpaceToDepth / DepthToSpace, which the HTP handles natively.
    #
    # Ordering note: pixel_(un)shuffle uses CRD — channel-major, then the two
    # block axes — whereas a Z-Image token is laid out (pH, pW, C). The extra
    # reshape/permute pair converts between the two.

    def _patchify(self, sample):
        # (1, C, H, W) -> (1, n_img, pH*pW*C) with per-token layout (pH, pW, C)
        p, c = self.p, self.C
        x = torch.nn.functional.pixel_unshuffle(sample, p)   # [1, C*p*p, gh, gw]
        x = x.reshape(1, c, p * p, self.n_img)               # CRD: (C, pH*pW)
        x = x.permute(0, 3, 2, 1)                            # [1, n_img, p*p, C]
        return x.reshape(1, self.n_img, p * p * c)

    def _unpatchify(self, tokens):
        # exact inverse of _patchify
        p, c = self.p, self.C
        x = tokens.reshape(1, self.n_img, p * p, c)
        x = x.permute(0, 3, 2, 1)                            # [1, C, p*p, n_img]
        x = x.reshape(1, c * p * p, self.grid_h, self.grid_w)
        return torch.nn.functional.pixel_shuffle(x, p)       # [1, C, H, W]

    def _embed(self, sample, timestep, context, freqs, mask, cap_pad_mask):
        """Everything part 1 does before the main blocks."""
        m = self.m
        # `timestep` is already sigma * t_scale (the value t_embedder consumes);
        # the stock forward multiplies by t_scale itself, we do not.
        emb = m.t_embedder(timestep)

        img_freqs = freqs[:, :self.n_img]
        cap_freqs = freqs[:, self.n_img:]
        cap_mask = mask[:, self.n_img:]

        x = m.all_x_embedder[self.key](self._patchify(sample))
        for layer in m.noise_refiner:
            x = layer(x, None, img_freqs, emb)

        cap = m.cap_embedder(context)
        cap = torch.where(cap_pad_mask.unsqueeze(-1) > 0.5, m.cap_pad_token, cap)
        for layer in m.context_refiner:
            cap = layer(cap, cap_mask, cap_freqs)

        return torch.cat([x, cap], dim=1), emb

    # -- forward -----------------------------------------------------------
    def forward(self, *args):
        if self.first:
            sample, timestep, context, pos_ids, attn_mask, cap_pad_mask = args
        else:
            hidden, emb, pos_ids, attn_mask = args

        # RoPE frequencies are gathered from the baked tables by pos_ids, so the
        # tables stay constant while the coordinates stay prompt-dependent.
        freqs = self.m.rope_embedder(pos_ids.reshape(-1, 3)).unsqueeze(0)
        mask = attn_mask > 0.5

        if self.first:
            hidden, emb = self._embed(sample, timestep, context, freqs,
                                      mask, cap_pad_mask)

        for layer in self.m.layers[self.block_start:self.block_end]:
            hidden = layer(hidden, mask, freqs, emb)

        if not self.last:
            return (hidden, emb) if self.first else hidden

        out = self.m.all_final_layer[self.key](hidden, c=emb)
        return self._unpatchify(out[:, :self.n_img])

    # -- export metadata ---------------------------------------------------
    @property
    def input_names(self):
        return (["sample", "timestep", "context", "pos_ids", "attn_mask", "cap_pad_mask"]
                if self.first else ["hidden", "emb", "pos_ids", "attn_mask"])

    @property
    def emb_dim(self):
        """Width of the adaLN vector handed between parts.

        This is TimestepEmbedder's OUTPUT width, which is not `dim`: the real
        model runs 256 -> 1024 -> 256 while dim is 3840. Small test configs can
        make the two coincide, which hides the difference until real weights
        turn up.
        """
        return self.m.t_embedder.mlp[-1].out_features

    @property
    def output_names(self):
        if self.last:
            return ["out_sample"]
        return ["hidden", "emb"] if self.first else ["hidden"]

    def example_inputs(self, true_len=None, dim=None):
        """Dummy inputs of the exact exported shapes."""
        true_len = self.cap_slots if true_len is None else true_len
        dim = self.m.config.dim if dim is None else dim
        pos, attn, cap_pad = build_positions(true_len, self.cap_slots,
                                             self.grid_h, self.grid_w)
        pos = pos.unsqueeze(0)
        attn = attn.unsqueeze(0)
        cap_pad = cap_pad.unsqueeze(0)
        if self.first:
            return (torch.randn(1, self.C, self.H, self.W),
                    torch.tensor([1000.0]),
                    torch.randn(1, self.cap_slots, self.m.config.cap_feat_dim),
                    pos, attn, cap_pad)
        total = self.n_img + self.cap_slots
        return (torch.randn(1, total, dim), torch.randn(1, self.emb_dim), pos, attn)

"""Fixed-shape, split-capable wrapper around ZImageTransformer2DModel.

The stock `forward` takes `list[Tensor]` and packs variable-length sequences,
which cannot become a static ONNX graph (`aten::pad_sequence`). This rebuilds it
for the one case the runner needs — batch 1, a fixed caption slot count, a fixed
canvas — so every length is a compile-time constant.

It also cuts the model into pieces, since 6B parameters do not fit one HTP
context. Each piece is its own nn.Module and exports separately:

    caption    : (context, pos_ids, attn_mask, cap_pad_mask) -> cap
    part 1     : (sample, timestep, cap, pos_ids[, attn_mask])
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


# The caption branch is its own graph. It has to be: everything the first part
# would otherwise carry -- the two embedders and BOTH refiner stacks, four
# full-width blocks -- put its context-binary generation at 15.9 GB peak RSS,
# which no 16 GB machine survives, while a one-block part peaks at 11.3 GB.
#
# The cut is along the model's own grain rather than an arbitrary one. The
# caption path (cap_embedder + context_refiner) and the image path
# (x_embedder + noise_refiner) are independent until the concatenation that
# forms the residual stream; nothing crosses between them. So the caption
# branch runs alone, and part 1 takes its result as an input and does the
# concatenation. It is also the only piece of the DiT that does not depend on
# the timestep, so the runner computes it once per prompt instead of once per
# step -- two of the model's 34 blocks that no longer run eight times.
CAP_INPUT_NAMES = ["context", "pos_ids", "attn_mask", "cap_pad_mask"]
CAP_OUTPUT_NAMES = ["cap"]


def dit_input_names(first, has_blocks=True, concat=False, has_refiner=None):
    """The graph IO contract, as free functions so it can be checked without
    building a part -- the resume path has an ONNX on disk and no model.

    "hidden_in", not "hidden": torch.onnx.export refuses to give a graph an
    input and an output with the same name and silently renames the input to
    "hidden.1", which the app -- which binds by name -- would only discover on
    device. Matches kZImageStateInNames in QnnModel.hpp.

    `cap`, `pos_ids` and `attn_mask` are present only on the graphs that read
    them, and the runner binds all three by presence for the same reason: the
    converter drops inputs no operation reads, so declaring one anyway would
    only mean the app looking for a tensor the graph does not have.

      `cap`        whichever graph performs the concatenation.
      `pos_ids`    any graph running a refiner or transformer block; RoPE
                   frequencies are gathered from the baked tables by it, so a
                   graph running neither never touches it.
      `attn_mask`  any graph running a transformer block. The noise refiner is
                   unmasked -- every image token is real.

    `has_refiner` defaults to `first`, which is true of every split where the
    head is one graph.
    """
    if has_refiner is None:
        has_refiner = first
    head = ["sample", "timestep"] if first else ["hidden_in", "emb"]
    return (head + (["cap"] if concat else [])
            + (["pos_ids"] if (has_refiner or has_blocks) else [])
            + (["attn_mask"] if has_blocks else []))


def dit_output_names(first, last):
    if last:
        return ["out_sample"]
    return ["hidden", "emb"] if first else ["hidden"]


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

    A piece is described by four independent things, because the model has two
    streams and the cut can land anywhere in either:

      `first`        runs t_embedder and the image embedder, so it takes
                     `sample`/`timestep` rather than `hidden_in`/`emb`, and
                     emits `emb` for the rest of the chain.
      `refiner`      which noise_refiner blocks it runs. These operate on the
                     IMAGE stream only (n_img tokens), so they must all come
                     before the concatenation.
      `concat`       whether it appends `cap` -- the caption branch's output --
                     turning the n_img stream into the full T-token one.
      `block_start`  which transformer blocks it runs. These need the full
      `block_end`    stream, so they only ever follow the concatenation.
      `last`         runs the final layer and unpatchifies.

    All of them may be true at once (N = 1, the whole DiT in one graph); the
    shipped split spreads them over 32 graphs plus the caption branch, because
    each refiner block over 4096 tokens costs ~4.7 GB at context-binary
    generation and two of them in one graph is 15.97 GB -- an OOM kill on a
    16 GB machine, measured twice.
    """

    def __init__(self, model, cap_slots, latent_h, latent_w,
                 block_start=0, block_end=None, patch=2, f_patch=1,
                 first=None, last=None, refiner=None, concat=None):
        super().__init__()
        self.m = model
        self.key = f"{patch}-{f_patch}"
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
        # Derived from the block range for a whole model, but overridable: the
        # 6B model never fits in RAM at once, so each part is built as a REDUCED
        # ZImageTransformer2DModel holding only its own blocks (renumbered from
        # 0). Such a piece looks like [0, n) — i.e. both first and last — when it
        # is really neither, so the caller states which it is. Resolved before
        # the range check because an empty range is legal only for a first part.
        self.first = (self.block_start == 0) if first is None else bool(first)
        self.last = (self.block_end == n_layers) if last is None else bool(last)
        self.has_blocks = self.block_end > self.block_start

        # Defaults describe the WHOLE model in one graph: it runs every refiner
        # block it holds and does the concatenation itself. A split states its
        # own share; nothing is inferred from the block range, because a piece
        # built as a reduced model cannot tell where it sits from its own shape.
        n_ref = len(getattr(model, "noise_refiner", ()))
        self.ref_start, self.ref_end = (0, n_ref) if refiner is None else \
            (int(refiner[0]), int(refiner[1]))
        self.concat = self.first if concat is None else bool(concat)
        self.has_refiner = self.ref_end > self.ref_start

        if not (0 <= self.block_start <= self.block_end <= n_layers):
            raise ValueError(f"bad block range [{block_start}, {block_end}) "
                             f"of {n_layers}")
        if not (0 <= self.ref_start <= self.ref_end <= n_ref):
            raise ValueError(f"bad refiner range [{self.ref_start}, "
                             f"{self.ref_end}) of {n_ref}")
        # A graph must DO something. An empty block range is fine on its own --
        # the shipped part 1 and part 2 carry no transformer block, only the
        # embedders and one refiner block each -- but a piece with no blocks, no
        # refiner, no concatenation and no final layer is a plan that lost work,
        # and it would export and run without complaint.
        if not (self.has_blocks or self.has_refiner or self.concat or self.last):
            raise ValueError(
                f"part [{block_start}, {block_end}) runs no refiner block, no "
                f"transformer block, no concatenation and no final layer")
        # A part whose only job is the concatenation is worse than useless: it
        # reads nothing but `hidden_in` and `cap`, so the converter drops `emb`
        # and `pos_ids` as unread and the graph stops matching the contract --
        # and the work it does is a memcpy the runner could do itself.
        if not (self.first or self.has_refiner or self.has_blocks or self.last):
            raise ValueError(
                "a part that only concatenates is not worth a context binary; "
                "give it the refiner block that precedes the concatenation")
        # Transformer blocks need the full T-token stream, which only exists
        # after the concatenation. A part that has blocks must therefore either
        # do the concatenation itself or follow one that did -- and the one case
        # this can catch locally is the part that does both in the wrong order.
        if self.has_blocks and self.has_refiner and not self.concat:
            raise ValueError(
                "a part running both refiner and transformer blocks must also "
                "concatenate: the refiner works on image tokens, the blocks on "
                "the full sequence")

        # Checked here rather than up front because a middle part has neither
        # module: export_dit.py deletes them before loading so a 2-block part
        # does not carry a GB of untraced, randomly-initialised weights through
        # ONNX export. Only a part that will actually index them needs them.
        if self.first and self.key not in model.all_x_embedder:
            raise KeyError(f"no {self.key} embedder; has {list(model.all_x_embedder)}")
        if self.last and self.key not in model.all_final_layer:
            raise KeyError(f"no {self.key} final layer; has {list(model.all_final_layer)}")

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

    # -- forward -----------------------------------------------------------
    def forward(self, *args):
        # Bound by name against this part's own contract rather than unpacked
        # positionally: the contract varies along four axes now, and a
        # positional unpack that silently shifts by one is exactly the kind of
        # error that survives every stage and shows up as a wrong image.
        kw = dict(zip(self.input_names, args))
        pos_ids = kw.get("pos_ids")
        attn_mask = kw.get("attn_mask")

        # RoPE frequencies are gathered from the baked tables by pos_ids, so the
        # tables stay constant while the coordinates stay prompt-dependent. A
        # part that runs neither a refiner nor a transformer block has no
        # attention to rotate and never asks for them.
        freqs = None if pos_ids is None else \
            self.m.rope_embedder(pos_ids.reshape(-1, 3)).unsqueeze(0)
        mask = None if attn_mask is None else attn_mask > 0.5
        m = self.m

        if self.first:
            # `timestep` is already sigma * t_scale (the value t_embedder
            # consumes); the stock forward multiplies by t_scale itself, we do
            # not.
            emb = m.t_embedder(kw["timestep"])
            hidden = m.all_x_embedder[self.key](self._patchify(kw["sample"]))
        else:
            hidden, emb = kw["hidden_in"], kw["emb"]

        # The noise refiner is unmasked in the reference and stays unmasked
        # here: every image token is real.
        if self.has_refiner:
            img_freqs = freqs[:, :self.n_img]
            for layer in m.noise_refiner[self.ref_start:self.ref_end]:
                hidden = layer(hidden, None, img_freqs, emb)

        if self.concat:
            hidden = torch.cat([hidden, kw["cap"]], dim=1)

        for layer in self.m.layers[self.block_start:self.block_end]:
            hidden = layer(hidden, mask, freqs, emb)

        if not self.last:
            return (hidden, emb) if self.first else hidden

        out = self.m.all_final_layer[self.key](hidden, c=emb)
        return self._unpatchify(out[:, :self.n_img])

    # -- export metadata ---------------------------------------------------
    @property
    def input_names(self):
        return dit_input_names(self.first, self.has_blocks, self.concat,
                               self.has_refiner)

    @property
    def emb_dim(self):
        """Width of the adaLN vector handed between parts.

        This is TimestepEmbedder's OUTPUT width, which is not `dim`: the real
        model runs 256 -> 1024 -> 256 while dim is 3840. Small test configs can
        make the two coincide, which hides the difference until real weights
        turn up.

        A part that does not run the embedder still needs the width, since `emb`
        is one of its graph inputs — so fall back to the adaLN modulation that
        every block carries, which is derived from the same vector.
        """
        t_embedder = getattr(self.m, "t_embedder", None)
        if t_embedder is not None:
            return t_embedder.mlp[-1].out_features
        return self.m.layers[0].adaLN_modulation[-1].in_features

    @property
    def output_names(self):
        return dit_output_names(self.first, self.last)

    def example_inputs(self, true_len=None, dim=None):
        """Dummy inputs of the exact exported shapes."""
        true_len = self.cap_slots if true_len is None else true_len
        dim = self.m.config.dim if dim is None else dim
        pos, attn, cap_pad = build_positions(true_len, self.cap_slots,
                                             self.grid_h, self.grid_w)
        pos = pos.unsqueeze(0)
        attn = attn.unsqueeze(0)
        if self.first:
            head = (torch.randn(1, self.C, self.H, self.W),
                    torch.tensor([1000.0]))
        else:
            # The incoming stream is n_img tokens wide until something
            # concatenates the caption onto it, and T wide afterwards.
            n_in = self.n_img if self.concat else self.n_img + self.cap_slots
            head = (torch.randn(1, n_in, dim), torch.randn(1, self.emb_dim))
        mid = (torch.randn(1, self.cap_slots, dim),) if self.concat else ()
        tail = ((pos,) if (self.has_refiner or self.has_blocks) else ())
        return head + mid + tail + ((attn,) if self.has_blocks else ())


class StaticZImageCaption(nn.Module):
    """The caption branch of part 1, as a standalone graph.

    Inputs `context` [1, S, cap_feat_dim] (the text encoder's hidden states),
    the shared `pos_ids` / `attn_mask` over the WHOLE sequence, and
    `cap_pad_mask` [1, S]; output `cap` [1, S, dim], ready to be concatenated
    onto the image tokens.

    pos_ids and attn_mask cover the whole sequence rather than just the caption
    slots so the runner builds one set of coordinates and hands the same two
    buffers to every graph. The slice is done here instead, where it is checked
    against the same n_img the rest of the export uses.
    """

    def __init__(self, model, cap_slots, latent_h, latent_w, patch=2, f_patch=1):
        super().__init__()
        self.m = model
        self.cap_slots = int(cap_slots)
        p = int(patch)
        self.grid_h, self.grid_w = int(latent_h) // p, int(latent_w) // p
        self.n_img = self.grid_h * self.grid_w

    def forward(self, context, pos_ids, attn_mask, cap_pad_mask):
        m = self.m
        freqs = m.rope_embedder(pos_ids.reshape(-1, 3)).unsqueeze(0)
        cap_freqs = freqs[:, self.n_img:]
        # Slice the FLOAT mask and compare afterwards, never the other way
        # round. QNN lowers a slice of a boolean tensor to StridedSlice on
        # Bool_8, which the HTP rejects outright:
        #     OpConfig validation failed for StridedSlice
        # and it only says so at context-binary generation, an hour in.
        cap_mask = attn_mask[:, self.n_img:] > 0.5

        cap = m.cap_embedder(context)
        cap = torch.where(cap_pad_mask.unsqueeze(-1) > 0.5, m.cap_pad_token, cap)
        for layer in m.context_refiner:
            cap = layer(cap, cap_mask, cap_freqs)
        return cap

    @property
    def input_names(self):
        return list(CAP_INPUT_NAMES)

    @property
    def output_names(self):
        return list(CAP_OUTPUT_NAMES)

    def example_inputs(self, true_len=None, dim=None):
        true_len = self.cap_slots if true_len is None else true_len
        pos, attn, cap_pad = build_positions(true_len, self.cap_slots,
                                             self.grid_h, self.grid_w)
        return (torch.randn(1, self.cap_slots, self.m.config.cap_feat_dim),
                pos.unsqueeze(0), attn.unsqueeze(0), cap_pad.unsqueeze(0))

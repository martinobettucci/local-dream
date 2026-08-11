"""Replace Z-Image's complex-valued RoPE with the equivalent real-valued form.

ONNX has no complex tensor type, so torch.polar / view_as_complex make the
model unexportable. The rotation is identical in real arithmetic:

    (x0 + i*x1) * (cos + i*sin) = (x0*cos - x1*sin) + i*(x0*sin + x1*cos)

freqs_cis carries a trailing dim of 2 holding (cos, sin) instead of complex64;
every other shape is unchanged.
"""
import os

import torch
from diffusers.models.transformers import transformer_z_image as tzi


def _precompute_freqs_cis_real(dim, end, theta: float = 256.0):
    with torch.device("cpu"):
        out = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            timestep = torch.arange(e, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()          # [e, d/2]
            out.append(torch.stack([freqs.cos(), freqs.sin()], dim=-1))  # [e, d/2, 2]
        return out


def _rope_call_real(self, ids: torch.Tensor):
    assert ids.ndim == 2 and ids.shape[-1] == len(self.axes_dims)
    device = ids.device
    if self.freqs_cis is None:
        self.freqs_cis = _precompute_freqs_cis_real(self.axes_dims, self.axes_lens, theta=self.theta)
    if self.freqs_cis[0].device != device:
        self.freqs_cis = [f.to(device) for f in self.freqs_cis]
    # cat over the head-dim axis, keeping the trailing (cos, sin) pair intact
    return torch.cat([self.freqs_cis[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-2)


# The additive mask value, and it is NOT just "a large negative".
#
# `scores + mask` is a real activation, and act_bitwidth 16 encodes it over the
# min/max observed during calibration -- where the mask IS observed, since
# make_calib.py feeds a genuine 0/1 attn_mask. So the mask value single-handedly
# sets the bottom of that tensor's encoding range, and every unmasked score has
# to share whatever resolution is left.
#
# Measured at the real shapes (head_dim 128, T 4608, 384 masked, q/k RMSNormed,
# asymmetric uint16 over observed min/max), as mean relative error on the
# attention output:
#
#     mask     quant step   error
#     -1e4       0.15276    4.40 %      <- range [-10004, +6]
#     -200       0.00324    0.09 %
#     -100       0.00171    0.05 %      <- range [-106, +6]
#
# An 85x precision loss on the most sensitive tensor in the block, repeated over
# 30 blocks and 8 steps, for no benefit: -100 masks just as totally. The mask
# only has to underflow the softmax, and with max|score| measured at 6-17,
# exp(-100 - 17) = 4e-51 against the softmax output's own 16-bit resolution of
# 1.5e-5. That is ~45 decades of margin. Do not "harden" this back toward -inf.
MASK_NEG = -100.0

# Heads per attention group; see the note in _processor_call. 0 means "all of
# them", i.e. the original single-shot attention. 5 divides the model's 30
# heads evenly and puts the live score tensor at 4608^2 x 5 x 2 = 212 MB.
ATTN_HEAD_CHUNK = int(os.environ.get("ZIMAGE_ATTN_HEAD_CHUNK", "5"))


def _apply_rotary_emb_real(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x_in: [B, seq, heads, head_dim]; freqs_cis: [B, seq, head_dim/2, 2]
    x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)   # [B, seq, heads, hd/2, 2]
    f = freqs_cis.unsqueeze(2)                          # [B, seq, 1, hd/2, 2]
    cos, sin = f[..., 0], f[..., 1]
    x0, x1 = x[..., 0], x[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.flatten(3).type_as(x_in)


def _processor_call(self, attn, hidden_states, encoder_hidden_states=None,
                    attention_mask=None, freqs_cis=None):
    query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
    key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
    value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    if freqs_cis is not None:
        query = _apply_rotary_emb_real(query, freqs_cis)
        key = _apply_rotary_emb_real(key, freqs_cis)

    dtype = query.dtype
    query, key = query.to(dtype), key.to(dtype)
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask[:, None, None, :]

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # Attention written out rather than F.scaled_dot_product_attention.
    #
    # SDPA with a *boolean* mask does not export to something the HTP can run:
    # torch's ONNX decomposition adds an IsNaN/Where guard for rows where every
    # key is masked (softmax of all -inf is NaN), and qnn-context-binary-generator
    # rejects it outright --
    #     validateNativeOps master op validator .../IsNaN:qti.aisw:IsNan failed 3110
    #     Input[0] has incorrect Datatype 0x416
    # after the DLC has already converted and quantized cleanly. Writing the
    # four steps out keeps the graph to MatMul / Add / Softmax.
    #
    # No NaN guard is needed here because no row is ever fully masked: this
    # runner's mask always keeps at least the image tokens, which come first.
    # An additive float mask, not a boolean one, for the same reason -- 0 where
    # attending is allowed, MASK_NEG where it is not.
    # ...and computed in GROUPS OF HEADS, not all thirty at once.
    #
    # The score tensor is [1, heads, T, T]. At T = 4608 and 30 heads that is
    # 4608^2 x 30 x 2 bytes = 1.27 GB live, and the HTP sizes a context to hold
    # it. Measured on device, by what loaded and what did not:
    #
    #   caption branch  T =  512   0.51 GB estimate   loads in 0.8 s
    #   part 1          T = 4096   1.50 GB estimate   loads in 4.5 s
    #   part 2          T = 4608   2.17 GB estimate   REFUSED
    #     "Failed to find available PD for contextId 1 ... with context size
    #      estimate 2171250944"
    #
    # So the ceiling sits between 1.5 and 2.17 GB, and the term that crosses it
    # is quadratic in sequence length. Nothing about the weights is the problem:
    # every part is the same 490 MB.
    #
    # ATTN_HEAD_CHUNK heads at a time cuts the live score tensor by that factor
    # while computing exactly the same result -- heads are independent all the
    # way from the q/k/v projections to the concatenation, so this is a
    # regrouping of the arithmetic, not an approximation. The cost is more,
    # smaller MatMuls; at 5 heads each is still 4608x4608x128, far above the
    # size where per-op overhead matters.
    scale = q.shape[-1] ** -0.5
    if attention_mask is not None and attention_mask.dtype == torch.bool:
        attention_mask = torch.where(
            attention_mask,
            torch.zeros((), dtype=q.dtype),
            torch.full((), MASK_NEG, dtype=q.dtype))

    n_heads = q.shape[1]
    chunk = ATTN_HEAD_CHUNK if ATTN_HEAD_CHUNK > 0 else n_heads
    parts = []
    for h0 in range(0, n_heads, chunk):
        h1 = min(h0 + chunk, n_heads)
        scores = torch.matmul(q[:, h0:h1], k[:, h0:h1].transpose(-2, -1)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask.to(scores.dtype)
        parts.append(torch.matmul(torch.softmax(scores, dim=-1), v[:, h0:h1]))
    hs = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    hs = hs.transpose(1, 2).flatten(2).type_as(query)
    return attn.to_out[0](hs)


def apply():
    tzi.RopeEmbedder.precompute_freqs_cis = staticmethod(_precompute_freqs_cis_real)
    tzi.RopeEmbedder.__call__ = _rope_call_real
    tzi.ZSingleStreamAttnProcessor.__call__ = _processor_call

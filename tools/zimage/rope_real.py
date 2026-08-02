"""Replace Z-Image's complex-valued RoPE with the equivalent real-valued form.

ONNX has no complex tensor type, so torch.polar / view_as_complex make the
model unexportable. The rotation is identical in real arithmetic:

    (x0 + i*x1) * (cos + i*sin) = (x0*cos - x1*sin) + i*(x0*sin + x1*cos)

freqs_cis carries a trailing dim of 2 holding (cos, sin) instead of complex64;
every other shape is unchanged.
"""
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
    # attending is allowed, a large negative where it is not, and -1e4 rather
    # than -inf so 16-bit activation quantization has a finite range to encode.
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.where(
                attention_mask,
                torch.zeros((), dtype=scores.dtype),
                torch.full((), -1e4, dtype=scores.dtype))
        scores = scores + attention_mask.to(scores.dtype)
    hs = torch.matmul(torch.softmax(scores, dim=-1), v)
    hs = hs.transpose(1, 2).flatten(2).type_as(query)
    return attn.to_out[0](hs)


def apply():
    tzi.RopeEmbedder.precompute_freqs_cis = staticmethod(_precompute_freqs_cis_real)
    tzi.RopeEmbedder.__call__ = _rope_call_real
    tzi.ZSingleStreamAttnProcessor.__call__ = _processor_call

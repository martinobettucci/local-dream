#!/usr/bin/env python3
"""Proves the static-shape export is faithful, on a tiny random-weight model.

A QNN graph must fix the caption at 512 slots, but the reference pipeline feeds
a variable-length caption (padded only up to a multiple of 32). This checks that
the two agree anyway -- provided the graph applies an attention mask in BOTH
places the reference gets away without one:

  1. the caption refiner (context_refiner self-attention), and
  2. the main transformer blocks (the unified [image, caption] sequence)

masking off everything past cap_len = ceil(true_len/32)*32.

With both masks the agreement is exact. With only (2) it is not -- roughly 3.6%
mean relative error on a 12-token prompt -- which is why this is worth a test
rather than a comment. Run:

    python tools/zimage/verify_static_equivalence.py
"""
import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rope_real  # noqa: E402

rope_real.apply()
from diffusers import ZImageTransformer2DModel
from diffusers.models.transformers import transformer_z_image as tzi

S_STATIC = 512
STATE = {"cap_len": None, "img_len": None}

def build():
    torch.manual_seed(0)
    return ZImageTransformer2DModel(
        all_patch_size=[2], all_f_patch_size=[1], in_channels=16,
        dim=64, n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2,
        norm_eps=1e-5, qk_norm=True, cap_feat_dim=32,
        rope_theta=256.0, t_scale=1000.0,
        axes_dims=[8, 12, 12], axes_lens=[1536, 512, 512],
    ).eval()

model = build()
orig_patch = tzi.ZImageTransformer2DModel.patchify_and_embed
orig_unified = tzi.ZImageTransformer2DModel._build_unified_sequence
orig_prepare = tzi.ZImageTransformer2DModel._prepare_sequence

def masked_prepare(self, feats, pos_ids, inner_pad_mask, pad_token, noise_mask=None, device=None):
    out = orig_prepare(self, feats, pos_ids, inner_pad_mask, pad_token, noise_mask, device)
    seq, freqs, attn, seqlens, nm = out
    # Only the caption call carries S_STATIC slots; mask the ones the reference
    # would never have created.
    if seq.shape[1] == S_STATIC and STATE["cap_len"] is not None:
        m = torch.zeros((seq.shape[0], S_STATIC), dtype=torch.bool, device=seq.device)
        m[:, :STATE["cap_len"]] = True
        attn = m
    return seq, freqs, attn, seqlens, nm

def static_patchify(self, all_image, all_cap_feats, patch_size, f_patch_size):
    device = all_image[0].device
    img_o, img_s, img_p, img_m, cap_o, cap_p, cap_m = [], [], [], [], [], [], []
    for image, cap_feat in zip(all_image, all_cap_feats):
        n = len(cap_feat)
        cap_len = math.ceil(n / tzi.SEQ_MULTI_OF) * tzi.SEQ_MULTI_OF
        STATE["cap_len"] = cap_len
        feat = torch.cat([cap_feat, cap_feat[-1:].repeat(S_STATIC - n, 1)], 0)
        pos = torch.zeros(S_STATIC, 3, dtype=torch.int32, device=device)
        pos[:cap_len, 0] = torch.arange(1, cap_len + 1, dtype=torch.int32, device=device)
        inner = torch.ones(S_STATIC, dtype=torch.bool, device=device)  # True -> pad_token
        inner[:n] = False
        cap_o.append(feat); cap_p.append(pos); cap_m.append(inner)

        patches, size, (F_t, H_t, W_t) = self._patchify_image(image, patch_size, f_patch_size)
        o, p, m, tot, _ = self._pad_with_ids(patches, (F_t, H_t, W_t), (cap_len + 1, 0, 0), device)
        STATE["img_len"] = tot
        img_o.append(o); img_s.append(size); img_p.append(p); img_m.append(m)
    return img_o, cap_o, img_s, img_p, cap_p, img_m, cap_m

def masked_unified(self, *a, **kw):
    unified, freqs, _, noise = orig_unified(self, *a, **kw)
    T = unified.shape[1]
    keep = STATE["img_len"] + STATE["cap_len"]          # [image | real caption]
    m = torch.zeros((unified.shape[0], T), dtype=torch.bool, device=unified.device)
    m[:, :keep] = True
    return unified, freqs, m, noise

torch.manual_seed(1)
x = [torch.randn(16, 1, 16, 16)]
t = torch.tensor([1000.0])

failures = []
print(f"static caption slots = {S_STATIC}, masking refiner + unified\n")
for n in (12, 40, 100, 300):
    cap = [torch.randn(n, 32)]
    tzi.ZImageTransformer2DModel.patchify_and_embed = orig_patch
    tzi.ZImageTransformer2DModel._build_unified_sequence = orig_unified
    tzi.ZImageTransformer2DModel._prepare_sequence = orig_prepare
    with torch.no_grad():
        r = model(x, t, cap, return_dict=False)[0]
    r = r[0] if isinstance(r, list) else r

    tzi.ZImageTransformer2DModel.patchify_and_embed = static_patchify
    tzi.ZImageTransformer2DModel._build_unified_sequence = masked_unified
    tzi.ZImageTransformer2DModel._prepare_sequence = masked_prepare
    with torch.no_grad():
        g = model(x, t, cap, return_dict=False)[0]
    g = g[0] if isinstance(g, list) else g

    d = (r - g).abs().max().item()
    rel = (r - g).abs().mean().item() / r.abs().mean().item()
    ok = d < 1e-4
    failures.append(not ok)
    print(f"prompt {n:4d} tok (cap_len {math.ceil(n / 32) * 32:4d}): max {d:.3e}  "
          f"mean rel {rel * 100:7.4f}%  {'EQUIVALENT' if ok else 'DIFFERS'}")

tzi.ZImageTransformer2DModel.patchify_and_embed = orig_patch
tzi.ZImageTransformer2DModel._build_unified_sequence = orig_unified
tzi.ZImageTransformer2DModel._prepare_sequence = orig_prepare

raise SystemExit(1 if any(failures) else 0)

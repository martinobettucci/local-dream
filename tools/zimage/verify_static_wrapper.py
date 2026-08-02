#!/usr/bin/env python3
"""Check StaticZImageDiT against the stock model, whole and split.

Runs on a tiny random-weight model in seconds — no checkpoint needed. Three
things are checked, in increasing order of what they would break:

  1. single-piece wrapper == reference (the static rewrite is faithful)
  2. an N-way split chain == the single piece (the cut is lossless)
  3. every piece exports to ONNX and onnxruntime reproduces torch

    python tools/zimage/verify_static_wrapper.py
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rope_real  # noqa: E402

rope_real.apply()

from diffusers import ZImageTransformer2DModel  # noqa: E402

from static_export import StaticZImageDiT, build_positions  # noqa: E402

CAP_SLOTS = 512
LAT = 16          # latent edge -> 8x8 = 64 image tokens (a multiple of 32)
DIM = 384        # > 256 so dim != emb_dim, as in the real model
N_LAYERS = 6
T_SCALE = 1000.0


def build():
    torch.manual_seed(0)
    return ZImageTransformer2DModel(
        all_patch_size=[2], all_f_patch_size=[1], in_channels=16,
        dim=DIM, n_layers=N_LAYERS, n_refiner_layers=1, n_heads=3, n_kv_heads=3,
        norm_eps=1e-5, qk_norm=True, cap_feat_dim=32,
        rope_theta=256.0, t_scale=T_SCALE,
        # head_dim = 384/3 = 128 = 32+48+48, the real model's RoPE split
        axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512],
    ).eval()


def reference(model, sample, sigma, cap_feat):
    """Reference output as (1, C, H, W).

    unpatchify returns a list of (C, F, H, W) — video-shaped, F = 1 for stills.
    Dropping F rather than unsqueezing is the difference between comparing the
    right elements and silently broadcasting into a 5-D tensor.
    """
    with torch.no_grad():
        out = model([sample.squeeze(0)], torch.tensor([sigma]), [cap_feat],
                    return_dict=False)[0]
    out = out[0] if isinstance(out, list) else out
    return out.reshape(1, out.shape[0], out.shape[-2], out.shape[-1])


def static_inputs(true_len, cap_feat_dim, sigma, sample):
    pos, attn, cap_pad = build_positions(true_len, CAP_SLOTS, LAT // 2, LAT // 2)
    context = torch.zeros(1, CAP_SLOTS, cap_feat_dim)
    context[0, :true_len] = cap_feat
    return (sample, torch.tensor([sigma * T_SCALE]), context,
            pos.unsqueeze(0), attn.unsqueeze(0), cap_pad.unsqueeze(0))


def report(tag, a, b, tol=1e-4):
    d = (a - b).abs().max().item()
    ok = d < tol
    print(f"  {tag:<46s} max abs {d:.3e}  {'OK' if ok else '*** FAIL ***'}")
    return ok


if __name__ == "__main__":
    ok = True
    model = build()
    torch.manual_seed(1)
    sample = torch.randn(1, 16, LAT, LAT)
    sigma = 0.7

    whole_probe = StaticZImageDiT(model, CAP_SLOTS, LAT, LAT)
    assert whole_probe.emb_dim != DIM, "test config must keep dim and emb_dim distinct"
    print(f"tiny model: dim={DIM} emb_dim={whole_probe.emb_dim} layers={N_LAYERS} "
          f"cap_slots={CAP_SLOTS} latent={LAT}x{LAT} "
          f"({(LAT // 2) ** 2} image tokens)\n")

    print("1. single-piece wrapper vs reference")
    whole = StaticZImageDiT(model, CAP_SLOTS, LAT, LAT)
    for true_len in (12, 40, 100, 300):
        cap_feat = torch.randn(true_len, 32)
        ref = reference(model, sample.reshape(16, 1, LAT, LAT), sigma, cap_feat)
        args = static_inputs(true_len, 32, sigma, sample)
        with torch.no_grad():
            got = whole(*args)
        ok &= report(f"prompt {true_len} tok", ref, got)

    print("\n2. split chain vs single piece")
    cap_feat = torch.randn(40, 32)
    args = static_inputs(40, 32, sigma, sample)
    with torch.no_grad():
        single = whole(*args)
    for cuts in ([0, 2, 4, N_LAYERS], [0, 1, 2, 3, 4, 5, N_LAYERS], [0, N_LAYERS]):
        parts = [StaticZImageDiT(model, CAP_SLOTS, LAT, LAT, a, b)
                 for a, b in zip(cuts[:-1], cuts[1:])]
        with torch.no_grad():
            out = parts[0](*args)
            if len(parts) == 1:
                chained = out
            else:
                hidden, emb = out
                for p in parts[1:]:
                    r = p(hidden, emb, args[3], args[4])
                    if p.last:
                        chained = r
                    else:
                        hidden = r
        ok &= report(f"{len(parts)}-way split {cuts}", single, chained)

    print("\n3. ONNX export + onnxruntime agreement")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime missing, skipping")
        ort = None

    if ort is not None:
        cuts = [0, 3, N_LAYERS]
        parts = [StaticZImageDiT(model, CAP_SLOTS, LAT, LAT, a, b)
                 for a, b in zip(cuts[:-1], cuts[1:])]
        with tempfile.TemporaryDirectory() as td:
            hidden = emb = None
            for i, part in enumerate(parts):
                inputs = (args if part.first else (hidden, emb, args[3], args[4]))
                path = os.path.join(td, f"part{i + 1}.onnx")
                torch.onnx.export(
                    part, inputs, path, opset_version=17, dynamo=False,
                    input_names=part.input_names, output_names=part.output_names)
                with torch.no_grad():
                    ref_out = part(*inputs)
                sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                feed = {n: v.numpy() for n, v in zip(part.input_names, inputs)}
                got = sess.run(None, feed)
                refs = ref_out if isinstance(ref_out, tuple) else (ref_out,)
                for name, r, g in zip(part.output_names, refs, got):
                    ok &= report(f"part{i + 1} onnx '{name}'", r, torch.from_numpy(g), 1e-3)
                if not part.last:
                    hidden, emb = refs if len(refs) == 2 else (refs[0], emb)
            print(f"  exported {len(parts)} parts, all inputs/outputs named per contract")

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** FAILURES ABOVE ***"))
    raise SystemExit(0 if ok else 1)

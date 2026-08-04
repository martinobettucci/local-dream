#!/usr/bin/env python3
"""Check StaticZImageDiT against the stock model, whole and split.

Runs on a tiny random-weight model in seconds — no checkpoint needed. Three
things are checked, in increasing order of what they would break:

  1. caption branch + single-piece wrapper == reference (the static rewrite is
     faithful, and splitting the caption out changes no arithmetic)
  2. an N-way split chain == the single piece (the cut is lossless), including
     the block-less part 1 the shipped 31-way split uses
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

from static_export import (StaticZImageCaption, StaticZImageDiT,  # noqa: E402
                           build_positions)

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


def raw_inputs(true_len, cap_feat, cap_feat_dim, sigma, sample):
    """What the runner builds per prompt, before any graph has run."""
    pos, attn, cap_pad = build_positions(true_len, CAP_SLOTS, LAT // 2, LAT // 2)
    context = torch.zeros(1, CAP_SLOTS, cap_feat_dim)
    context[0, :true_len] = cap_feat
    return (sample, torch.tensor([sigma * T_SCALE]), context,
            pos.unsqueeze(0), attn.unsqueeze(0), cap_pad.unsqueeze(0))


def dit_args(part, capgraph, raw):
    """The caption branch's output plus whatever part 1's contract asks for."""
    sample, timestep, context, pos, attn, cap_pad = raw
    with torch.no_grad():
        cap = capgraph(context, pos, attn, cap_pad)
    head = (sample, timestep, cap, pos)
    return head + ((attn,) if part.has_blocks else ())


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

    capgraph = StaticZImageCaption(model, CAP_SLOTS, LAT, LAT)
    whole_probe = StaticZImageDiT(model, CAP_SLOTS, LAT, LAT)
    assert whole_probe.emb_dim != DIM, "test config must keep dim and emb_dim distinct"
    print(f"tiny model: dim={DIM} emb_dim={whole_probe.emb_dim} layers={N_LAYERS} "
          f"cap_slots={CAP_SLOTS} latent={LAT}x{LAT} "
          f"({(LAT // 2) ** 2} image tokens)\n")

    print("1. caption branch + single-piece wrapper vs reference")
    whole = StaticZImageDiT(model, CAP_SLOTS, LAT, LAT)
    for true_len in (12, 40, 100, 300):
        cap_feat = torch.randn(true_len, 32)
        ref = reference(model, sample.reshape(16, 1, LAT, LAT), sigma, cap_feat)
        raw = raw_inputs(true_len, cap_feat, 32, sigma, sample)
        with torch.no_grad():
            got = whole(*dit_args(whole, capgraph, raw))
        ok &= report(f"prompt {true_len} tok", ref, got)

    print("\n2. split chain vs single piece")
    cap_feat = torch.randn(40, 32)
    raw = raw_inputs(40, cap_feat, 32, sigma, sample)
    with torch.no_grad():
        single = whole(*dit_args(whole, capgraph, raw))
    # The last plan is the shape the shipped build uses: part 1 owns no block,
    # so `has_blocks` is false and it declares no attn_mask. It is checked here
    # rather than only in production because a block-less part is exactly the
    # case an off-by-one in the plan would produce silently.
    plans = ([0, 2, 4, N_LAYERS],
             [0, 1, 2, 3, 4, 5, N_LAYERS],
             [0, N_LAYERS],
             [0, 0, 1, 2, 3, 4, 5, N_LAYERS])
    for cuts in plans:
        parts = [StaticZImageDiT(model, CAP_SLOTS, LAT, LAT, a, b,
                                 first=(i == 0), last=(i == len(cuts) - 2))
                 for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]))]
        args = dit_args(parts[0], capgraph, raw)
        with torch.no_grad():
            out = parts[0](*args)
            if len(parts) == 1:
                chained = out
            else:
                hidden, emb = out
                for p in parts[1:]:
                    r = p(hidden, emb, raw[3], raw[4])
                    if p.last:
                        chained = r
                    else:
                        hidden = r
        blockless = " (block-less part 1)" if cuts[0] == cuts[1] else ""
        ok &= report(f"{len(parts)}-way split {cuts}{blockless}", single, chained)

    print("\n3. ONNX export + onnxruntime agreement")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime missing, skipping")
        ort = None

    if ort is not None:
        cuts = [0, 0, 3, N_LAYERS]
        graphs = [capgraph] + [
            StaticZImageDiT(model, CAP_SLOTS, LAT, LAT, a, b,
                            first=(i == 0), last=(i == len(cuts) - 2))
            for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]))]
        with tempfile.TemporaryDirectory() as td:
            hidden = emb = None
            for i, part in enumerate(graphs):
                if part is capgraph:
                    inputs = (raw[2], raw[3], raw[4], raw[5])
                elif getattr(part, "first", False):
                    inputs = dit_args(part, capgraph, raw)
                else:
                    inputs = (hidden, emb, raw[3], raw[4])
                path = os.path.join(td, f"graph{i}.onnx")
                torch.onnx.export(
                    part, inputs, path, opset_version=17, dynamo=False,
                    input_names=part.input_names, output_names=part.output_names)
                with torch.no_grad():
                    ref_out = part(*inputs)
                sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                feed = {n: v.numpy() for n, v in zip(part.input_names, inputs)}
                got = sess.run(None, feed)
                refs = ref_out if isinstance(ref_out, tuple) else (ref_out,)
                tag = "cap" if part is capgraph else f"part{i}"
                for name, r, g in zip(part.output_names, refs, got):
                    ok &= report(f"{tag} onnx '{name}'", r, torch.from_numpy(g), 1e-3)
                if part is not capgraph and not part.last:
                    hidden, emb = refs if len(refs) == 2 else (refs[0], emb)
            print(f"  exported {len(graphs)} graphs, all inputs/outputs named "
                  f"per contract")

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** FAILURES ABOVE ***"))
    raise SystemExit(0 if ok else 1)

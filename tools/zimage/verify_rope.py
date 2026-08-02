#!/usr/bin/env python3
"""Check that the real-valued RoPE patch is a no-op numerically, and that it is
what unblocks ONNX export.

Runs on a tiny random-weight model — no checkpoint download needed, a few
seconds on CPU. Run this first after bumping diffusers: if the patch stops
matching, the export path is wrong before any weights are involved.

    python tools/zimage/verify_rope.py

Expected: max |complex - real| at float32 noise level (~1e-6), and the export
failing on `pad_sequence` rather than on `ComplexFloat` — see
docs/zimage.md §9 for why that second failure is the expected next step and
not a regression.
"""
import sys

import torch
from diffusers import ZImageTransformer2DModel

sys.path.insert(0, __file__.rsplit("/", 1)[0])


def build():
    # head_dim = dim / n_heads = 32 = sum(axes_dims), as the real config also
    # satisfies (3840/30 = 128 = 32+48+48).
    torch.manual_seed(0)
    return ZImageTransformer2DModel(
        all_patch_size=[2],
        all_f_patch_size=[1],
        in_channels=16,
        dim=64,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=32,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[8, 12, 12],
        axes_lens=[64, 64, 64],
    ).eval()


def main():
    torch.manual_seed(1)
    # Latents are (C, F, H, W) — video-shaped, F = 1 for stills, which is what
    # the three RoPE axes (t, h, w) index.
    x = [torch.randn(16, 1, 16, 16)]
    cap = [torch.randn(12, 32)]
    t = torch.tensor([1000.0])

    ref_model = build()
    with torch.no_grad():
        ref = ref_model(x, t, cap, return_dict=False)[0]
    ref = ref[0] if isinstance(ref, list) else ref

    import rope_real

    rope_real.apply()
    new_model = build()
    new_model.load_state_dict(ref_model.state_dict())
    with torch.no_grad():
        got = new_model(x, t, cap, return_dict=False)[0]
    got = got[0] if isinstance(got, list) else got

    delta = (ref - got).abs().max().item()
    print(f"max |complex - real| = {delta:.3e}  shape={tuple(got.shape)}")
    if delta >= 1e-4:
        print("*** MISMATCH: the real-valued rotation no longer matches ***")
        return 1
    print("numerically equivalent")

    print("\n--- ONNX export ---")
    try:
        torch.onnx.export(new_model, (x, t, cap), "/tmp/zimage_probe.onnx",
                          opset_version=17, dynamo=False)
        print("export OK (a static forward wrapper is evidently in place)")
    except Exception as exc:  # noqa: BLE001 - the message is the result here
        msg = str(exc)
        if "ComplexFloat" in msg:
            print("*** REGRESSION: still hitting complex tensors ***")
            return 1
        if "pad_sequence" in msg:
            print("expected: blocked on pad_sequence (the variable-length list API),")
            print("not on complex tensors. The RoPE patch did its job.")
            return 0
        print(f"unexpected failure: {type(exc).__name__}: {msg[:300]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

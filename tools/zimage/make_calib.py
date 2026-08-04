#!/usr/bin/env python3
"""Build calibration inputs for qairt-quantizer that look like real runtime data.

This matters more than it sounds. The quantizer derives every activation's
min/max by running the graph on these tensors, so an input whose calibration
distribution does not match runtime produces an encoding that clips or wastes
its whole range -- and the result is a model that loads, runs, and generates
garbage, discovered only after all 30 parts are built.

Filling every float input with N(0,1) is wrong for four of the DiT's six:

  timestep      sigma * 1000, so roughly 1000 down to 3 over the 8 steps.
                Calibrating it at N(0,1) sets the range about three orders of
                magnitude too small, and everything derived from it saturates.
  pos_ids       real 3D RoPE coordinates (t, h, w). Zeros mean every token sits
                at position 0, where cos = 1 and sin = 0, so the rotation is the
                identity and no downstream tensor ever sees its true range.
  attn_mask     a 0/1 indicator. Noise around zero masks roughly half the
                sequence at random and leaves the rest partially attenuated.
  cap_pad_mask  likewise 0/1.

Only `sample` (latents), `hidden_in` and `cap` (both RMS-normed activations),
`emb` and `context` are legitimately near-unit-Gaussian.

    python make_calib.py <onnx> <out_dir> <list_file> [--true-len 128]

Writes one .raw per graph input plus the input_list file qairt-quantizer wants.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Latent grid at 1024x1024 with patch 2: 128/2 = 64, so 4096 image tokens.
GRID = 64
CAP_SLOTS = 512

# The largest timestep the 8-step schedule produces. A single calibration sample
# can only pin one value per input, and the quantizer's range for a tensor spans
# what it observed -- so give it the top of the schedule, which makes the range
# cover everything below it too. Calibrating at the *small* end would clip every
# early step, which is where the image is actually decided.
TIMESTEP_MAX = 1000.0


def positions(true_len):
    """Real pos_ids / attn_mask / cap_pad_mask, from the same code the export
    and the C++ runtime both mirror."""
    from static_export import build_positions

    pos, attn, cap_pad = build_positions(true_len, CAP_SLOTS, GRID, GRID)
    return (pos.numpy().astype(np.int32),
            attn.numpy().astype(np.float32),
            cap_pad.numpy().astype(np.float32))


def make(onnx_path, out_dir, list_path, true_len=128, seed=0):
    import onnx

    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(onnx_path, load_external_data=False)
    rng = np.random.default_rng(seed)
    pos, attn, cap_pad = positions(true_len)

    entries, described = [], []
    for inp in model.graph.input:
        name = inp.name
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        path = os.path.abspath(os.path.join(out_dir, f"{name}.raw"))

        if name == "pos_ids":
            data, how = pos.reshape(dims), "real RoPE coordinates"
        elif name == "attn_mask":
            data, how = attn.reshape(dims), f"1 up to cap_len, 0 beyond"
        elif name == "cap_pad_mask":
            data, how = cap_pad.reshape(dims), "1 past the true prompt length"
        elif name == "timestep":
            data = np.full(dims, TIMESTEP_MAX, dtype=np.float32)
            how = f"sigma*1000 at the top of the schedule ({TIMESTEP_MAX:g})"
        elif name == "attention_mask":
            # Text encoder: right-padded, 1 for real tokens.
            data = np.zeros(dims, dtype=np.float32)
            data[..., :true_len] = 1.0
            how = f"1 for the first {true_len} tokens, 0 padding"
        else:
            # sample, hidden_in, cap, emb, context, input_embedding -- all
            # genuinely near-unit-Gaussian at runtime.
            data, how = rng.standard_normal(dims, dtype=np.float32), "N(0,1)"

        if inp.type.tensor_type.elem_type == onnx.TensorProto.INT32:
            data = data.astype(np.int32)
        else:
            data = data.astype(np.float32)
        if list(data.shape) != dims:
            raise RuntimeError(f"{name}: built {data.shape}, graph wants {dims}")
        data.tofile(path)
        entries.append(f"{name}:={path}")
        described.append(f"{name} {tuple(dims)} <- {how}")

    with open(list_path, "w") as f:
        f.write(" ".join(entries) + "\n")
    for d in described:
        print(f"  calib {d}")
    return list_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("out_dir")
    ap.add_argument("list_file")
    ap.add_argument("--true-len", type=int, default=128,
                    help="prompt length to calibrate at; picks cap_len and the "
                         "mask boundaries")
    args = ap.parse_args()
    make(args.onnx, args.out_dir, args.list_file, args.true_len)


if __name__ == "__main__":
    main()

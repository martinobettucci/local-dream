#!/usr/bin/env python3
"""Export the real Z-Image DiT to per-part ONNX graphs.

Constraints this is built around:

  * The released transformer is fp32 across three shards, 24.6 GB. It cannot be
    held in RAM (15 GB here) and barely fits on disk, so shards are fetched one
    at a time, split into per-part fp16 weight files, and deleted.
  * Each part is then built as a REDUCED ZImageTransformer2DModel holding only
    its own blocks, so peak RAM is one part rather than the whole model.
  * Every stage is skipped if its output already exists, so the script is
    resumable — which matters when a full run is measured in hours.

    python tools/zimage/export_dit.py --work /path/to/scratch --parts 8

Produces work/onnx/unet_partK.onnx plus a manifest. Feeding those to
qairt-converter / qnn-context-binary-generator is convert_qnn.py's job.
"""
import argparse
import gc
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "Tongyi-MAI/Z-Image-Turbo"
SUBDIR = "transformer"
N_LAYERS = 30
CAP_SLOTS = 512
LATENT = 128          # 1024 / 8
DIM = 3840
CAP_FEAT_DIM = 2560


def log(msg):
    print(f"[export_dit] {msg}", flush=True)


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def plan_parts(n_parts, n_layers=N_LAYERS):
    """Contiguous, near-equal block ranges. Part 1 and part N carry the extra
    embedder / final-layer work, so they get one fewer block where it divides
    unevenly."""
    base, extra = divmod(n_layers, n_parts)
    cuts, at = [], 0
    for i in range(n_parts):
        # hand the remainder to the middle parts, which do the least besides blocks
        take = base + (1 if 0 < i < n_parts - 1 and extra > 0 else 0)
        if 0 < i < n_parts - 1 and extra > 0:
            extra -= 1
        cuts.append((at, at + take))
        at += take
    if at != n_layers:            # remainder left over: give it to the last part
        cuts[-1] = (cuts[-1][0], n_layers)
    return cuts


def shard_to_parts(work, cuts):
    """Stream the three shards into per-part fp16 weight files.

    Non-block tensors (embedders, refiners, final layer, pad tokens) go to the
    first or last part as appropriate. Block tensors are renumbered so each part
    sees its layers as 0..k.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    wdir = os.path.join(work, "parts")
    os.makedirs(wdir, exist_ok=True)
    done = os.path.join(wdir, ".complete")
    if os.path.exists(done):
        log("per-part weights already built, skipping shard pass")
        return wdir

    index = hf_hub_download(REPO, f"{SUBDIR}/diffusion_pytorch_model.safetensors.index.json")
    weight_map = json.load(open(index))["weight_map"]
    shards = sorted(set(weight_map.values()))
    log(f"{len(weight_map)} tensors across {len(shards)} shards")

    n_parts = len(cuts)

    def target_part(name):
        m = re.match(r"layers\.(\d+)\.(.*)", name)
        if m:
            layer, rest = int(m.group(1)), m.group(2)
            for pi, (a, b) in enumerate(cuts):
                if a <= layer < b:
                    return pi, f"layers.{layer - a}.{rest}"
            raise KeyError(f"layer {layer} outside {cuts}")
        if name.startswith("all_final_layer"):
            return n_parts - 1, name
        # everything else (embedders, refiners, t_embedder, pad tokens) is part 1
        return 0, name

    # How many tensors each part expects, so a part can be flushed to disk the
    # moment it is complete. Holding all of them would mean ~12 GB of fp16 in
    # RAM against 15 GB total; layers are contiguous and shards are ordered, so
    # in practice only a part or two is ever pending.
    expected = [0] * n_parts
    for name in weight_map:
        expected[target_part(name)[0]] += 1
    log(f"tensors per part: {expected}")

    acc = [dict() for _ in range(n_parts)]
    written = [False] * n_parts

    def flush_complete():
        for pi in range(n_parts):
            if written[pi] or len(acc[pi]) != expected[pi]:
                continue
            out = os.path.join(wdir, f"part{pi + 1}.safetensors")
            save_file(acc[pi], out)
            log(f"  part{pi + 1} complete: {len(acc[pi])} tensors -> "
                f"{os.path.getsize(out) / 1e9:.2f} GB")
            acc[pi].clear()
            written[pi] = True
            gc.collect()

    for shard in shards:
        log(f"fetching {shard} (free {free_gb(work):.1f} GB)")
        path = hf_hub_download(REPO, f"{SUBDIR}/{shard}")
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                pi, newname = target_part(name)
                acc[pi][newname] = f.get_tensor(name).to(torch.float16)
        # hf_hub_download hands back a path inside the shared cache; drop the
        # blob so the next shard has room.
        real = os.path.realpath(path)
        if os.path.exists(real):
            os.remove(real)
        if os.path.islink(path):
            os.remove(path)
        gc.collect()
        flush_complete()
        log(f"  released {shard} (free {free_gb(work):.1f} GB)")

    if not all(written):
        raise RuntimeError(f"parts never completed: "
                           f"{[i + 1 for i, w in enumerate(written) if not w]}")
    open(done, "w").close()
    return wdir


def build_part(weights_path, n_blocks, first, last):
    """A reduced model holding only this part's blocks, then the static wrapper."""
    import rope_real
    rope_real.apply()
    from diffusers import ZImageTransformer2DModel
    from safetensors.torch import load_file

    from static_export import StaticZImageDiT

    model = ZImageTransformer2DModel(
        all_patch_size=[2], all_f_patch_size=[1], in_channels=16,
        dim=DIM, n_layers=n_blocks, n_refiner_layers=2, n_heads=30, n_kv_heads=30,
        norm_eps=1e-5, qk_norm=True, cap_feat_dim=CAP_FEAT_DIM,
        rope_theta=256.0, t_scale=1000.0,
        axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512],
    )
    sd = load_file(weights_path)
    sd = {k: v.to(torch.float32) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # A middle part legitimately lacks embedders and the final layer; those
    # modules exist on the reduced model but are never traced.
    missing = [m for m in missing if not m.startswith("layers.")]
    if unexpected:
        raise RuntimeError(f"unexpected tensors for {weights_path}: {unexpected[:5]}")
    if first and any(m.startswith(("all_x_embedder", "cap_embedder", "t_embedder",
                                   "noise_refiner", "context_refiner")) for m in missing):
        raise RuntimeError(f"part 1 is missing embedder weights: {missing[:5]}")
    if last and any(m.startswith("all_final_layer") for m in missing):
        raise RuntimeError(f"last part is missing the final layer: {missing[:5]}")
    model.eval()
    return StaticZImageDiT(model, CAP_SLOTS, LATENT, LATENT,
                           block_start=0, block_end=n_blocks,
                           first=first, last=last)


def export_part(part, path, dim=DIM):
    inputs = part.example_inputs(true_len=CAP_SLOTS, dim=dim)
    torch.onnx.export(part, inputs, path, opset_version=17, dynamo=False,
                      input_names=part.input_names, output_names=part.output_names)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--only", type=int, default=0, help="export just this part (1-based)")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    cuts = plan_parts(args.parts)
    log(f"{args.parts} parts, block ranges {cuts}")

    wdir = shard_to_parts(args.work, cuts)
    odir = os.path.join(args.work, "onnx")
    os.makedirs(odir, exist_ok=True)

    manifest = []
    for pi, (a, b) in enumerate(cuts):
        n = pi + 1
        if args.only and n != args.only:
            continue
        # Each part gets its own directory: parts above 2 GB are written in
        # ONNX external-data format, whose side files are named after the
        # tensors ("onnx__MatMul_1034") with no part prefix — two parts sharing
        # a directory would overwrite each other's weights.
        pdir = os.path.join(odir, f"part{n}")
        os.makedirs(pdir, exist_ok=True)
        out = os.path.join(pdir, f"unet_part{n}.onnx")
        if os.path.exists(out):
            log(f"part{n} already exported, skipping")
            manifest.append({"part": n, "blocks": [a, b], "onnx": out})
            continue
        log(f"building part{n} (blocks {a}..{b}, free {free_gb(args.work):.1f} GB)")
        part = build_part(os.path.join(wdir, f"part{n}.safetensors"),
                          b - a, first=(pi == 0), last=(pi == len(cuts) - 1))
        log(f"  exporting -> {out}")
        export_part(part, out)
        size = sum(os.path.getsize(os.path.join(pdir, f)) for f in os.listdir(pdir))
        log(f"  part{n} onnx {size / 1e9:.2f} GB")
        manifest.append({"part": n, "blocks": [a, b], "onnx": out,
                         "inputs": part.input_names, "outputs": part.output_names})
        del part
        gc.collect()

    mpath = os.path.join(args.work, "dit_manifest.json")
    json.dump({"parts": manifest, "cap_slots": CAP_SLOTS, "latent": LATENT},
              open(mpath, "w"), indent=2)
    log(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()

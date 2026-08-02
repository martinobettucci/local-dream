#!/usr/bin/env python3
"""Export the real Z-Image DiT to per-part ONNX graphs.

Constraints this is built around:

  * The released transformer is fp32 across three shards, 24.6 GB. Earlier
    versions of this script downloaded each shard, sliced it into per-part
    weight files and deleted it — which needs ~12 GB of disk for the parts
    alone, and re-downloads a shard for every part that touches it. Instead the
    weights are now read straight out of the remote safetensors with HTTP range
    requests (see remote_safetensors.py): one part costs its own size in
    transfer and nothing on disk.
  * Each part is built as a REDUCED ZImageTransformer2DModel holding only its
    own blocks, with the modules it does not trace deleted outright, so peak RAM
    is one part rather than the whole model.
  * Every stage is skipped if its output already exists, so the script is
    resumable — which matters when a full run is measured in hours.

    python tools/zimage/export_dit.py --work /path/to/scratch --parts 30 --only 4

The part count is free to choose. It trades peak memory during quantization —
which is what actually caps this, `qairt-quantizer` was OOM-killed at 19 GB on a
4-block part — against per-part fixed overhead and the number of context
switches per step on device. See docs/zimage.md.
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

# Modules a part only needs when it owns the head or the tail of the model.
FIRST_ONLY = ("all_x_embedder", "cap_embedder", "t_embedder",
              "noise_refiner", "context_refiner")
LAST_ONLY = ("all_final_layer",)

# ...and the subset actually worth deleting on a middle part. The two refiner
# stacks are two full-width blocks each, so at dim 3840 they are a couple of GB
# of randomly-initialised fp32 that a middle part would otherwise carry all the
# way through ONNX export. The rest are small, and deleting them only creates
# ways for code that legitimately reads their shapes to fail: StaticZImageDiT
# needs t_embedder's output width even on a part that never runs it, because
# `emb` is one of that part's graph inputs.
DROP_ON_MIDDLE = ("noise_refiner", "context_refiner")


def log(msg):
    print(f"[export_dit] {msg}", flush=True)


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def plan_parts(n_parts, n_layers=N_LAYERS):
    """Contiguous, near-equal block ranges. Part 1 and part N carry the extra
    embedder / final-layer work, so they get one fewer block where it divides
    unevenly."""
    if not 1 <= n_parts <= n_layers:
        raise ValueError(f"n_parts must be in 1..{n_layers}, got {n_parts}")
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


def target_part(name, cuts):
    """Which part owns a checkpoint tensor, and what it is called there.

    Block tensors are renumbered so every part sees its layers as 0..k, which is
    what lets a part load into a model built with n_layers = its own count.
    """
    n_parts = len(cuts)
    m = re.match(r"layers\.(\d+)\.(.*)", name)
    if m:
        layer, rest = int(m.group(1)), m.group(2)
        for pi, (a, b) in enumerate(cuts):
            if a <= layer < b:
                return pi, f"layers.{layer - a}.{rest}"
        raise KeyError(f"layer {layer} outside {cuts}")
    if name.startswith(LAST_ONLY):
        return n_parts - 1, name
    # everything else (embedders, refiners, t_embedder, pad tokens) is part 1
    return 0, name


def fetch_part_weights(readers, weight_map, cuts, pi, cache=None):
    """This part's tensors, fp16, pulled over HTTP range reads.

    Grouped by shard so each shard's ranges are coalesced into as few requests
    as possible; cast to fp16 as each range lands so the fp32 original is never
    held for more than one run of tensors.
    """
    from safetensors.torch import load_file, save_file

    if cache and os.path.exists(cache):
        log(f"  using cached weights {cache}")
        return load_file(cache)

    wanted = {}                                  # shard -> {ckpt name: part name}
    for name, shard in weight_map.items():
        owner, newname = target_part(name, cuts)
        if owner == pi:
            wanted.setdefault(shard, {})[name] = newname
    if not wanted:
        raise RuntimeError(f"part {pi + 1} matched no tensors")

    total = sum(readers[s].nbytes(n) for s, ns in wanted.items() for n in ns)
    log(f"  fetching {sum(len(v) for v in wanted.values())} tensors, "
        f"{total / 1e9:.2f} GB fp32, from {len(wanted)} shard(s)")

    sd = {}
    for shard, names in wanted.items():
        got = readers[shard].get_tensors(list(names), dtype=torch.float16)
        for ckpt_name, tensor in got.items():
            sd[names[ckpt_name]] = tensor
        del got
        gc.collect()
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        save_file(sd, cache)
    return sd


def build_part(sd, n_blocks, first, last):
    """A reduced model holding only this part's blocks, then the static wrapper."""
    import rope_real
    rope_real.apply()
    from diffusers import ZImageTransformer2DModel

    from static_export import StaticZImageDiT

    model = ZImageTransformer2DModel(
        all_patch_size=[2], all_f_patch_size=[1], in_channels=16,
        dim=DIM, n_layers=n_blocks, n_refiner_layers=2, n_heads=30, n_kv_heads=30,
        norm_eps=1e-5, qk_norm=True, cap_feat_dim=CAP_FEAT_DIM,
        rope_theta=256.0, t_scale=1000.0,
        axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512],
    )
    # Drop what this part will not trace before loading, so the randomly
    # initialised originals are freed rather than merely overwritten.
    if not first:
        for attr in DROP_ON_MIDDLE:
            if hasattr(model, attr):
                delattr(model, attr)
        gc.collect()

    sd = {k: v.to(torch.float32) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    gc.collect()
    # A middle part legitimately lacks embedders and the final layer; those
    # modules have just been deleted, so they cannot show up as missing either.
    missing = [m for m in missing if not m.startswith("layers.")]
    if unexpected:
        raise RuntimeError(f"unexpected tensors for part: {unexpected[:5]}")
    if first and any(m.startswith(FIRST_ONLY) for m in missing):
        raise RuntimeError(f"part 1 is missing embedder weights: {missing[:5]}")
    if last and any(m.startswith(LAST_ONLY) for m in missing):
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
    ap.add_argument("--parts", type=int, default=30)
    ap.add_argument("--only", type=int, default=0, help="export just this part (1-based)")
    ap.add_argument("--cache-weights", action="store_true",
                    help="keep the fetched fp16 weights on disk (costs the whole "
                         "model in disk; only worth it when re-exporting a part)")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the block ranges and exit, fetching nothing")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    cuts = plan_parts(args.parts)
    log(f"{args.parts} parts, block ranges {cuts}")
    if args.plan_only:
        return

    from remote_safetensors import open_shards

    odir = os.path.join(args.work, "onnx")
    os.makedirs(odir, exist_ok=True)

    readers = weight_map = None
    manifest = []
    for pi, (a, b) in enumerate(cuts):
        n = pi + 1
        entry = {"part": n, "blocks": [a, b]}
        if args.only and n != args.only:
            manifest.append(entry)
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
            manifest.append({**entry, "onnx": out})
            continue

        if readers is None:
            log(f"opening {REPO} shard headers")
            readers, weight_map = open_shards(REPO, SUBDIR)

        log(f"building part{n} (blocks {a}..{b}, free {free_gb(args.work):.1f} GB)")
        cache = os.path.join(args.work, "parts", f"part{n}.safetensors") \
            if args.cache_weights else None
        sd = fetch_part_weights(readers, weight_map, cuts, pi, cache=cache)
        part = build_part(sd, b - a, first=(pi == 0), last=(pi == len(cuts) - 1))
        del sd
        gc.collect()
        log(f"  exporting -> {out}")
        export_part(part, out)
        size = sum(os.path.getsize(os.path.join(pdir, f)) for f in os.listdir(pdir))
        log(f"  part{n} onnx {size / 1e9:.2f} GB (free {free_gb(args.work):.1f} GB)")
        manifest.append({**entry, "onnx": out, "inputs": part.input_names,
                         "outputs": part.output_names})
        del part
        gc.collect()

    # Written per invocation, so a --only run still records the full plan and a
    # later run can tell how many parts the device should look for.
    mpath = os.path.join(args.work, "dit_manifest.json")
    json.dump({"parts": manifest, "n_parts": len(cuts),
               "cap_slots": CAP_SLOTS, "latent": LATENT},
              open(mpath, "w"), indent=2)
    log(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()

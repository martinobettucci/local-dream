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
N_REFINER = 2
CAP_SLOTS = 512
LATENT = 128          # 1024 / 8
DIM = 3840
CAP_FEAT_DIM = 2560

# Modules only the tail of the model owns.
LAST_ONLY = ("all_final_layer",)
# The caption branch, which is its own graph -- see CAP_INPUT_NAMES in
# static_export.py for why. Its weights belong to no numbered part.
CAP_ONLY = ("cap_embedder", "cap_pad_token", "context_refiner")

# What the caption branch does not trace. Deleting before loading matters: a
# refiner stack is two full-width blocks, so at dim 3840 it is a couple of GB of
# randomly-initialised fp32 that would otherwise ride all the way through ONNX
# export. Numbered parts compute their own drop list in build_part, from the
# refiner range they were given rather than from where they sit -- with one
# refiner block per graph, "not first" no longer implies "no refiner".
DROP_ON_CAP = ("noise_refiner", "layers", "all_final_layer")


def log(msg):
    print(f"[export_dit] {msg}", flush=True)


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def plan_parts(n_parts, n_layers=N_LAYERS):
    """Contiguous, near-equal block ranges. Part 1 and part N carry the extra
    embedder / final-layer work, so they get one fewer block where it divides
    unevenly.

    Two counts above n_layers are special, and both exist because the head of
    the model is expensive out of all proportion to its parameter count:

      n_layers + 1  part 1 carries no transformer block, only the embedders and
                    the whole noise refiner.
      n_layers + 2  parts 1 and 2 carry no transformer block either, and take
                    ONE noise-refiner block each. This is what the shipped
                    build uses.

    The measurements behind that, all on a 16 GB machine:

      one transformer block, 4608 tokens      11.1 GB    builds
      2 refiner blocks + a transformer block  >21 GB     OOM at quantize
      2 refiner blocks (n_layers + 1 part 1)  15.97 GB   OOM at context binary
      1 refiner block  (n_layers + 2)         ~11 GB     builds

    A refiner block costs ~4.7 GB at context-binary generation -- 4096^2 x 30
    heads of attention, with scores and softmax both retained -- so one per
    graph is the only arrangement that fits. The caption refiner is not in this
    list at all: it moved into a graph of its own (see CAP_INPUT_NAMES).
    """
    if n_parts in (n_layers + 1, n_layers + 2):
        head = n_parts - n_layers
        return [(0, 0)] * head + [(i, i + 1) for i in range(n_layers)]
    if not 1 <= n_parts <= n_layers:
        raise ValueError(f"n_parts must be in 1..{n_layers + 2}, got {n_parts}")
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


CAP_PART = "cap"


def head_plan(n_parts, n_layers=N_LAYERS, n_refiner=N_REFINER):
    """Per-part noise-refiner ranges, and which part does the concatenation.

    The refiner runs on the image stream alone, so every refiner block must
    precede the concatenation; the concatenation is what turns that stream into
    the full sequence the transformer blocks need. So the concatenating part is
    always the last one holding refiner blocks.
    """
    ref = [(0, 0)] * n_parts
    if n_parts == n_layers + 2:
        # One refiner block per graph -- the only arrangement that fits 16 GB.
        for i in range(n_refiner):
            ref[i] = (i, i + 1)
        concat_at = n_refiner - 1
    else:
        ref[0] = (0, n_refiner)
        concat_at = 0
    return ref, concat_at


def target_part(name, cuts):
    """Which part owns a checkpoint tensor, and what it is called there.

    Returns a 0-based part index, or the string CAP_PART for the caption
    branch, which is a graph of its own rather than one of the numbered parts.

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
    if name.startswith(CAP_ONLY):
        return CAP_PART, name
    # The noise refiner is split across the head parts exactly as the
    # transformer blocks are across the rest, and renumbered the same way so
    # each part loads into a model built with its own count.
    m = re.match(r"noise_refiner\.(\d+)\.(.*)", name)
    if m:
        idx, rest = int(m.group(1)), m.group(2)
        ref, _ = head_plan(len(cuts))
        for pi, (a, b) in enumerate(ref):
            if a <= idx < b:
                return pi, f"noise_refiner.{idx - a}.{rest}"
        raise KeyError(f"noise_refiner.{idx} outside {ref}")
    # everything else (image embedder, t_embedder) is part 1
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


def new_model(n_blocks, n_refiner=N_REFINER):
    """A ZImageTransformer2DModel holding only this part's blocks and refiners.

    Either count may be 0 -- parts 1 and 2 of the shipped split carry no
    transformer block, and every part after them carries no refiner block. The
    model is still constructed with one of each and then emptied, rather than
    asked for zero: the module lists are built from the config before anything
    validates it, and a config claiming zero layers is not something the
    reference class is written to survive. One block's worth of
    randomly-initialised weights is allocated and immediately freed, which costs
    a moment and nothing else.
    """
    import rope_real
    rope_real.apply()
    from diffusers import ZImageTransformer2DModel

    model = ZImageTransformer2DModel(
        all_patch_size=[2], all_f_patch_size=[1], in_channels=16,
        dim=DIM, n_layers=max(n_blocks, 1), n_refiner_layers=max(n_refiner, 1),
        n_heads=30, n_kv_heads=30,
        norm_eps=1e-5, qk_norm=True, cap_feat_dim=CAP_FEAT_DIM,
        rope_theta=256.0, t_scale=1000.0,
        axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512],
    )
    if n_blocks == 0:
        model.layers = torch.nn.ModuleList()
        gc.collect()
    return model


def load_in_place(model, sd, drop, kind):
    """Copy `sd` into `model` tensor by tensor, freeing each source as it lands.

    The obvious `model.load_state_dict({k: v.float() for ...})` holds three full
    copies of the part at once -- the fp16 dict that was fetched, the fp32 dict
    built from it, and the model's own parameters. For a middle part that is
    0.7 + 1.4 + 1.4 GB and nobody notices. For a part that also carries a
    refiner stack the same code peaks around 9 GB of weights on top of ~7 GB of
    ONNX tracing and gets OOM-killed. Popping keeps it to the model plus what is
    left of the fp16.

    Returns the names present in the model that `sd` did not supply.
    """
    # Drop what this part will not trace BEFORE loading, so the randomly
    # initialised originals are freed rather than merely overwritten.
    for attr in drop:
        if hasattr(model, attr):
            delattr(model, attr)
    gc.collect()

    msd = model.state_dict()
    unexpected, seen = [], set()
    with torch.no_grad():
        for k in list(sd.keys()):
            t = sd.pop(k)
            dst = msd.get(k)
            if dst is None:
                unexpected.append(k)
            else:
                dst.copy_(t.to(torch.float32))
                seen.add(k)
            del t
    missing = [k for k in msd if k not in seen]
    del msd
    gc.collect()
    if unexpected:
        raise RuntimeError(f"unexpected tensors for {kind}: {unexpected[:5]}")
    return missing


def build_cap(sd):
    """The caption branch: cap_embedder + context_refiner, as its own graph."""
    from static_export import StaticZImageCaption

    model = new_model(0, N_REFINER)
    missing = load_in_place(model, sd, DROP_ON_CAP, "the caption branch")
    absent = [m for m in missing if m.startswith(CAP_ONLY)]
    if absent:
        raise RuntimeError(
            f"the caption branch is missing {len(absent)} weight(s), which "
            f"would export as random values: {absent[:5]}")
    model.eval()
    return StaticZImageCaption(model, CAP_SLOTS, LATENT, LATENT)


def build_part(sd, n_blocks, first, last, refiner=(0, N_REFINER), concat=None):
    """A reduced model holding only this part's blocks, then the static wrapper."""
    from static_export import StaticZImageDiT

    n_ref = refiner[1] - refiner[0]
    concat = first if concat is None else concat
    model = new_model(n_blocks, n_ref)

    # Drop by what the part actually runs, not by where it sits. The head is no
    # longer "part 1 does everything": with one refiner block per graph, part 2
    # is not `first` and still needs noise_refiner, while part 1 needs only one
    # of its two blocks. context_refiner is never wanted here -- it belongs to
    # the caption branch.
    drop = ["context_refiner"] + ([] if n_ref else ["noise_refiner"])
    missing = load_in_place(
        model, sd, drop,
        f"part (blocks={n_blocks}, refiner={n_ref}, first={first})")

    # `layers.*` and `noise_refiner.*` are the things that must NEVER be
    # missing: an absent block weight leaves that block randomly initialised,
    # and the part then exports, converts, quantizes and compiles without
    # complaint. An earlier version filtered these out of `missing` before
    # checking anything, which is exactly backwards -- it silenced the only
    # fatal case and kept the benign ones.
    absent_blocks = [m for m in missing
                     if m.startswith(("layers.", "noise_refiner."))]
    if absent_blocks:
        raise RuntimeError(
            f"part is missing {len(absent_blocks)} block weight(s), which would "
            f"export as random values: {absent_blocks[:5]}")
    # A part that runs neither embedder nor final layer legitimately lacks both.
    if first and any(m.startswith(("all_x_embedder", "t_embedder"))
                     for m in missing):
        raise RuntimeError(f"part 1 is missing embedder weights: {missing[:5]}")
    if last and any(m.startswith(LAST_ONLY) for m in missing):
        raise RuntimeError(f"last part is missing the final layer: {missing[:5]}")
    model.eval()
    return StaticZImageDiT(model, CAP_SLOTS, LATENT, LATENT,
                           block_start=0, block_end=n_blocks,
                           first=first, last=last,
                           refiner=(0, n_ref), concat=concat)


def export_part(part, path, dim=DIM):
    inputs = part.example_inputs(true_len=CAP_SLOTS, dim=dim)
    torch.onnx.export(part, inputs, path, opset_version=17, dynamo=False,
                      input_names=part.input_names, output_names=part.output_names)
    check_io_names(path, part.input_names, part.output_names)
    return path


def check_io_names(path, want_in, want_out):
    """torch.onnx.export treats input_names/output_names as requests, not
    guarantees. A name that collides with something already in the graph is
    silently suffixed -- an input called `hidden` alongside an output of the
    same name comes out as `hidden.1`. The app binds tensors by name, so that
    rename survives conversion, survives quantization, and finally shows up as a
    missing tensor on the device. Catch it here instead.
    """
    import onnx

    m = onnx.load(path, load_external_data=False)
    got_in = [i.name for i in m.graph.input]
    got_out = [o.name for o in m.graph.output]
    if got_in != list(want_in) or got_out != list(want_out):
        raise RuntimeError(
            f"ONNX IO names do not match the contract:\n"
            f"  inputs  wanted {list(want_in)}\n"
            f"          got    {got_in}\n"
            f"  outputs wanted {list(want_out)}\n"
            f"          got    {got_out}")
    check_htp_ops(m)


# Ops the HTP backend has no implementation for, which qnn-context-binary-generator
# only rejects at the very last stage -- after convert and quantize have both
# spent several minutes succeeding. IsNan is the one that actually bit:
# F.scaled_dot_product_attention with a boolean mask emits it to guard rows
# where every key is masked. The rest are here because they arrive by the same
# route (a decomposition nobody asked for) and would fail the same way.
UNSUPPORTED_OPS = {"IsNaN", "IsInf", "NonZero", "Loop", "If", "Scan",
                   "SequenceAt", "GatherND", "Unique"}


def check_htp_ops(model):
    bad = sorted({n.op_type for n in model.graph.node} & UNSUPPORTED_OPS)
    if bad:
        raise RuntimeError(
            f"graph contains ops the HTP cannot run: {bad}\n"
            f"  qnn-context-binary-generator would reject this after convert and "
            f"quantize both succeed, which costs minutes per part. IsNaN usually "
            f"means F.scaled_dot_product_attention with a boolean mask -- use an "
            f"additive float mask and write the attention out.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--parts", type=int, default=30)
    ap.add_argument("--only", type=int, default=0, help="export just this part (1-based)")
    ap.add_argument("--cap", action="store_true",
                    help="export the caption branch (unet_cap.onnx) instead of "
                         "a numbered part")
    ap.add_argument("--cache-weights", action="store_true",
                    help="keep the fetched fp16 weights on disk (costs the whole "
                         "model in disk; only worth it when re-exporting a part)")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the block ranges and exit, fetching nothing")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    cuts = plan_parts(args.parts)
    ref_plan, concat_at = head_plan(args.parts)
    log(f"{args.parts} parts, block ranges {cuts}")
    log(f"  noise refiner {[r for r in ref_plan if r[1] > r[0]]}, "
        f"concatenation on part {concat_at + 1}")
    if args.plan_only:
        return

    from remote_safetensors import open_shards
    from static_export import (CAP_INPUT_NAMES, CAP_OUTPUT_NAMES,
                               dit_input_names, dit_output_names)

    odir = os.path.join(args.work, "onnx")
    os.makedirs(odir, exist_ok=True)

    if args.cap:
        pdir = os.path.join(odir, "cap")
        os.makedirs(pdir, exist_ok=True)
        out = os.path.join(pdir, "unet_cap.onnx")
        if os.path.exists(out):
            check_io_names(out, CAP_INPUT_NAMES, CAP_OUTPUT_NAMES)
            log("caption branch already exported and still valid, skipping")
            return
        log(f"opening {REPO} shard headers")
        readers, weight_map = open_shards(REPO, SUBDIR)
        sd = fetch_part_weights(readers, weight_map, cuts, CAP_PART)
        part = build_cap(sd)
        del sd
        gc.collect()
        log(f"  exporting -> {out}")
        export_part(part, out)
        size = sum(os.path.getsize(os.path.join(pdir, f))
                   for f in os.listdir(pdir))
        log(f"  caption branch onnx {size / 1e9:.2f} GB "
            f"(free {free_gb(args.work):.1f} GB)")
        return

    readers = weight_map = None
    manifest = []
    for pi, (a, b) in enumerate(cuts):
        n = pi + 1
        refiner = ref_plan[pi]
        concat = pi == concat_at
        entry = {"part": n, "blocks": [a, b], "refiner": list(refiner),
                 "concat": concat}
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
        first, last = pi == 0, pi == len(cuts) - 1
        want_in = dit_input_names(first, b > a, concat,
                                  refiner[1] > refiner[0])
        want_out = dit_output_names(first, last)
        if os.path.exists(out):
            # Re-check rather than trust it. Resuming is the normal way this
            # runs, so an ONNX on disk is usually from an *earlier* version of
            # this script -- possibly one that predates these checks, or that
            # was interrupted mid-write. Skipping validation on exactly the
            # files least likely to have been validated is backwards.
            check_io_names(out, want_in, want_out)
            log(f"part{n} already exported and still valid, skipping")
            manifest.append({**entry, "onnx": out, "inputs": want_in,
                             "outputs": want_out})
            continue

        if readers is None:
            log(f"opening {REPO} shard headers")
            readers, weight_map = open_shards(REPO, SUBDIR)

        log(f"building part{n} (blocks {a}..{b}, free {free_gb(args.work):.1f} GB)")
        # Cache filename carries the plan, not just the part number: part 8 of 8
        # is blocks 27..30 while part 8 of 30 is block 7, and reusing one for the
        # other would export the wrong weights under the right name.
        cache = os.path.join(args.work, "parts",
                             f"p{n}of{len(cuts)}.safetensors") \
            if args.cache_weights else None
        sd = fetch_part_weights(readers, weight_map, cuts, pi, cache=cache)
        part = build_part(sd, b - a, first=first, last=last,
                          refiner=refiner, concat=concat)
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
    # Merge rather than overwrite: convert_all.sh drives this one --only part at
    # a time, so a plain rewrite would leave a manifest describing the last part
    # and nothing else.
    mpath = os.path.join(args.work, "dit_manifest.json")
    prior = {}
    if os.path.exists(mpath):
        try:
            old = json.load(open(mpath))
            if old.get("n_parts") == len(cuts):
                prior = {p["part"]: p for p in old.get("parts", [])}
        except (ValueError, KeyError):
            pass
    merged = []
    for e in manifest:
        was = prior.get(e["part"], {})
        merged.append({**was, **e} if len(e) > 2 else (was or e))
    json.dump({"parts": merged, "n_parts": len(cuts),
               "cap_slots": CAP_SLOTS, "latent": LATENT},
              open(mpath, "w"), indent=2)
    log(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()

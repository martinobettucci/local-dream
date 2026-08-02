#!/usr/bin/env python3
"""Check that any N-way DiT split covers the model exactly once.

The split count is not a fixed choice -- it is set by how much RAM the
converting machine has, and it changed from 8 to 15 to 30 during one afternoon.
A routing bug at a part boundary would not fail loudly: the export would
succeed, the graph would convert, quantize and compile, and the model would
simply produce wrong images because two parts both ran block 14, or none did.

    python tools/zimage/verify_part_plan.py           # planning only, offline
    python tools/zimage/verify_part_plan.py --remote  # also route real tensors
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_dit import N_LAYERS, plan_parts, target_part  # noqa: E402


def check_plans(counts):
    ok = True
    for n in counts:
        cuts = plan_parts(n)
        blocks = [b for a, bb in cuts for b in range(a, bb)]
        problems = []
        if len(cuts) != n:
            problems.append(f"{len(cuts)} parts, wanted {n}")
        if sorted(blocks) != list(range(N_LAYERS)):
            problems.append("blocks are not exactly 0..N-1 once each")
        if any(bb <= a for a, bb in cuts):
            problems.append("empty part")
        if any(cuts[i][1] != cuts[i + 1][0] for i in range(len(cuts) - 1)):
            problems.append("not contiguous")
        sizes = sorted({bb - a for a, bb in cuts})
        if len(sizes) > 2:
            problems.append(f"block counts vary too much: {sizes}")
        if problems:
            ok = False
            print(f"  n={n:2d} FAIL {'; '.join(problems)}  cuts={cuts}")
        else:
            print(f"  n={n:2d} ok, block-counts {sizes}")
    return ok


def check_routing(counts):
    """Every checkpoint tensor lands on exactly one part, renumbered in range."""
    import json

    from huggingface_hub import hf_hub_download

    from export_dit import REPO, SUBDIR

    index = hf_hub_download(
        REPO, f"{SUBDIR}/diffusion_pytorch_model.safetensors.index.json")
    weight_map = json.load(open(index))["weight_map"]

    ok = True
    for n in counts:
        cuts = plan_parts(n)
        placed, per_part = 0, {}
        for name in weight_map:
            pi, new = target_part(name, cuts)
            per_part.setdefault(pi, set()).add(new)
            placed += 1
            if new.startswith("layers."):
                idx = int(new.split(".")[1])
                span = cuts[pi][1] - cuts[pi][0]
                if not 0 <= idx < span:
                    ok = False
                    print(f"  n={n} FAIL {name} -> part{pi + 1} as {new}, "
                          f"but that part only has {span} block(s)")
        empty = [i + 1 for i in range(n) if i not in per_part]
        if empty:
            ok = False
            print(f"  n={n} FAIL parts with no tensors: {empty}")
        elif placed == len(weight_map):
            print(f"  n={n:2d} routing ok, {placed}/{len(weight_map)} tensors, "
                  f"all {n} parts non-empty")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true",
                    help="also route the real checkpoint's tensor names "
                         "(reads one index file from the Hub)")
    args = ap.parse_args()

    counts = list(range(1, N_LAYERS + 1))
    print(f"planning, n = 1..{N_LAYERS}")
    ok = check_plans(counts)
    if args.remote:
        print("\nrouting real tensors")
        ok = check_routing([1, 8, 15, 16, 29, 30]) and ok

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

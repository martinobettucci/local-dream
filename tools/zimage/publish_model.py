#!/usr/bin/env python3
"""Assemble the published model directory on the Hub and write its manifest.

The conversion uploads each piece to `partial/...` as it is built. This turns
those pieces into the thing the app downloads, without ever holding the model
locally -- which matters because the model is larger than this machine's free
disk, and always was.

Two things make that possible:

  * `CommitOperationCopy` copies a file inside a repo server-side. The bytes
    never move through here.
  * The app downloads loose files listed in a manifest rather than one archive,
    so nothing has to be zipped. At several GB an archive is the wrong shape
    anyway: the device would need the zip and its extraction free at the same
    time, and a dropped connection would cost the whole download.

    python tools/zimage/publish_model.py --dry-run
    python tools/zimage/publish_model.py
"""
import argparse
import json
import sys

DEFAULT_REPO = "P2Enjoy/z-image-turbo-qnn"
DEST = "model"

def collect(files, prefix, stem, expected, label):
    """Parts 1..expected under `prefix`, or a report of exactly which are absent.

    Deliberately not "scan upward until a gap": the conversion builds part 1
    last (it carries both refiner stacks, so it wants the machine to itself),
    which means a gap-stopping scan reports zero parts for the entire run and
    says nothing useful about what is left.
    """
    want = [f"{prefix}/{stem}{i}.bin" for i in range(1, expected + 1)]
    missing = [w for w in want if w not in files]
    if missing:
        short = ", ".join(m.rsplit("/", 1)[1] for m in missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise SystemExit(
            f"{label} incomplete: {len(missing)} of {expected} missing "
            f"from {prefix} -- {short}{more}")
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--dit-dir", default=None,
                    help="the DiT directory to publish from, e.g. "
                         "partial/n32-attn5. Required once more than one "
                         "exists: two builds of the same split differ only in "
                         "how much device memory they ask for, which is "
                         "exactly the kind of difference a guess would hide.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationCopy, HfApi, get_token

    if not get_token():
        raise SystemExit("no HF token; set HF_TOKEN")
    api = HfApi()

    info = api.repo_info(args.repo, repo_type="model", files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings}
    names = set(sizes)

    # Find the DiT and encoder directories that were actually produced. They are
    # named by part count (n30, clip_m6) precisely so that a re-run at a
    # different split cannot be silently mixed with an older one.
    dit_dirs = sorted({f.split("/")[1] for f in names
                       if f.startswith("partial/n") and f.count("/") >= 2})
    clip_dirs = sorted({f.split("/")[1] for f in names
                        if f.startswith("partial/clip_m") and f.count("/") >= 2})
    if args.dit_dir:
        dit_dirs = [args.dit_dir.strip("/").split("/")[-1]]
        if not any(f.startswith(f"partial/{dit_dirs[0]}/") for f in names):
            raise SystemExit(f"no files under partial/{dit_dirs[0]}/")
    elif not dit_dirs:
        raise SystemExit("no partial/n<N>/ directory in the repo yet")
    elif len(dit_dirs) > 1:
        raise SystemExit(f"several DiT builds present ({dit_dirs}); "
                         "pass --dit-dir to say which one to publish")
    if len(clip_dirs) > 1:
        raise SystemExit(f"several encoder splits present ({clip_dirs})")

    dit_prefix = f"partial/{dit_dirs[0]}"
    import re as _re
    m = _re.match(r"n(\d+)", dit_dirs[0])
    if not m:
        raise SystemExit(f"cannot read a part count from {dit_dirs[0]!r}")
    dit_parts = collect(names, dit_prefix, "unet_part", int(m.group(1)), "DiT")
    # The caption branch is a graph of its own, not a numbered part, and the
    # backend fails without it -- so it is checked here rather than assumed.
    cap = f"{dit_prefix}/unet_cap.bin"
    if cap not in names:
        raise SystemExit(f"{cap} missing (the DiT's caption branch)")
    dit_parts.append(cap)

    plan = [(src, src.rsplit("/", 1)[1]) for src in dit_parts]

    # The text encoder, from partial/ if it is still there and from the last
    # published model/ if it is not.
    #
    # This fallback is not tidiness. partial/clip_m6/ was deleted once the model
    # was published ("model/ holds the same bytes"), which is true -- but the
    # next publish then found no encoder, took the "publishing DiT only" branch,
    # and wrote a manifest of 38 files instead of 45. The bytes were still in
    # model/; only the list of them was wrong, so a fresh install downloaded a
    # directory the backend cannot start from. A warning is the wrong severity
    # for a piece main.cpp calls showHelpAndExit over: it is fatal below.
    if clip_dirs:
        clip_prefix = f"partial/{clip_dirs[0]}"
        n_clip = int(clip_dirs[0].split("_m")[1])
    else:
        clip_prefix = DEST
        n_clip = 0
        while f"{DEST}/clip_part{n_clip + 1}.bin" in names:
            n_clip += 1
        if n_clip:
            print(f"note: no partial/clip_m*/; taking the {n_clip}-part encoder "
                  f"from {DEST}/, which is where it already lives")
    if not n_clip:
        raise SystemExit(
            "no text encoder found under partial/clip_m*/ or "
            f"{DEST}/clip_part*.bin, and the backend cannot start without it")
    clip_parts = collect(names, clip_prefix, "clip_part", n_clip, "text encoder")
    plan += [(src, src.rsplit("/", 1)[1]) for src in clip_parts]
    tok = f"{clip_prefix}/token_emb.bin"
    if tok not in names:
        raise SystemExit(f"{tok} missing, and the backend requires it")
    plan.append((tok, "token_emb.bin"))

    # tokenizer.json, token_emb.bin, clip and the VAE decoder are hard
    # requirements in main.cpp's zimage branch -- the backend calls
    # showHelpAndExit if any is absent -- so a missing one is fatal here rather
    # than a warning. config.json and the ZIMAGE marker only matter for a
    # manually imported copy, so they are included when present and skipped
    # otherwise.
    def find(name):
        for prefix in ("partial", DEST):
            if f"{prefix}/{name}" in names:
                return f"{prefix}/{name}"
        return None

    for extra in ("tokenizer.json", "vae_decoder.bin"):
        src = find(extra)
        if src is None:
            raise SystemExit(f"{extra} is in neither partial/ nor {DEST}/, "
                             f"and the backend requires it")
        plan.append((src, extra))
    for extra in ("vae_encoder.bin", "config.json", "ZIMAGE"):
        src = find(extra)
        if src:
            plan.append((src, extra))
        else:
            print(f"note: {extra} absent, skipping (optional)")

    total = sum(sizes.get(src) or 0 for src, _ in plan)
    print(f"{len(plan)} files, {total / 1e9:.2f} GB")
    for src, dst in plan:
        print(f"  {src:44} -> {DEST}/{dst}  "
              f"({(sizes.get(src) or 0) / 1e6:.0f} MB)")

    manifest = {
        "format": "zimage",
        "files": [{"name": dst, "size": sizes.get(src) or 0} for src, dst in plan],
    }
    approx = f"{total / 1e9:.1f}GB"
    print(f"\napproximateSize for Model.kt: \"{approx}\"")

    if args.dry_run:
        print("dry run: nothing published")
        return

    # Server-side copies: the bytes never pass through this machine.
    ops = [CommitOperationCopy(src_path_in_repo=src, path_in_repo=f"{DEST}/{dst}")
           for src, dst in plan if src != f"{DEST}/{dst}"]
    if ops:
        api.create_commit(repo_id=args.repo, repo_type="model", operations=ops,
                          commit_message=f"publish {len(plan)} model files to {DEST}/")
    print(f"copied {len(ops)} files to {DEST}/ "
          f"({len(plan) - len(ops)} already in place)")

    api.upload_file(
        path_or_fileobj=json.dumps(manifest, indent=2).encode(),
        path_in_repo=f"{DEST}/manifest.json",
        repo_id=args.repo, repo_type="model")
    print(f"manifest -> {DEST}/manifest.json")
    print(f"done: https://huggingface.co/{args.repo}/tree/main/{DEST}")


if __name__ == "__main__":
    sys.exit(main())

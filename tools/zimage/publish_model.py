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
    if not dit_dirs:
        raise SystemExit("no partial/n<N>/ directory in the repo yet")
    if len(dit_dirs) > 1:
        raise SystemExit(f"several DiT splits present ({dit_dirs}); "
                         "delete the stale one before publishing")
    if len(clip_dirs) > 1:
        raise SystemExit(f"several encoder splits present ({clip_dirs})")

    dit_prefix = f"partial/{dit_dirs[0]}"
    dit_parts = collect(names, dit_prefix, "unet_part",
                        int(dit_dirs[0][1:]), "DiT")

    plan = [(src, src.rsplit("/", 1)[1]) for src in dit_parts]

    if clip_dirs:
        clip_prefix = f"partial/{clip_dirs[0]}"
        clip_parts = collect(names, clip_prefix, "clip_part",
                             int(clip_dirs[0].split("_m")[1]), "text encoder")
        plan += [(src, src.rsplit("/", 1)[1]) for src in clip_parts]
        tok = f"{clip_prefix}/token_emb.bin"
        if tok not in names:
            raise SystemExit(f"{tok} missing")
        plan.append((tok, "token_emb.bin"))
    else:
        print("warning: no text encoder parts found; publishing DiT only")

    # tokenizer.json, token_emb.bin, clip and the VAE decoder are hard
    # requirements in main.cpp's zimage branch -- the backend calls
    # showHelpAndExit if any is absent -- so a missing one is fatal here rather
    # than a warning. config.json and the ZIMAGE marker only matter for a
    # manually imported copy, so they are included when present and skipped
    # otherwise.
    for extra in ("tokenizer.json", "vae_decoder.bin"):
        src = f"partial/{extra}"
        if src not in names:
            raise SystemExit(f"{src} missing, and the backend requires it")
        plan.append((src, extra))
    for extra in ("vae_encoder.bin", "config.json", "ZIMAGE"):
        src = f"partial/{extra}"
        if src in names:
            plan.append((src, extra))
        else:
            print(f"note: {src} absent, skipping (optional)")

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
           for src, dst in plan]
    api.create_commit(repo_id=args.repo, repo_type="model", operations=ops,
                      commit_message=f"publish {len(plan)} model files to {DEST}/")
    print(f"copied {len(ops)} files to {DEST}/")

    api.upload_file(
        path_or_fileobj=json.dumps(manifest, indent=2).encode(),
        path_in_repo=f"{DEST}/manifest.json",
        repo_id=args.repo, repo_type="model")
    print(f"manifest -> {DEST}/manifest.json")
    print(f"done: https://huggingface.co/{args.repo}/tree/main/{DEST}")


if __name__ == "__main__":
    sys.exit(main())

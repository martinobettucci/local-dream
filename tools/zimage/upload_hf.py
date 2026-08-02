#!/usr/bin/env python3
"""Publish the converted Z-Image artifacts (and optionally the APK) to the Hub.

    python tools/zimage/upload_hf.py --model-dir <dir> [--apk <path>] \
        [--repo P2Enjoy/z-image-turbo-qnn] [--dry-run]

The model card is uploaded from tools/zimage/MODEL_CARD.md, which carries an
UNTESTED banner. Do not remove that banner until someone has confirmed a
successful generation on a real device — publishing unvalidated binaries is
defensible, quietly implying they work is not.
"""
import argparse
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = "P2Enjoy/z-image-turbo-qnn"
ARCHIVE = "z_image_turbo_w4a16_qnn2.39_8gen3.zip"

# What the app's built-in entry expects to find after unzipping. Keep in sync
# with docs/zimage.md section 3 and ModelRepository.createZImageTurboModel.
REQUIRED = ["tokenizer.json", "token_emb.bin", "clip.bin",
            "vae_decoder.bin", "unet_part1.bin"]
OPTIONAL = ["vae_encoder.bin", "config.json", "ZIMAGE"]


def build_archive(model_dir, out_path):
    """Zip the model directory with files at the archive ROOT.

    The downloader extracts straight into the model directory, so a nested
    folder would put every file one level too deep and the backend would report
    each one missing.
    """
    names = sorted(os.listdir(model_dir))
    missing = [f for f in REQUIRED if f not in names]
    if missing:
        raise SystemExit(f"model dir is missing required files: {missing}")
    parts = sorted(n for n in names if n.startswith("unet_part"))
    print(f"  {len(parts)} DiT parts: {parts}")

    total = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            p = os.path.join(model_dir, n)
            if not os.path.isfile(p):
                continue
            total += os.path.getsize(p)
            z.write(p, arcname=n)          # arcname without a directory prefix
    print(f"  archive {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB "
          f"from {total / 1e9:.2f} GB of files)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", help="directory of converted .bin files")
    ap.add_argument("--apk", help="APK to publish alongside the weights")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--archive-out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, get_token

    if not get_token():
        raise SystemExit("no HF token; run `huggingface-cli login` or set HF_TOKEN")
    api = HfApi()
    who = api.whoami()
    print(f"authenticated as {who.get('name')}; target repo {args.repo}")

    uploads = []
    if args.model_dir:
        out = args.archive_out or os.path.join(
            os.path.dirname(os.path.abspath(args.model_dir)), ARCHIVE)
        print("building archive...")
        build_archive(args.model_dir, out)
        uploads.append((out, ARCHIVE))
    if args.apk:
        uploads.append((args.apk, os.path.basename(args.apk)))
    uploads.append((os.path.join(HERE, "MODEL_CARD.md"), "README.md"))

    for local, remote in uploads:
        size = os.path.getsize(local) / 1e6
        print(f"  {'(dry-run) ' if args.dry_run else ''}upload {remote}  ({size:.1f} MB)")

    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    for local, remote in uploads:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=args.repo, repo_type="model")
        print(f"  uploaded {remote}")
    print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    sys.exit(main())

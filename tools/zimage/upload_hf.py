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
    # Per-artifact modes, used by convert_all.sh to checkpoint each part to the
    # Hub the moment it is built. The conversion box is ephemeral and has less
    # free disk than the finished model needs, so "upload it and delete it" is
    # the only way a long run survives either.
    ap.add_argument("--put", help="upload a single file and exit")
    ap.add_argument("--as", dest="remote", help="path in the repo for --put")
    ap.add_argument("--list-remote", metavar="PREFIX", nargs="?", const="",
                    help="print repo files under PREFIX and exit")
    args = ap.parse_args()

    from huggingface_hub import HfApi, get_token

    if not get_token():
        raise SystemExit("no HF token; run `huggingface-cli login` or set HF_TOKEN")
    api = HfApi()

    if args.list_remote is not None:
        # This listing IS the resume point: convert_all.sh rebuilds whatever it
        # does not name. So a failure must never look like an empty repo -- and
        # it did. One transient Hub error was swallowed here, the caller read
        # "already on the Hub: 0 graphs", and a run with every part already
        # published started rebuilding all of them from scratch. Nothing in the
        # log said anything was wrong.
        #
        # Retry, then fail loudly. Only a repo that genuinely does not exist yet
        # is legitimately an empty listing.
        import time

        from huggingface_hub.errors import RepositoryNotFoundError

        last = None
        for attempt in range(5):
            try:
                files = list(api.list_repo_files(args.repo, repo_type="model"))
            except RepositoryNotFoundError:
                return                          # nothing published yet
            except Exception as exc:  # noqa: BLE001 - reported below
                last = exc
                if attempt + 1 < 5:
                    time.sleep(2 ** attempt * 5)
                continue
            for f in files:
                if f.startswith(args.list_remote):
                    print(f)
            return
        raise SystemExit(
            f"could not list {args.repo} after 5 tries: {type(last).__name__}: "
            f"{last}\nRefusing to report an empty listing -- the caller would "
            f"read it as 'nothing is published' and rebuild everything.")

    if args.put:
        remote = args.remote or os.path.basename(args.put)
        if args.dry_run:
            print(f"(dry-run) would upload {args.put} -> {remote}")
            return
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        # Retry with backoff. A conversion run is hours long and every part is
        # uploaded the moment it exists, so a few seconds of network trouble
        # would otherwise abort the whole run under `set -e` and throw away the
        # 40 minutes of compute that produced the part. Observed in practice:
        # "httpx.ConnectError: [Errno 111] Connection refused" mid-run, from a
        # transient proxy blip rather than anything wrong with the file.
        import time

        last = None
        for attempt in range(5):
            try:
                api.upload_file(path_or_fileobj=args.put, path_in_repo=remote,
                                repo_id=args.repo, repo_type="model")
                print(f"uploaded {remote} "
                      f"({os.path.getsize(args.put) / 1e6:.1f} MB)"
                      + (f" [attempt {attempt + 1}]" if attempt else ""))
                return
            except Exception as exc:  # noqa: BLE001 - re-raised below
                last = exc
                if attempt + 1 < 5:
                    wait = 2 ** attempt * 5
                    print(f"upload of {remote} failed ({type(exc).__name__}), "
                          f"retrying in {wait}s")
                    time.sleep(wait)
        raise SystemExit(f"upload of {remote} failed after 5 tries: {last}")

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

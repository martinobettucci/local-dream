#!/usr/bin/env bash
# Every text encoder part plus token_emb.bin, each published as it completes.
#
# Same shape as convert_all.sh and for the same reasons: the box is ephemeral
# and has less free disk than the finished model, so a part is uploaded the
# moment it exists and deleted locally, and a restart resumes from the first
# gap in the remote listing.
#
#   export HF_TOKEN=...
#   PY=... QNN=... ./convert_clip_all.sh <work_dir> <n_parts> [bits] [arch]
set -euo pipefail

W="${1:?work dir}"; NPARTS="${2:?number of parts}"
WBITS="${3:-4}"; ARCH="${4:-v73}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_REPO="${HF_REPO:-P2Enjoy/z-image-turbo-qnn}"
REMOTE_DIR="${REMOTE_DIR:-partial/clip_m$NPARTS}"
PY="${PY:?set PY to the torch venv python}"
QNN="${QNN:?set QNN to the python3.10 venv python}"

mkdir -p "$W/out" "$W/stats"
free_gb() { df -BG --output=avail "$W" | tail -1 | tr -dc '0-9'; }

echo "==> text encoder: $NPARTS parts, w${WBITS}a16, Hexagon $ARCH -> $HF_REPO/$REMOTE_DIR"
"$PY" "$HERE/export_clip.py" --work "$W" --parts "$NPARTS" --plan-only

# One listing for the whole run rather than one per part: this is the resume
# point, and re-asking after every upload would only ever confirm what we just
# did.
#
# NOT `|| true`, and not `2>/dev/null`. This listing decides what gets rebuilt,
# so one that fails and reads as "nothing is published" costs the whole run --
# which is what happened: a transient Hub error put all 30 already-published
# parts back in the queue, silently. upload_hf.py now retries and exits non-zero
# rather than printing nothing, and `set -e` stops here instead of starting over.
DONE="$("$PY" "$HERE/upload_hf.py" --repo "$HF_REPO" --list-remote "$REMOTE_DIR/")"

# token_emb.bin first: it is a plain fp16 dump with no conversion, it is the
# single biggest file in the model at 778 MB, and getting it out of the way
# means a later failure never leaves the archive missing the one piece that
# needs no toolchain at all.
if echo "$DONE" | grep -qx "$REMOTE_DIR/token_emb.bin"; then
  echo "==> token_emb.bin already published"
else
  echo "==> token_emb.bin (free $(free_gb)G)"
  "$PY" "$HERE/export_clip.py" --work "$W" --token-emb
  "$PY" "$HERE/upload_hf.py" --repo "$HF_REPO" \
      --put "$W/token_emb.bin" --as "$REMOTE_DIR/token_emb.bin"
  rm -f "$W/token_emb.bin"
fi

for N in $(seq 1 "$NPARTS"); do
  if echo "$DONE" | grep -qx "$REMOTE_DIR/clip_part$N.bin"; then
    echo "==> clip part$N already published, skipping"
    continue
  fi
  echo "==> clip part$N of $NPARTS (free $(free_gb)G)"
  PY="$PY" QNN="$QNN" "$HERE/convert_clip.sh" "$W" "$N" "$NPARTS" "$WBITS" "$ARCH"
  "$PY" "$HERE/upload_hf.py" --repo "$HF_REPO" \
      --put "$W/out/clip_part$N.bin" --as "$REMOTE_DIR/clip_part$N.bin"
  rm -f "$W/out/clip_part$N.bin"
  "$PY" "$HERE/upload_hf.py" --repo "$HF_REPO" \
      --put "$W/stats/clip_part$N.tsv" --as "$REMOTE_DIR/stats/clip_part$N.tsv" || true
done

echo
echo "==> peak RSS by stage (MB) across the encoder parts built here"
awk -F'\t' '{ if ($2 > peak[$1]) peak[$1] = $2; secs[$1] += $3 }
     END { for (s in peak) printf "  %-9s %6d MB   %5d s total\n", s, peak[s]/1024, secs[s] }' \
    "$W"/stats/clip_part*.tsv 2>/dev/null | sort || echo "  (none built this run)"
echo "==> https://huggingface.co/$HF_REPO/tree/main/$REMOTE_DIR"

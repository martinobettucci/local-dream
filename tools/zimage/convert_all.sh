#!/usr/bin/env bash
# Convert every DiT part, checkpointing each one to the Hub as it completes.
#
# The conversion box is ephemeral and has less free disk than the finished model
# needs. Both problems have the same answer: a part is uploaded the moment its
# context binary exists and deleted locally, so peak disk is one part's
# intermediates rather than the whole model, and a machine that dies mid-run
# costs one part rather than the run.
#
# Restarting is therefore free and is the normal way to use this: it asks the
# Hub what is already there and starts from the first gap.
#
#   export HF_TOKEN=...
#   PY=... QNN=... ./convert_all.sh <work_dir> <n_parts> [bits] [dsp_arch]
#
# n_parts is the RAM lever — see the note in convert_part.sh. Peak RSS per
# stage lands in <work_dir>/stats/partN.tsv and is summarised at the end.
set -euo pipefail

W="${1:?work dir}"; NPARTS="${2:?number of parts}"
WBITS="${3:-4}"; ARCH="${4:-v73}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_REPO="${HF_REPO:-P2Enjoy/z-image-turbo-qnn}"
# The split count is part of the remote path, not just the filename. "part 15"
# means block 14 in a 30-way split and blocks 14-15 in a 15-way split, and the
# split count has already changed three times as the RAM ceiling was measured.
# Without this, resuming after a change would skip parts built under the old
# plan and assemble a model out of two incompatible splits -- silently, since
# every file would be present and correctly named.
REMOTE_DIR="${REMOTE_DIR:-partial/n$NPARTS}"
PY="${PY:?set PY to the torch venv python}"
QNN="${QNN:?set QNN to the python3.10 venv python}"
KEEP_LOCAL="${KEEP_LOCAL:-0}"

mkdir -p "$W/out" "$W/stats"
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

echo "==> $NPARTS parts, w${WBITS}a16, Hexagon $ARCH -> $HF_REPO/$REMOTE_DIR"
"$PY" "$HERE/export_dit.py" --work "$W" --parts "$NPARTS" --plan-only

# One listing for the whole run rather than one per part: this is the resume
# point, and re-asking after every upload would only ever confirm what we just
# did.
DONE="$("$PY" "$HERE/upload_hf.py" --repo "$HF_REPO" --list-remote "$REMOTE_DIR/" 2>/dev/null || true)"
echo "==> already on the Hub: $(echo "$DONE" | grep -c 'unet_part' || true) parts"

# Part order, not part range. Part 1 is not like the others: it carries both
# refiner stacks and every embedder on top of its own block, which is 3.63 GB of
# fp32 against ~0.72 GB for a middle part -- five times the weights, five times
# the ONNX, and an export that peaks near the RAM ceiling. Doing it last means
# it runs when every other part has been uploaded and deleted, so it gets the
# whole disk and the whole machine instead of competing with 29 siblings.
ORDER="${ORDER:-$(seq 2 "$NPARTS"; echo 1)}"

for N in $ORDER; do
  if echo "$DONE" | grep -qx "$REMOTE_DIR/unet_part$N.bin"; then
    echo "==> part$N already published, skipping"
    continue
  fi
  echo "==> part$N of $NPARTS (free $(free_gb)G)"
  PY="$PY" QNN="$QNN" "$HERE/convert_part.sh" "$W" "$N" "$NPARTS" "$WBITS" "$ARCH"

  "$PY" "$HERE/upload_hf.py" --repo "$HF_REPO" \
      --put "$W/out/unet_part$N.bin" --as "$REMOTE_DIR/unet_part$N.bin"
  # Uploaded means committed. Keeping it costs the disk the next part needs.
  [ "$KEEP_LOCAL" = 1 ] || rm -f "$W/out/unet_part$N.bin"
  "$PY" "$HERE/upload_hf.py" --repo "$HF_REPO" \
      --put "$W/stats/part$N.tsv" --as "$REMOTE_DIR/stats/part$N.tsv" || true
done

echo
echo "==> peak RSS by stage (MB), across all parts built on this machine"
awk -F'\t' '{ if ($2 > peak[$1]) peak[$1] = $2; secs[$1] += $3 }
     END { for (s in peak) printf "  %-9s %6d MB   %5d s total\n", s, peak[s]/1024, secs[s] }' \
    "$W"/stats/part*.tsv 2>/dev/null | sort || echo "  (no parts built this run)"
echo "==> https://huggingface.co/$HF_REPO/tree/main/$REMOTE_DIR"

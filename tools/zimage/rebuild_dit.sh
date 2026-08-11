#!/usr/bin/env bash
# Rebuild every DiT graph with head-chunked attention, and publish as it goes.
#
# WHY THIS EXISTS
#
# The published DiT does not load on a Snapdragon 8 Gen 2. The device says so
# exactly:
#
#   Failed to find available PD for contextId 1 on deviceId 0 coreId 0
#   with context size estimate 2171250944
#
# 2.02 GB for a graph whose weights are 490 MB. The remainder is the attention
# score tensor [1, heads, T, T]: at T = 4608 over 30 heads that is 1.27 GB live.
# Measured on device by what loaded and what did not --
#
#   caption branch  T =  512   0.51 GB   loads in 0.8 s
#   part 1          T = 4096   1.50 GB   loads in 4.5 s
#   part 2          T = 4608   2.17 GB   refused, no PD that size
#
# -- so the ceiling is between 1.5 and 2.17 GB and the term that crosses it is
# quadratic in sequence length. rope_real.py now computes attention
# ZIMAGE_ATTN_HEAD_CHUNK heads at a time (5 of 30), which puts the live score
# tensor at 212 MB. That is a regrouping of the arithmetic, not an
# approximation: bit-identical to single-shot attention, verified at chunk
# sizes 1, 3, 5, 6 and 30.
#
# It is a graph change, so every DiT context binary has to be built again. The
# text encoder does not: its sequence is 512 tokens and it already loads.
#
# HOW TO RUN IT
#
#   export HF_TOKEN=...
#   ./tools/zimage/rebuild_dit.sh                 # and leave it
#
# Safe to start again at any time, from anything. It bootstraps its own
# toolchain if the machine has none, asks the Hub what is already built, and
# starts from the first gap -- so a machine that dies mid-run costs one graph
# rather than the run. A lock file keeps two copies from fighting over the
# same work directory.
#
# Roughly 35 minutes per graph, 33 graphs, so most of a day on one machine.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
W="${W:-/data/zwork}"
HF_REPO="${HF_REPO:-P2Enjoy/z-image-turbo-qnn}"
NPARTS="${NPARTS:-32}"
WBITS="${WBITS:-4}"
ARCH="${ARCH:-v73}"
LOG="${LOG:-$W/rebuild.log}"

# The attention grouping is part of the artifact, so it is part of the path.
# The published n32 graphs are numerically identical to these and differ only
# in how much memory they ask the device for -- which is exactly the kind of
# difference that a shared directory would hide until it failed on someone's
# phone.
export ZIMAGE_ATTN_HEAD_CHUNK="${ZIMAGE_ATTN_HEAD_CHUNK:-5}"
REMOTE_DIR="${REMOTE_DIR:-partial/n${NPARTS}-attn${ZIMAGE_ATTN_HEAD_CHUNK}}"

mkdir -p "$W"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# One at a time. This is restarted by a scheduler as well as by hand, and two
# copies sharing $W would delete each other's intermediates mid-stage.
LOCK="$W/.rebuild.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -f "$LOCK/pid" ] && kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then
    say "another rebuild is running (pid $(cat "$LOCK/pid")); nothing to do"
    exit 0
  fi
  say "clearing a lock left by a dead run"
  rm -rf "$LOCK"; mkdir "$LOCK" || exit 1
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

say "=== DiT rebuild, attention in groups of $ZIMAGE_ATTN_HEAD_CHUNK heads ==="
say "work=$W  repo=$HF_REPO  remote=$REMOTE_DIR  arch=$ARCH  w${WBITS}a16"

# ---------------------------------------------------------------------------
# 1. Toolchain. The machine this lands on may have nothing: these containers
#    are reclaimed on idle and come back empty. Bootstrapping here rather than
#    in a README is what makes "start it again" a complete instruction.
# ---------------------------------------------------------------------------
export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/data/qairt/2.39.0.250926}"
PY="${PY:-/data/zenv/exportvenv/bin/python}"
QNN="${QNN:-/data/zenv/qnnvenv/bin/python}"
if [ ! -x "$PY" ] || [ ! -x "$QNN" ] || [ ! -d "$QNN_SDK_ROOT" ]; then
  say "toolchain missing, bootstrapping (SDK + both venvs, ~20 min)"
  WORK="$W/.envtmp" bash "$HERE/setup_qnn_env.sh" || { say "bootstrap FAILED"; exit 1; }
fi
for t in "$PY" "$QNN"; do
  [ -x "$t" ] || { say "still no interpreter at $t"; exit 1; }
done
[ -n "${HF_TOKEN:-}" ] || say "WARNING: HF_TOKEN unset; uploads will fail"

# ---------------------------------------------------------------------------
# 2. What is already built. One listing for the run; upload_hf.py retries and
#    exits non-zero rather than reporting an empty repo, so a network blip
#    cannot read as "nothing is published" and rebuild everything.
# ---------------------------------------------------------------------------
DONE="$("$PY" "$HERE/upload_hf.py" --repo "$HF_REPO" --list-remote "$REMOTE_DIR/")" || {
  say "could not list $HF_REPO; refusing to guess. Try again later."; exit 1; }
say "already built: $(echo "$DONE" | grep -c 'unet_' || true) of $((NPARTS + 1)) graphs"

# Part 2 first: it is the graph that failed on the device, so if head chunking
# has not moved the number there is no reason to build the other 32. Then the
# caption branch (cheap, 512 tokens) and part 1, then the rest in order.
ORDER="${ORDER:-2 cap 1 $(seq 3 "$NPARTS")}"

built=0; failed=0
for N in $ORDER; do
  if [ "$N" = cap ]; then STEM="unet_cap"; else STEM="unet_part$N"; fi
  if echo "$DONE" | grep -qx "$REMOTE_DIR/$STEM.bin"; then
    say "$STEM already published, skipping"
    continue
  fi

  say "--- $STEM (free $(df -BG --output=avail "$W" | tail -1 | tr -dc '0-9')G) ---"
  # Every stage of the previous graph is gone by now; anything left is debris
  # from a run that was killed mid-stage, and it would be read as a resume
  # point for the graph we are about to build under a different name.
  rm -rf "$W/onnx" "$W"/*.dlc "$W"/calib* 2>/dev/null

  if PY="$PY" QNN="$QNN" ZIMAGE_ATTN_HEAD_CHUNK="$ZIMAGE_ATTN_HEAD_CHUNK" \
      "$HERE/convert_part.sh" "$W" "$N" "$NPARTS" "$WBITS" "$ARCH"; then
    if "$PY" "$HERE/upload_hf.py" --repo "$HF_REPO" \
        --put "$W/out/$STEM.bin" --as "$REMOTE_DIR/$STEM.bin"; then
      "$PY" "$HERE/upload_hf.py" --repo "$HF_REPO" \
          --put "$W/stats/$STEM.tsv" --as "$REMOTE_DIR/stats/$STEM.tsv" || true
      rm -f "$W/out/$STEM.bin"
      built=$((built + 1))
      say "$STEM done and published"
    else
      say "$STEM built but UPLOAD FAILED; keeping the local copy for the retry"
      failed=$((failed + 1))
    fi
  else
    # Keep going. One graph that will not build should not stop the other 32,
    # and the next run picks it up from the gap in the listing.
    say "$STEM FAILED to build; continuing with the rest"
    failed=$((failed + 1))
  fi
done

say "=== built $built, failed $failed this run ==="

# ---------------------------------------------------------------------------
# 3. Assemble model/ once every graph exists. publish_model.py is the gate: it
#    names exactly which pieces are missing rather than publishing a directory
#    that is short a part.
# ---------------------------------------------------------------------------
if "$PY" "$HERE/publish_model.py" --repo "$HF_REPO" --dit-dir "$REMOTE_DIR" --dry-run; then
  say "every graph present -- publishing model/"
  "$PY" "$HERE/publish_model.py" --repo "$HF_REPO" --dit-dir "$REMOTE_DIR" \
    && say "model/ updated; the app will download the rebuilt DiT"
else
  say "not complete yet; run this again"
fi
say "https://huggingface.co/$HF_REPO/tree/main/$REMOTE_DIR"

#!/usr/bin/env bash
# One Qwen3-4B text encoder part: remote weights -> ONNX -> DLC -> quantized
# DLC -> context binary, then upload and delete.
#
# Deliberately a separate script from convert_part.sh rather than a flag on it.
# convert_all.sh re-invokes convert_part.sh once per DiT part, so editing that
# file while a 30-part run is in flight changes the script for every part that
# has not started yet. Two files cost a little duplication and remove that
# entire class of accident.
#
# The encoder is far cheaper to split than the DiT: it runs once per prompt and
# the result is prompt-cached, so extra parts cost load time once per image
# rather than once per step. Its activations are also tiny by comparison --
# sequence 512 against the DiT's 4608, so attention is 512^2 rather than
# 4608^2, which is ~81x less per head.
#
#   PY=... QNN=... ./convert_clip.sh <work_dir> <part> <n_parts> [bits] [arch]
set -euo pipefail

W="${1:?work dir}"; N="${2:?part number}"; NPARTS="${3:?total parts}"
WBITS="${4:-4}"; ARCH="${5:-v73}"
SCRATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/data/qairt/2.39.0.250926}"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
BIN="$QNN_SDK_ROOT/bin/x86_64-linux-clang"
PY="${PY:?set PY to the torch venv python}"
QNN="${QNN:?set QNN to the python3.10 venv python}"

ODIR="$W/onnx/clip_part$N"; OUT="$W/out"; STATS="$W/stats"; mkdir -p "$OUT" "$STATS"
free_gb() { df -BG --output=avail "$W" | tail -1 | tr -dc '0-9'; }
say() { echo "[clip$N/$NPARTS] $* (free $(free_gb)G)"; }

MIN_BIN_BYTES="${MIN_BIN_BYTES:-1048576}"
bin_ok() { [ -f "$1" ] && [ "$(stat -c%s "$1")" -ge "$MIN_BIN_BYTES" ]; }

TSV="$STATS/clip_part$N.tsv"
run() {
  local stage="$1"; shift
  local t0 tmp rc kb secs
  tmp="$(mktemp)"; t0=$SECONDS
  set +e; /usr/bin/time -v -o "$tmp" "$@"; rc=$?; set -e
  kb=$(awk '/Maximum resident set size/ {print $NF}' "$tmp")
  secs=$((SECONDS - t0))
  printf '%s\t%s\t%s\t%s\n' "$stage" "${kb:-0}" "$secs" "$rc" >> "$TSV"
  say "  $stage: peak $(( ${kb:-0} / 1024 )) MB, ${secs}s, rc=$rc"
  rm -f "$tmp"; return $rc
}

if bin_ok "$OUT/clip_part$N.bin"; then say "already built, skipping"; exit 0; fi
rm -f "$OUT/clip_part$N.bin"
: > "$TSV"

say "1/5 export ONNX"
run export "$PY" "$SCRATCH/export_clip.py" --work "$W" --parts "$NPARTS" --only "$N"
[ -f "$ODIR/clip_part$N.onnx" ] || { echo "export produced no ONNX"; exit 1; }
say "2/5 ONNX built"

# Same calibration generator as the DiT: attention_mask must be a real 0/1
# indicator, not noise, because the quantizer sets every activation's range by
# running the graph on these tensors.
"$PY" "$SCRATCH/make_calib.py" "$ODIR/clip_part$N.onnx" "$W/ccalib$N" "$W/ccalib$N.txt"

run convert "$QNN" "$BIN/qairt-converter" --input_network "$ODIR/clip_part$N.onnx" \
    --output_path "$W/clip_part$N.dlc" --preserve_io_datatype \
    > "$W/cconvert$N.log" 2>&1 \
    || { echo "convert FAILED:"; tail -20 "$W/cconvert$N.log"; exit 1; }
[ -s "$W/clip_part$N.dlc" ] || {
    echo "converter exited 0 but produced no DLC (free $(free_gb)G):"
    tail -20 "$W/cconvert$N.log"; exit 1; }
rm -rf "$ODIR"
say "3/5 DLC built, ONNX released"

# No --pack_4_bit_weights: Hexagon v73's FullyConnected accepts no 4-bit weight
# datatype, so packing produces a graph the HTP rejects. See convert_part.sh.
run quantize "$QNN" "$BIN/qairt-quantizer" --input_dlc "$W/clip_part$N.dlc" \
    --output_dlc "$W/clip_part${N}_q.dlc" --input_list "$W/ccalib$N.txt" \
    --weights_bitwidth "$WBITS" --act_bitwidth 16 --bias_bitwidth 32 \
    > "$W/cquant$N.log" 2>&1 || { echo "quantize FAILED:"; tail -20 "$W/cquant$N.log"; exit 1; }
[ -s "$W/clip_part${N}_q.dlc" ] || { echo "no quantized DLC:"; tail -20 "$W/cquant$N.log"; exit 1; }
rm -f "$W/clip_part$N.dlc"; rm -rf "$W/ccalib$N"
say "4/5 quantized w${WBITS}a16, float DLC released"

cat > "$W/htp_$ARCH.json" <<J
{"devices":[{"dsp_arch":"$ARCH","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
J
cat > "$W/ext_$ARCH.json" <<J
{"backend_extensions":{"shared_library_path":"libQnnHtpNetRunExtensions.so","config_file_path":"$(cd "$W" && pwd)/htp_$ARCH.json"}}
J
run context "$BIN/qnn-context-binary-generator" --dlc_path "$W/clip_part${N}_q.dlc" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file "$W/ext_$ARCH.json" --output_dir "$OUT" \
    --binary_file "clip_part$N" > "$W/cctx$N.log" 2>&1 \
    || { echo "context binary FAILED:"; tail -20 "$W/cctx$N.log"; exit 1; }
bin_ok "$OUT/clip_part$N.bin" || {
    echo "no usable .bin (got $(stat -c%s "$OUT/clip_part$N.bin" 2>/dev/null || echo 0) bytes):"
    tail -20 "$W/cctx$N.log"; exit 1; }
rm -f "$W/clip_part${N}_q.dlc"
say "5/5 done -> $(stat -c%s "$OUT/clip_part$N.bin") bytes"

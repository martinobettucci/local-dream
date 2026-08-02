#!/usr/bin/env bash
# One DiT part: remote weights -> ONNX -> DLC -> quantized DLC -> context binary.
#
# Two resources cap this, and they pull in opposite directions:
#
#   RAM   qairt-quantizer holds the whole graph plus one calibration sample's
#         activations. It was OOM-killed at 19 GB (15 GB + 4 GB swap) on a
#         4-block part. Smaller parts are the only lever.
#   Disk  every stage's input is deleted as soon as its output exists, which is
#         not tidiness — the intermediates for one part add up to more than the
#         free space they run in.
#
# Peak RSS of every stage is reported so the part count can be chosen from
# measurements rather than guesses.
#
#   PY=... QNN=... ./convert_part.sh <work_dir> <part> <n_parts> [bits] [dsp_arch]
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

ODIR="$W/onnx/part$N"; OUT="$W/out"; STATS="$W/stats"; mkdir -p "$OUT" "$STATS"
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
say() { echo "[part$N/$NPARTS] $* (free $(free_gb)G)"; }

# Peak RSS and wall clock per stage, appended to a per-part TSV. This is the
# measurement that decides how many parts the split needs; without it the answer
# is a guess that costs an hour per iteration to test.
TSV="$STATS/part$N.tsv"
run() {
  local stage="$1"; shift
  local t0 tmp rc
  tmp="$(mktemp)"
  t0=$SECONDS
  set +e
  /usr/bin/time -v -o "$tmp" "$@"
  rc=$?
  set -e
  local kb secs
  kb=$(awk '/Maximum resident set size/ {print $NF}' "$tmp")
  secs=$((SECONDS - t0))
  printf '%s\t%s\t%s\t%s\n' "$stage" "${kb:-0}" "$secs" "$rc" >> "$TSV"
  say "  $stage: peak $(( ${kb:-0} / 1024 )) MB, ${secs}s, rc=$rc"
  rm -f "$tmp"
  return $rc
}

if [ -f "$OUT/unet_part$N.bin" ]; then say "already built, skipping"; exit 0; fi
: > "$TSV"

say "1/5 export ONNX"
run export "$PY" "$SCRATCH/export_dit.py" --work "$W" --parts "$NPARTS" --only "$N"
[ -f "$ODIR/unet_part$N.onnx" ] || { echo "export produced no ONNX"; exit 1; }
say "2/5 ONNX built"

# Calibration inputs must match the graph's declared names, shapes and dtypes,
# so they are read off the ONNX itself rather than hardcoded — and therefore
# built while it still exists. pos_ids is int32, everything else fp32. ONE
# sample only: qairt-quantizer holds activations for every sample and OOM-killed
# a 15 GB box with four.
"$QNN" - "$W" "$N" "$ODIR/unet_part$N.onnx" <<'PY'
import numpy as np, os, sys, onnx
W, N, path = sys.argv[1], sys.argv[2], sys.argv[3]
c = f"{W}/calib{N}"; os.makedirs(c, exist_ok=True)
m = onnx.load(path, load_external_data=False)
rng = np.random.default_rng(0); parts = []
for i in m.graph.input:
    dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
    p = os.path.abspath(f"{c}/{i.name}.raw")
    if i.type.tensor_type.elem_type == onnx.TensorProto.INT32:
        np.zeros(dims, dtype=np.int32).tofile(p)
    else:
        rng.standard_normal(dims, dtype=np.float32).tofile(p)
    parts.append(f"{i.name}:={p}")
open(f"{W}/calib{N}.txt", "w").write(" ".join(parts) + "\n")
print("  calib inputs:", " ".join(x.split(":=")[0] for x in parts))
PY

# Never pipe a stage through tail: the pipeline's exit status is tail's, so a
# failed converter looks like success under `set -e`. Log in full, then verify
# the artifact actually exists before deleting anything upstream of it — an
# earlier version reported "INFO_CONVERSION_SUCCESS" while writing no DLC, and
# had already destroyed the fp16 source by then. That turned out to be how
# qairt-converter reports running out of disk.
run convert "$QNN" "$BIN/qairt-converter" --input_network "$ODIR/unet_part$N.onnx" \
    --output_path "$W/unet_part$N.dlc" --preserve_io_datatype \
    > "$W/convert$N.log" 2>&1 \
    || { echo "convert FAILED:"; tail -20 "$W/convert$N.log"; exit 1; }
if [ ! -s "$W/unet_part$N.dlc" ]; then
    echo "converter exited 0 but produced no DLC (free $(free_gb)G) — last lines:"
    tail -20 "$W/convert$N.log"; exit 1
fi
# Only now is the ONNX redundant.
rm -rf "$ODIR"
say "3/5 DLC built, ONNX released"

run quantize "$QNN" "$BIN/qairt-quantizer" --input_dlc "$W/unet_part$N.dlc" \
    --output_dlc "$W/unet_part${N}_q.dlc" --input_list "$W/calib$N.txt" \
    --weights_bitwidth "$WBITS" --act_bitwidth 16 --bias_bitwidth 32 \
    > "$W/quant$N.log" 2>&1 || { echo "quantize FAILED:"; tail -20 "$W/quant$N.log"; exit 1; }
[ -s "$W/unet_part${N}_q.dlc" ] || { echo "no quantized DLC:"; tail -20 "$W/quant$N.log"; exit 1; }
rm -f "$W/unet_part$N.dlc"; rm -rf "$W/calib$N"
say "4/5 quantized w${WBITS}a16, float DLC released"

cat > "$W/htp_$ARCH.json" <<J
{"devices":[{"dsp_arch":"$ARCH","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
J
cat > "$W/ext_$ARCH.json" <<J
{"backend_extensions":{"shared_library_path":"libQnnHtpNetRunExtensions.so","config_file_path":"$(cd "$W" && pwd)/htp_$ARCH.json"}}
J
run context "$BIN/qnn-context-binary-generator" --dlc_path "$W/unet_part${N}_q.dlc" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file "$W/ext_$ARCH.json" --output_dir "$OUT" \
    --binary_file "unet_part$N" > "$W/ctx$N.log" 2>&1 \
    || { echo "context binary FAILED:"; tail -20 "$W/ctx$N.log"; exit 1; }
[ -s "$OUT/unet_part$N.bin" ] || { echo "no .bin:"; tail -20 "$W/ctx$N.log"; exit 1; }
rm -f "$W/unet_part${N}_q.dlc"
say "5/5 done -> $(stat -c%s "$OUT/unet_part$N.bin") bytes"
printf 'peak RSS across stages: %s MB\n' \
    "$(awk -F'\t' 'BEGIN{m=0} $2>m{m=$2} END{print int(m/1024)}' "$TSV")"

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

W="${1:?work dir}"; N="${2:?part number, or \"cap\"}"; NPARTS="${3:?total parts}"
WBITS="${4:-4}"; ARCH="${5:-v73}"
SCRATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/data/qairt/2.39.0.250926}"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
BIN="$QNN_SDK_ROOT/bin/x86_64-linux-clang"
PY="${PY:?set PY to the torch venv python}"
QNN="${QNN:?set QNN to the python3.10 venv python}"

# The caption branch (cap_embedder + context_refiner) is a graph of its own,
# built by exactly these five stages under a different name. See
# CAP_INPUT_NAMES in static_export.py for why it is separate at all.
if [ "$N" = "cap" ]; then
  STEM="unet_cap"; ODIR="$W/onnx/cap"; EXPORT_ARGS=(--cap)
else
  STEM="unet_part$N"; ODIR="$W/onnx/part$N"; EXPORT_ARGS=(--only "$N")
fi
OUT="$W/out"; STATS="$W/stats"; mkdir -p "$OUT" "$STATS"
free_gb() { df -BG --output=avail "$W" | tail -1 | tr -dc '0-9'; }
say() { echo "[$STEM/$NPARTS] $* (free $(free_gb)G)"; }

# A context binary is hundreds of MB; anything much smaller is a run that died
# mid-write, and the resume gate must not accept it -- convert_all.sh would
# publish the truncated file and delete the local copy, making the damage
# permanent and invisible.
MIN_BIN_BYTES="${MIN_BIN_BYTES:-1048576}"
bin_ok() { [ -f "$1" ] && [ "$(stat -c%s "$1")" -ge "$MIN_BIN_BYTES" ]; }

# Peak RSS and wall clock per stage, appended to a per-part TSV. This is the
# measurement that decides how many parts the split needs; without it the answer
# is a guess that costs an hour per iteration to test.
TSV="$STATS/$STEM.tsv"
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

if bin_ok "$OUT/$STEM.bin"; then say "already built, skipping"; exit 0; fi
rm -f "$OUT/$STEM.bin"   # present but too small: a dead run, not a result
: > "$TSV"

say "1/5 export ONNX"
run export "$PY" "$SCRATCH/export_dit.py" --work "$W" --parts "$NPARTS" "${EXPORT_ARGS[@]}"
[ -f "$ODIR/$STEM.onnx" ] || { echo "export produced no ONNX"; exit 1; }
say "2/5 ONNX built"

# Calibration inputs are read off the ONNX (names, shapes and dtypes have to
# match exactly) and therefore built while it still exists. They are NOT random:
# make_calib.py gives timestep, pos_ids and the masks their real runtime
# distributions, because the quantizer derives every activation's range from
# these and a wrong one produces a model that runs and generates garbage.
# ONE sample: qairt-quantizer holds activations for every sample and OOM-killed
# a 15 GB box with four.
#
# Run with $PY, not $QNN: the positions come from static_export.build_positions,
# the same function the export uses and the C++ mirrors, so it needs torch. That
# is deliberate -- reimplementing the coordinate arithmetic here in numpy would
# be a second source of truth for the one piece of this pipeline whose earlier
# divergence measured 15.6 % error.
"$PY" "$SCRATCH/make_calib.py" "$ODIR/$STEM.onnx" "$W/calib$N" "$W/calib$N.txt"

# Never pipe a stage through tail: the pipeline's exit status is tail's, so a
# failed converter looks like success under `set -e`. Log in full, then verify
# the artifact actually exists before deleting anything upstream of it — an
# earlier version reported "INFO_CONVERSION_SUCCESS" while writing no DLC, and
# had already destroyed the fp16 source by then. That turned out to be how
# qairt-converter reports running out of disk.
run convert "$QNN" "$BIN/qairt-converter" --input_network "$ODIR/$STEM.onnx" \
    --output_path "$W/$STEM.dlc" --preserve_io_datatype \
    > "$W/convert$N.log" 2>&1 \
    || { echo "convert FAILED:"; tail -20 "$W/convert$N.log"; exit 1; }
if [ ! -s "$W/$STEM.dlc" ]; then
    echo "converter exited 0 but produced no DLC (free $(free_gb)G) — last lines:"
    tail -20 "$W/convert$N.log"; exit 1
fi
# Only now is the ONNX redundant.
rm -rf "$ODIR"
say "3/5 DLC built, ONNX released"

# DO NOT add --pack_4_bit_weights. It is tempting -- it halves the DLC, from
# 181,246,412 to 90,806,732 bytes on this part, by switching the weight tensor
# from uFxp_8 to the genuinely packed QNN_DATATYPE_UFIXED_POINT_4 -- and the HTP
# then refuses to run it:
#
#   Unsupported input/output datatypes requested for the HTP Op 'FullyConnected'
#     in[0]:QNN_DATATYPE_UFIXED_POINT_16
#     in[1]:QNN_DATATYPE_UFIXED_POINT_4      <- the packed weights
#     out[0]:QNN_DATATYPE_UFIXED_POINT_16
#
# The backend then lists every combination it does accept, and across all three
# configurations (FP16 / INT16 / INT8) the weight datatypes are only ever
# FLOAT_16, FLOAT_32, SFIXED_POINT_8/16 and UFIXED_POINT_8/16. There is no
# 4-bit weight datatype on Hexagon v73 at all.
#
# So the working form is 4-bit values in an 8-bit container, which is what
# --weights_bitwidth 4 alone produces. That is also what the HTP's own int4
# guidelines mean by "int4 encodings": the benefit is lower VTCM traffic, not a
# smaller tensor type.
#
# Measured on this part, one float DLC quantized four ways:
#     float                724,077,604 B   fp32
#     w8a16                181,246,372 B   uFxp_8, bitwidth 8   runs
#     w4a16                181,246,412 B   uFxp_8, bitwidth 4   runs
#     w2a16                181,246,412 B   uFxp_8, bitwidth 2   runs, pointless
#     w4a16 --pack_4_...    90,806,732 B   uFxp_4, bitwidth 4   REJECTED by HTP
#
# All three runnable widths are the same size on disk. w4 is kept over w8 for
# the documented VTCM latency benefit, not for bytes; switch to 8 if quality
# turns out to be the binding problem, at no size cost.
run quantize "$QNN" "$BIN/qairt-quantizer" --input_dlc "$W/$STEM.dlc" \
    --output_dlc "$W/${STEM}_q.dlc" --input_list "$W/calib$N.txt" \
    --weights_bitwidth "$WBITS" --act_bitwidth 16 --bias_bitwidth 32 \
    > "$W/quant$N.log" 2>&1 || { echo "quantize FAILED:"; tail -20 "$W/quant$N.log"; exit 1; }
[ -s "$W/${STEM}_q.dlc" ] || { echo "no quantized DLC:"; tail -20 "$W/quant$N.log"; exit 1; }
rm -f "$W/$STEM.dlc"; rm -rf "$W/calib$N"
say "4/5 quantized w${WBITS}a16, float DLC released"

cat > "$W/htp_$ARCH.json" <<J
{"devices":[{"dsp_arch":"$ARCH","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
J
cat > "$W/ext_$ARCH.json" <<J
{"backend_extensions":{"shared_library_path":"libQnnHtpNetRunExtensions.so","config_file_path":"$(cd "$W" && pwd)/htp_$ARCH.json"}}
J
run context "$BIN/qnn-context-binary-generator" --dlc_path "$W/${STEM}_q.dlc" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file "$W/ext_$ARCH.json" --output_dir "$OUT" \
    --binary_file "$STEM" > "$W/ctx$N.log" 2>&1 \
    || { echo "context binary FAILED:"; tail -20 "$W/ctx$N.log"; exit 1; }
bin_ok "$OUT/$STEM.bin" || { echo "no usable .bin (got $(stat -c%s "$OUT/$STEM.bin" 2>/dev/null || echo 0) bytes):"; tail -20 "$W/ctx$N.log"; exit 1; }
rm -f "$W/${STEM}_q.dlc"
say "5/5 done -> $(stat -c%s "$OUT/$STEM.bin") bytes"
printf 'peak RSS across stages: %s MB\n' \
    "$(awk -F'\t' 'BEGIN{m=0} $2>m{m=$2} END{print int(m/1024)}' "$TSV")"

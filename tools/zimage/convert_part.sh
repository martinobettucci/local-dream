#!/usr/bin/env bash
# One DiT part: fp16 weights -> ONNX -> DLC -> quantized DLC -> v73 context binary.
#
# Ordered so that peak disk stays under ~4 GB even though the intermediates add
# up to more than that: every stage's input is deleted as soon as its output
# exists. The 12.3 GB of per-part fp16 weights leaves very little headroom, so
# this is not optional tidiness.
#
#   ./convert_part.sh <work_dir> <part_number> [weights_bitwidth]
set -euo pipefail

W="${1:?work dir}"; N="${2:?part number}"; WBITS="${3:-4}"
SCRATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QNN_SDK_ROOT="${QNN_SDK_ROOT:-/data/qairt/2.39.0.250926}"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
BIN="$QNN_SDK_ROOT/bin/x86_64-linux-clang"
PY="${PY:?set PY to the torch venv python}"
QNN="${QNN:?set QNN to the python3.10 venv python}"

ODIR="$W/onnx/part$N"; OUT="$W/out"; mkdir -p "$OUT"
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
say() { echo "[part$N] $* (free $(free_gb)G)"; }

if [ -f "$OUT/unet_part$N.bin" ]; then say "already built, skipping"; exit 0; fi

say "1/5 export ONNX"
"$PY" "$SCRATCH/export_dit.py" --work "$W" --parts 8 --only "$N" 2>&1 | grep -E "^\[export_dit\]" || true
[ -f "$ODIR/unet_part$N.onnx" ] || { echo "export produced no ONNX"; exit 1; }
# The ONNX carries its own copy of the weights, so the fp16 source is dead now.
rm -f "$W/parts/part$N.safetensors"
say "2/5 fp16 source released"

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

"$QNN" "$BIN/qairt-converter" --input_network "$ODIR/unet_part$N.onnx" \
    --output_path "$W/unet_part$N.dlc" --preserve_io_datatype 2>&1 | tail -2
rm -rf "$ODIR"
say "3/5 DLC built, ONNX released"

"$QNN" "$BIN/qairt-quantizer" --input_dlc "$W/unet_part$N.dlc" \
    --output_dlc "$W/unet_part${N}_q.dlc" --input_list "$W/calib$N.txt" \
    --weights_bitwidth "$WBITS" --act_bitwidth 16 --bias_bitwidth 32 2>&1 | tail -2
rm -f "$W/unet_part$N.dlc"; rm -rf "$W/calib$N"
say "4/5 quantized w${WBITS}a16, float DLC released"

cat > "$W/htp_v73.json" <<'J'
{"devices":[{"dsp_arch":"v73","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
J
cat > "$W/ext_v73.json" <<J
{"backend_extensions":{"shared_library_path":"libQnnHtpNetRunExtensions.so","config_file_path":"$(cd "$W" && pwd)/htp_v73.json"}}
J
"$BIN/qnn-context-binary-generator" --dlc_path "$W/unet_part${N}_q.dlc" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file "$W/ext_v73.json" --output_dir "$OUT" \
    --binary_file "unet_part$N" 2>&1 | tail -3
rm -f "$W/unet_part${N}_q.dlc"
say "5/5 done -> $(ls -la "$OUT/unet_part$N.bin" | awk '{print $5}') bytes"

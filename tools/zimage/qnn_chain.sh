set -e
export QNN_SDK_ROOT=/data/qairt/2.39.0.250926
export PYTHONPATH=$QNN_SDK_ROOT/lib/python
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
QNN=./qnnvenv/bin/python
BIN=$QNN_SDK_ROOT/bin/x86_64-linux-clang
W=vaework
NAME=$1; WBITS=${2:-8}; CALIB_IN=${3:-latent}; CALIB_SHAPE=${4:-1,16,128,128}

echo "### 1/4 convert $NAME -> dlc"
time $QNN $BIN/qairt-converter --input_network $W/$NAME.onnx \
    --output_path $W/$NAME.dlc --preserve_io_datatype 2>&1 | tail -3

echo "### 2/4 calibration data"
$QNN - <<PY
import numpy as np, os
W="vaework"; os.makedirs(f"{W}/calib", exist_ok=True)
shape = tuple(int(x) for x in "$CALIB_SHAPE".split(","))
rng=np.random.default_rng(0); lines=[]
for i in range(4):
    p=f"{W}/calib/{i}.raw"
    rng.standard_normal(shape, dtype=np.float32).tofile(p)
    lines.append("$CALIB_IN:="+os.path.abspath(p))
open(f"{W}/calib_list.txt","w").write("\n".join(lines)+"\n")
print(f"  4 calibration samples {shape}")
PY

echo "### 3/4 quantize w${WBITS}a16"
time $QNN $BIN/qairt-quantizer --input_dlc $W/$NAME.dlc \
    --output_dlc $W/${NAME}_q.dlc --input_list $W/calib_list.txt \
    --weights_bitwidth $WBITS --act_bitwidth 16 --bias_bitwidth 32 2>&1 | tail -4

echo "### 4/4 context binary for Hexagon v73"
cat > $W/htp_v73.json <<'J'
{"devices":[{"dsp_arch":"v73","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
J
cat > $W/ext_v73.json <<J
{"backend_extensions":{"shared_library_path":"libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/$W/htp_v73.json"}}
J
time $BIN/qnn-context-binary-generator --dlc_path $W/${NAME}_q.dlc \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --config_file $W/ext_v73.json \
    --output_dir $W/out --binary_file ${NAME} 2>&1 | tail -6

ls -la $W/out/ 2>/dev/null

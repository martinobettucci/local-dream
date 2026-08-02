#!/usr/bin/env bash
# Bootstrap the Qualcomm AI Runtime SDK + the two Python environments the
# conversion needs.
#
# Two environments, not one, and they cannot be merged: the SDK's native
# extensions are compiled for Python 3.10 and need onnx < 1.16, while torch and
# a current diffusers want neither. See the notes on each below.
#
# Every step exists because leaving it out produces an error that does not name
# the real cause. Measured on Ubuntu 24.04 / x86_64.
#
#   ./tools/zimage/setup_qnn_env.sh              # SDK + both venvs
#   ./tools/zimage/setup_qnn_env.sh --sdk-only   # just the SDK (for APK builds)
set -euo pipefail

SDK_VER="${SDK_VER:-2.39.0.250926}"
SDK_ROOT="${SDK_ROOT:-/data/qairt}"
WORK="${WORK:-$PWD/.zimage-env}"
SDK_ONLY=0
[ "${1:-}" = "--sdk-only" ] && SDK_ONLY=1

echo "==> Qualcomm AI Runtime $SDK_VER -> $SDK_ROOT"

# 1. The SDK itself.
#    The "Community" edition needs NO Qualcomm account — it is a plain
#    unauthenticated download, 1.35 GB, unpacking to 3.1 GB. It carries the full
#    toolchain: qairt-converter, qnn-context-binary-generator, the
#    examples/QNN/SampleApp sources this app's CMakeLists copies, the
#    aarch64-android runtime libs, and Hexagon v66-v81.
if [ ! -d "$SDK_ROOT/$SDK_VER" ]; then
  mkdir -p "$WORK" "$SDK_ROOT"
  curl -sSL --max-time 1800 -o "$WORK/qairt.zip" \
    "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/$SDK_VER/v$SDK_VER.zip"
  unzip -q "$WORK/qairt.zip" -d "$WORK/sdk"
  mv "$WORK/sdk/qairt/$SDK_VER" "$SDK_ROOT/$SDK_VER"
  rm -rf "$WORK/qairt.zip" "$WORK/sdk"
fi

if [ "$SDK_ONLY" = 1 ]; then
  echo "==> SDK only: $SDK_ROOT/$SDK_VER"
  exit 0
fi

# 2. libc++. The SDK's native extensions link against LLVM's C++ runtime, which
#    the Community edition does not bundle and Ubuntu does not install by
#    default. Without it:
#        ImportError: libc++.so.1: cannot open shared object file
#    which then resurfaces as a misleading "circular import" traceback.
if ! ldconfig -p | grep -q "libc++.so.1"; then
  echo "==> installing libc++ runtime"
  apt-get install -y --no-install-recommends libc++1 libc++abi1
fi

# 3. Python 3.10, exactly. libDlModelToolsPy.so is compiled against 3.10:
#        ImportError: Python version mismatch: module was compiled for Python 3.10
#    3.11 and 3.12 both fail. Keep this venv separate from whatever runs the
#    torch/ONNX export — those are happy on newer Pythons.
if ! command -v python3.10 >/dev/null; then
  echo "==> installing python3.10"
  apt-get install -y --no-install-recommends python3.10 python3.10-venv
fi

if [ ! -x "$WORK/qnnvenv/bin/python" ]; then
  echo "==> creating converter venv (python3.10)"
  python3.10 -m venv "$WORK/qnnvenv"
  "$WORK/qnnvenv/bin/pip" -q install --upgrade pip
  # onnx MUST be < 1.16: the converter imports `onnx.mapping`, which 1.16
  # removed. With a newer onnx the SDK swallows the ImportError, leaves the
  # module as None, and fails much later with
  #     AttributeError: 'NoneType' object has no attribute 'AttributeProto'
  # numpy is pinned to what check-python-dependency asks for.
  "$WORK/qnnvenv/bin/pip" -q install \
    "onnx==1.15.0" "numpy==1.26.4" packaging pyyaml absl-py pydantic rich psutil scipy
fi

# 4. The export environment: torch, diffusers, and the HTTP bits that let
#    export_dit.py range-read weights instead of downloading 24.6 GB of shards.
#    CPU-only torch on purpose — the wheel is 200 MB instead of 2.5 GB, and
#    nothing in the conversion path uses CUDA. diffusers must be >= 0.39 for
#    ZImageTransformer2DModel.
if [ ! -x "$WORK/exportvenv/bin/python" ]; then
  echo "==> creating export venv (torch + diffusers)"
  python3 -m venv "$WORK/exportvenv"
  "$WORK/exportvenv/bin/pip" -q install --upgrade pip
  "$WORK/exportvenv/bin/pip" -q install --index-url https://download.pytorch.org/whl/cpu torch
  "$WORK/exportvenv/bin/pip" -q install \
    "diffusers>=0.39.0" transformers safetensors huggingface_hub accelerate \
    onnx numpy requests sentencepiece
fi

cat <<EOF

==> ready. Use it with:

  export QNN_SDK_ROOT=$SDK_ROOT/$SDK_VER
  export PYTHONPATH=\$QNN_SDK_ROOT/lib/python
  QNN=$WORK/qnnvenv/bin/python        # SDK tools: converter, quantizer
  PY=$WORK/exportvenv/bin/python      # torch/ONNX export

  \$QNN \$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter --help
  PY=\$PY QNN=\$QNN ./tools/zimage/convert_part.sh /path/to/work 1

Nothing here needs a GPU: qairt-converter and qnn-context-binary-generator are
host CPU compilers, and w4a16 calibration runs the ONNX graph on CPU.
EOF

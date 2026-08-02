#!/usr/bin/env bash
# Bootstrap a machine to build the Local Dream APK from a clean checkout.
#
# Everything here was learned by hitting the failure it prevents. The versions
# are not "latest that works" — several are the only version that works, and the
# comment on each says why. Ubuntu 24.04 / x86_64.
#
#   git clone https://github.com/martinobettucci/local-dream && cd local-dream
#   ./tools/zimage/setup_build_env.sh
#   ./tools/zimage/setup_build_env.sh --build      # ... and build the APK
#
# Roughly 12 GB of downloads and ~25 GB on disk once the NDK, the Gradle cache
# and a native build tree are all present. Budget 45 min on a cold machine.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SDK_ROOT="${ANDROID_SDK_ROOT:-/data/android-sdk}"
NDK_VER="${NDK_VER:-28.0.13004108}"
CMDLINE_VER="13114758"          # cmdline-tools 12.0, the current stable bundle
RUST_VER="$(sed -n 's/^channel = "\(.*\)"/\1/p' "$REPO_ROOT/rust-toolchain.toml")"
DO_BUILD=0
[ "${1:-}" = "--build" ] && DO_BUILD=1

say() { echo -e "\n==> $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 1. Host packages.
#    ninja + ccache are not optional in practice: the native tree is MNN plus
#    xtensor plus the QNN SampleApp, and a cold single-threaded make run is
#    measured in hours.
# ---------------------------------------------------------------------------
say "host packages"
if have apt-get; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    openjdk-21-jdk-headless git curl unzip zip ninja-build cmake ccache \
    build-essential pkg-config python3 file
fi
JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")}"
export JAVA_HOME
echo "    JAVA_HOME=$JAVA_HOME"

# ---------------------------------------------------------------------------
# 2. Android SDK.
#    AGP 9.1.1 wants compileSdk 37 and build-tools 37.0.0. The package names are
#    a trap: the SDK manager knows `platforms;android-37.0` (with the minor
#    version) rather than `platforms;android-37`, and asking for the latter
#    fails with a bare "Failed to find package" that reads like a network error.
# ---------------------------------------------------------------------------
say "Android SDK -> $SDK_ROOT"
export ANDROID_SDK_ROOT="$SDK_ROOT" ANDROID_HOME="$SDK_ROOT"
SDKMGR="$SDK_ROOT/cmdline-tools/latest/bin/sdkmanager"
if [ ! -x "$SDKMGR" ]; then
  mkdir -p "$SDK_ROOT/cmdline-tools"
  tmp="$(mktemp -d)"
  curl -sSL --max-time 900 -o "$tmp/cmdline.zip" \
    "https://dl.google.com/android/repository/commandlinetools-linux-${CMDLINE_VER}_latest.zip"
  unzip -q "$tmp/cmdline.zip" -d "$tmp"
  mv "$tmp/cmdline-tools" "$SDK_ROOT/cmdline-tools/latest"
  rm -rf "$tmp"
fi
yes | "$SDKMGR" --licenses >/dev/null 2>&1 || true
"$SDKMGR" --install \
  "platform-tools" \
  "platforms;android-37.0" \
  "build-tools;37.0.0" \
  "ndk;$NDK_VER" \
  "cmake;3.22.1" >/dev/null

# The native build script looks for the NDK at a fixed path rather than probing
# the SDK layout, so link it there instead of editing the script.
NDK_DIR="$SDK_ROOT/ndk/$NDK_VER"
if [ ! -e /data/android-ndk-r28 ] && [ -d "$NDK_DIR" ]; then
  mkdir -p /data && ln -sfn "$NDK_DIR" /data/android-ndk-r28
fi
export ANDROID_NDK_HOME="$NDK_DIR" ANDROID_NDK_ROOT="$NDK_DIR"
echo "    ndk $NDK_VER"

cat > "$REPO_ROOT/local.properties" <<EOF
sdk.dir=$SDK_ROOT
ndk.dir=$NDK_DIR
EOF

# ---------------------------------------------------------------------------
# 3. Rust, for tokenizers-cpp's `tokenizers-c` crate.
#    The pinned version is a two-sided constraint — see rust-toolchain.toml.
#    rustup honours that file automatically, but the android target has to be
#    installed for it explicitly.
# ---------------------------------------------------------------------------
say "Rust $RUST_VER + aarch64-linux-android"
export RUSTUP_HOME="${RUSTUP_HOME:-/data/rustup}" CARGO_HOME="${CARGO_HOME:-/data/cargo}"
export PATH="$CARGO_HOME/bin:$PATH"
if ! have rustup; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path \
    --default-toolchain "$RUST_VER" --profile minimal
fi
rustup toolchain install "$RUST_VER" --profile minimal >/dev/null 2>&1 || true
rustup target add --toolchain "$RUST_VER" aarch64-linux-android >/dev/null

# ---------------------------------------------------------------------------
# 4. Submodules. All ten are required to configure CMake; a missing one shows up
#    as a "file not found" on a header rather than as a submodule error. MNN
#    alone is most of the clone time, hence the shallow fetch.
# ---------------------------------------------------------------------------
say "submodules"
git -C "$REPO_ROOT" submodule update --init --recursive --depth 1 --jobs 4

# ---------------------------------------------------------------------------
# 5. QNN SDK. The native build copies headers and the SampleApp sources out of
#    it, so the APK needs it even though conversion happens elsewhere.
# ---------------------------------------------------------------------------
say "QNN SDK"
"$REPO_ROOT/tools/zimage/setup_qnn_env.sh" --sdk-only

cat <<EOF

==> ready. Environment for this shell:

  export JAVA_HOME=$JAVA_HOME
  export ANDROID_SDK_ROOT=$SDK_ROOT ANDROID_HOME=$SDK_ROOT
  export ANDROID_NDK_HOME=$NDK_DIR
  export PATH=$CARGO_HOME/bin:\$PATH

Build:

  cd $REPO_ROOT/app/src/main/cpp && ./build.sh     # native libs, ~40 min cold
  cd $REPO_ROOT && ./gradlew assembleBasicDebug    # APK

EOF

if [ "$DO_BUILD" = 1 ]; then
  say "native libraries"
  (cd "$REPO_ROOT/app/src/main/cpp" && ./build.sh)
  # basic (not filter) x debug: debug is signed with the standard debug key, so
  # it sideloads alongside a Play install instead of colliding with it.
  say "APK"
  (cd "$REPO_ROOT" && ./gradlew --no-daemon assembleBasicDebug)
  find "$REPO_ROOT/app/build/outputs/apk" -name "*.apk" -print
fi

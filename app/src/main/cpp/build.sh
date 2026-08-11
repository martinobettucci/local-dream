set -e
cmake --preset android-release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build --preset android-release

mkdir -p lib
cp -r ./build/android/qnnlibs ../assets/
mkdir -p ../jniLibs/arm64-v8a/
cp ./build/android/bin/arm64-v8a/libstable_diffusion_core.so ../jniLibs/arm64-v8a/

# Ship it stripped, keep the linked artifact whole.
#
# The link no longer passes -Wl,-s, because a binary with no symbols anywhere
# made every crash report a column of unresolvable hex (see CMakeLists). But
# the unstripped .so is 188 MB against 11.7 MB, and AGP does not strip it out
# of a debug APK -- measured, the APK went 134 MB -> 177 MB.
#
# So strip the shipped copy here and leave build/android/bin/ alone. llvm-strip
# drops .symtab and the debug sections but keeps .dynsym, which is what dladdr
# reads, so frames still name themselves on the device; anything inlined past
# that is symbolised here against the artifact that kept everything, at the
# same offsets, because it is the same link.
STRIP="$(find "${ANDROID_NDK_HOME:-$ANDROID_SDK_ROOT/ndk}" -name llvm-strip 2>/dev/null | head -1)"
if [ -n "$STRIP" ]; then
  "$STRIP" --strip-debug ../jniLibs/arm64-v8a/libstable_diffusion_core.so
  echo "stripped shipped copy: $(stat -c%s ../jniLibs/arm64-v8a/libstable_diffusion_core.so) bytes"
else
  echo "WARNING: llvm-strip not found; shipping an unstripped .so (large APK)" >&2
fi

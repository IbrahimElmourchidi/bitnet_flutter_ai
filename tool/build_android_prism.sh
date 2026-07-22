#!/usr/bin/env bash
# Cross-compile libprism_bridge.so for Android (arm64-v8a + x86_64) and copy it
# into android/src/main/jniLibs/<abi>/. This is the PrismML Q2_0 backend — a
# single self-contained .so (PrismML's llama.cpp fork linked static), separate
# from the bitnet.cpp .so trio. No libomp / dependency pre-load needed.
#
# Required env:
#   ANDROID_NDK       absolute path to a recent NDK (r26+; r28 tested)
# Optional env:
#   PRISM_LLAMA_CPP   path to a PrismML/llama.cpp clone (default: ../prism-llama.cpp)
#   ANDROID_API       minimum API level (default: 24)
#
# Usage:
#   ANDROID_NDK=$HOME/Android/Sdk/ndk/28.2.13676358 ./tool/build_android_prism.sh
set -euo pipefail

if [[ -z "${ANDROID_NDK:-}" ]]; then
    echo "ERROR: set ANDROID_NDK to your NDK install." >&2
    exit 1
fi
if [[ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]]; then
    echo "ERROR: '$ANDROID_NDK/build/cmake/android.toolchain.cmake' not found" >&2
    exit 1
fi

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRISM_LLAMA_CPP="${PRISM_LLAMA_CPP:-$PKG_ROOT/../prism-llama.cpp}"
ANDROID_API="${ANDROID_API:-24}"

if [[ ! -d "$PRISM_LLAMA_CPP" ]]; then
    echo "ERROR: PRISM_LLAMA_CPP=$PRISM_LLAMA_CPP does not exist." >&2
    echo "       git clone --branch prism https://github.com/PrismML-Eng/llama.cpp.git \"$PRISM_LLAMA_CPP\"" >&2
    exit 1
fi

JOBS="$(nproc 2>/dev/null || echo 4)"

build_abi () {
    local abi="$1"   # arm64-v8a | x86_64
    echo "=== Building Android $abi (prism) ==="
    local build_dir="$PKG_ROOT/native/build-android-prism-$abi"
    cmake -B "$build_dir" -G Ninja -S "$PKG_ROOT/native/prism" \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$abi" \
        -DANDROID_PLATFORM="android-$ANDROID_API" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPRISM_LLAMA_CPP="$PRISM_LLAMA_CPP"
    cmake --build "$build_dir" -- -j"$JOBS"

    local jni_dir="$PKG_ROOT/android/src/main/jniLibs/$abi"
    mkdir -p "$jni_dir"
    cp -f "$build_dir/libprism_bridge.so" "$jni_dir/libprism_bridge.so"
    echo "  ✓ wrote $jni_dir/libprism_bridge.so"
}

build_abi "arm64-v8a"
build_abi "x86_64"

echo
echo "Prism Android backend built. jniLibs now contain:"
ls -la "$PKG_ROOT"/android/src/main/jniLibs/*/libprism_bridge.so

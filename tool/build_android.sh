#!/usr/bin/env bash
# Cross-compile bitnet.cpp + libbitnet_bridge for Android (arm64-v8a + x86_64)
# and copy the resulting .so files into android/src/main/jniLibs/<abi>/.
#
# Required env:
#   ANDROID_NDK   absolute path to a recent NDK (r26+ tested with r28).
#
# Optional env:
#   BITNET_CPP    absolute path to a clone of microsoft/BitNet (default: ../bitnet.cpp)
#   ANDROID_API   minimum API level                              (default: 24)
#
# Usage:
#   ANDROID_NDK=$HOME/Android/Sdk/ndk/28.2.13676358 ./tool/build_android.sh
set -euo pipefail

if [[ -z "${ANDROID_NDK:-}" ]]; then
    echo "ERROR: set ANDROID_NDK to your NDK install (e.g. \$HOME/Android/Sdk/ndk/28.2.13676358)" >&2
    exit 1
fi
if [[ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]]; then
    echo "ERROR: '$ANDROID_NDK/build/cmake/android.toolchain.cmake' not found" >&2
    exit 1
fi

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BITNET_CPP="${BITNET_CPP:-$PKG_ROOT/../bitnet.cpp}"
ANDROID_API="${ANDROID_API:-24}"

if [[ ! -d "$BITNET_CPP" ]]; then
    echo "ERROR: BITNET_CPP=$BITNET_CPP does not exist." >&2
    echo "       Clone microsoft/BitNet with --recursive and set BITNET_CPP=<path>." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Apply the portable-i2s-kernels patch to the bitnet.cpp checkout.
#
# The pristine microsoft/BitNet i2_s matmul kernels only have AVX2 and NEON
# implementations (no scalar fallback), and the NEON path uses a wrong on-disk
# layout. Without this patch, libggml built for Android x86_64 (NDK baseline
# has no AVX2) contains empty i2_s kernels -> the model emits a runaway stream
# of "!". The patch adds a portable scalar fallback + a canonical-layout NEON
# path. Applied idempotently so re-running this script is safe.
# ---------------------------------------------------------------------------
PATCH="$PKG_ROOT/native/patches/0001-i2s-portable-kernels.patch"
if [[ ! -f "$PATCH" ]]; then
    echo "ERROR: kernel patch not found at $PATCH" >&2
    exit 1
fi
if git -C "$BITNET_CPP" apply --check "$PATCH" 2>/dev/null; then
    git -C "$BITNET_CPP" apply "$PATCH"
    echo "Applied i2s kernel patch to $BITNET_CPP"
elif git -C "$BITNET_CPP" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "i2s kernel patch already applied to $BITNET_CPP"
else
    echo "ERROR: i2s kernel patch does not apply cleanly to $BITNET_CPP." >&2
    echo "       The checkout may be modified or at an unexpected revision." >&2
    exit 1
fi

JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

build_abi () {
    local abi="$1"             # arm64-v8a | x86_64
    local codegen="$2"         # tl1 | tl2
    local extra_define="$3"    # -DBITNET_ARM_TL1=ON | -DBITNET_X86_TL2=OFF

    echo "==================================================="
    echo "Building Android $abi  (codegen=$codegen)"
    echo "==================================================="

    # 1. Codegen kernels for this arch.
    if [[ "$codegen" == "tl1" ]]; then
        ( cd "$BITNET_CPP" && python3 utils/codegen_tl1.py \
            --model bitnet_b1_58-3B \
            --BM 160,320,320 --BK 64,128,64 --bm 32,64,32 )
    else
        ( cd "$BITNET_CPP" && python3 utils/codegen_tl2.py \
            --model bitnet_b1_58-3B \
            --BM 160,320,320 --BK 96,96,96 --bm 32,32,32 )
    fi

    # 2. Build libllama.so / libggml.so.
    #
    # Optional: on x86_64, build the AVX2 code paths for much faster inference
    # on an emulator whose host CPU supports AVX2. OFF by default because a
    # binary with AVX2 instructions will SIGILL on a host CPU without them.
    # The portable scalar i2_s kernels (from the patch above) are always
    # correct regardless of this flag.
    local isa_define=""
    if [[ "$abi" == "x86_64" && "${BITNET_ANDROID_X86_AVX2:-0}" == "1" ]]; then
        isa_define="-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
        echo "  (x86_64) building with AVX2/FMA/F16C — requires an AVX2 host CPU"
    fi

    local build_dir="$BITNET_CPP/build-android-$abi"
    cmake -B "$build_dir" -G Ninja -S "$BITNET_CPP" \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$abi" \
        -DANDROID_PLATFORM="android-$ANDROID_API" \
        -DCMAKE_BUILD_TYPE=Release \
        $extra_define \
        $isa_define \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_NATIVE=OFF
    cmake --build "$build_dir" --target llama -- -j"$JOBS"

    # 3. Stage the install prefix the bridge expects.
    local stage="$BITNET_CPP/stage-android-$abi"
    rm -rf "$stage"
    mkdir -p "$stage/lib" "$stage/include"
    ln -sf "$build_dir/3rdparty/llama.cpp/src/libllama.so"     "$stage/lib/libllama.so"
    ln -sf "$build_dir/3rdparty/llama.cpp/ggml/src/libggml.so" "$stage/lib/libggml.so"
    ln -sf "$BITNET_CPP/3rdparty/llama.cpp/include/llama.h"    "$stage/include/llama.h"
    for h in "$BITNET_CPP"/3rdparty/llama.cpp/ggml/include/*.h; do
        ln -sf "$h" "$stage/include/$(basename "$h")"
    done

    # 4. Build libbitnet_bridge.so for this ABI.
    local bridge_build="$PKG_ROOT/native/build-android-$abi"
    rm -rf "$bridge_build"
    cmake -B "$bridge_build" -G Ninja -S "$PKG_ROOT/native" \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$abi" \
        -DANDROID_PLATFORM="android-$ANDROID_API" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_INSTALL_PREFIX="$stage"
    cmake --build "$bridge_build" -- -j"$JOBS"

    # 5. Copy the three .so files into the plugin's jniLibs/<abi>/ folder.
    local jni_dir="$PKG_ROOT/android/src/main/jniLibs/$abi"
    mkdir -p "$jni_dir"
    cp -L "$bridge_build/libbitnet_bridge.so"                       "$jni_dir/"
    cp -L "$build_dir/3rdparty/llama.cpp/src/libllama.so"           "$jni_dir/"
    cp -L "$build_dir/3rdparty/llama.cpp/ggml/src/libggml.so"       "$jni_dir/"

    # libggml.so links against the NDK's OpenMP runtime; ship it alongside.
    local ndk_arch
    case "$abi" in
        arm64-v8a) ndk_arch=aarch64 ;;
        x86_64)    ndk_arch=x86_64 ;;
        *) echo "unknown abi $abi" >&2; exit 1 ;;
    esac
    local omp
    omp=$(find "$ANDROID_NDK/toolchains/llvm/prebuilt" -path "*/lib/clang/*/lib/linux/$ndk_arch/libomp.so" | head -n1)
    if [[ -z "$omp" ]]; then
        echo "ERROR: could not find libomp.so for $abi inside the NDK" >&2
        exit 1
    fi
    cp -L "$omp" "$jni_dir/libomp.so"

    echo "  ✓ wrote $jni_dir/{libbitnet_bridge,libllama,libggml}.so"
}

build_abi "arm64-v8a" "tl1" "-DBITNET_ARM_TL1=ON"
build_abi "x86_64"    "tl2" "-DBITNET_X86_TL2=OFF"

echo
echo "All Android ABIs built. jniLibs contents:"
ls -la "$PKG_ROOT"/android/src/main/jniLibs/*/

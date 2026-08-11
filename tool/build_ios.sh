#!/usr/bin/env bash
# Cross-compile bitnet.cpp (llama + ggml) and libbitnet_bridge as STATIC
# libraries for Apple iOS, then bundle them into an .xcframework that the
# CocoaPods podspec vendors.
#
# Produces (under ios/Frameworks/):
#   bitnet_bridge.xcframework
#     ├── ios-arm64/                 (device)
#     └── ios-arm64_x86_64-simulator (simulator, fat)
#
# Each slice is a single static archive that merges:
#   libbitnet_bridge.a + libllama.a + libggml*.a
# so the app links one library and Dart's DynamicLibrary.process() resolves
# the bn_* symbols out of the app binary.
#
# Required tools: Xcode, cmake, ninja, python3, libtool, lipo (all standard on
# a machine with Xcode + Homebrew cmake/ninja).
#
# IMPORTANT — which bitnet.cpp tree to use:
#   Point BITNET_CPP at the SAME tree that built the working Android libs: the
#   one whose 3rdparty/llama.cpp is Eddie-Wang1120/llama.cpp with the classic
#   ggml-aarch64.c I2_S kernels (ggml_vec_dot_i2_i8_s_1x1/_1xN/_Nx1/_1x4_32W).
#   The newer microsoft/BitNet HEAD pins isHuangXin's embedding-model llama.cpp
#   branch, whose ggml-cpu-i2s.c kernel is for the 0.6B/270M *embedding* models
#   and mis-runs the 2B-4T *generation* GGUF (coherent first token, then garbage
#   — verified via native/smoke_test.c). This build's bitnet_bridge.cc targets
#   that classic tree's (older) llama C API.
#
# Optional env:
#   BITNET_CPP           path to the correct-kernel bitnet.cpp (see above).
#                        Default ../bitnet.cpp; the working tree used here was
#                        "../bitnet 2.cpp".
#   IOS_DEPLOY           minimum iOS version                    (default: 14.0)
#   BITNET_SKIP_PATCHES  set to 1 to skip the (embedding-branch-only) I2_S
#                        patches entirely. They no-op on the correct tree anyway.
#
# Usage:
#   BITNET_CPP="/Users/you/Desktop/bitnet 2.cpp" ./tool/build_ios.sh
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BITNET_CPP="${BITNET_CPP:-$PKG_ROOT/../bitnet.cpp}"
IOS_DEPLOY="${IOS_DEPLOY:-14.0}"

if [[ ! -d "$BITNET_CPP" ]]; then
    echo "ERROR: BITNET_CPP=$BITNET_CPP does not exist." >&2
    echo "       Clone microsoft/BitNet with --recursive and set BITNET_CPP=<path>." >&2
    exit 1
fi
if [[ ! -f "$BITNET_CPP/3rdparty/llama.cpp/CMakeLists.txt" ]]; then
    echo "ERROR: llama.cpp submodule not populated in $BITNET_CPP." >&2
    echo "       Run: git -C $BITNET_CPP submodule update --init --recursive" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Apply the portable-i2s-kernels patch (same one Android uses). Without it the
# NEON i2_s matmul kernel uses the wrong on-disk layout and the model emits a
# runaway stream of "!". Applied idempotently.
# ---------------------------------------------------------------------------
# apply_patch <git-dir> <patch-file>  — idempotent. Skips (does NOT error) when
# the patch is irrelevant to this tree: the "correct kernel" bitnet.cpp tree
# (the one that built the working Android libs, Eddie-Wang1120/llama.cpp with the
# classic ggml-aarch64.c I2_S kernels) already has the right kernels and does not
# need — and structurally cannot take — the embedding-branch fixups below.
apply_patch () {
    local dir="$1" patch="$2"
    if [[ ! -f "$patch" ]]; then
        echo "WARN: patch not found at $patch — skipping" >&2; return 0
    fi
    if [[ ! -d "$dir/.git" ]] && ! git -C "$dir" rev-parse --git-dir >/dev/null 2>&1; then
        echo "WARN: $dir is not a git checkout — skipping $(basename "$patch")" >&2; return 0
    fi
    if git -C "$dir" apply --check "$patch" 2>/dev/null; then
        git -C "$dir" apply "$patch"
        echo "Applied $(basename "$patch") to $dir"
    elif git -C "$dir" apply --check --reverse "$patch" 2>/dev/null; then
        echo "$(basename "$patch") already applied to $dir"
    else
        echo "NOTE: $(basename "$patch") does not apply to $dir — skipping" >&2
        echo "      (expected for the correct-kernel tree, which needs no I2_S patch)." >&2
    fi
}

# These patches only matter for the isHuangXin embedding-branch llama.cpp, whose
# I2_S kernels are for the 0.6B/270M embedding models and mis-run 2B-4T. For the
# correct-kernel tree they simply don't apply and are skipped. Set
# BITNET_SKIP_PATCHES=1 to bypass entirely.
if [[ "${BITNET_SKIP_PATCHES:-0}" != "1" ]]; then
    apply_patch "$BITNET_CPP" "$PKG_ROOT/native/patches/0001-i2s-portable-kernels.patch"
    apply_patch "$BITNET_CPP/3rdparty/llama.cpp" "$PKG_ROOT/native/patches/0002-i2s-arm-scalar-and-fastpath-gate.patch"
    # 0003 is NOT embedding-branch-specific: it fixes the classic tree's I2_S gemm
    # prefill path, which sizes a stack VLA as (rows-per-thread * ubatch tokens).
    # ggml's worker threads get the pthread default stack (512 KB on Darwin), so a
    # prompt long enough to fill the micro-batch overruns it and dies with SIGBUS
    # in the stack guard page inside ggml_vec_dot_i2_i8_s. REQUIRED for iOS.
    apply_patch "$BITNET_CPP/3rdparty/llama.cpp" "$PKG_ROOT/native/patches/0003-i2s-gemm-no-stack-vla.patch"
fi

# ---------------------------------------------------------------------------
# iOS-build hygiene for the classic (Eddie-Wang1120) llama.cpp tree. Under
# -DCMAKE_SYSTEM_NAME=iOS its examples/server subdirs emit install(TARGETS ...)
# for MACOSX_BUNDLE executables with no BUNDLE DESTINATION → configure FAILS.
# We only need the static `llama`/`ggml` libs, so:
#   (a) don't let the bitnet.cpp root force the server example ON, and
#   (b) gate add_subdirectory(examples) behind LLAMA_BUILD_EXAMPLES.
# Both edits are idempotent and only touch the local build-dep checkout (never
# committed). Newer trees already gate these, so the sed is a no-op there.
sanitize_cmake_for_ios () {
    local root_cml="$BITNET_CPP/CMakeLists.txt"
    local llama_cml="$BITNET_CPP/3rdparty/llama.cpp/CMakeLists.txt"
    if [[ -f "$root_cml" ]] && grep -q 'set(LLAMA_BUILD_SERVER ON CACHE BOOL "Build llama.cpp server" FORCE)' "$root_cml"; then
        perl -0pi -e 's/^set\(LLAMA_BUILD_SERVER ON CACHE BOOL "Build llama\.cpp server" FORCE\)/# [build_ios.sh] server example disabled for iOS static build\nset(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama.cpp server" FORCE)/m' "$root_cml"
        echo "Sanitized (server OFF): $root_cml"
    fi
    if [[ -f "$llama_cml" ]] && grep -qE '^add_subdirectory\(examples\)' "$llama_cml"; then
        perl -0pi -e 's/^add_subdirectory\(examples\)/if (LLAMA_BUILD_EXAMPLES)\n    add_subdirectory(examples)\nendif()/m' "$llama_cml"
        echo "Sanitized (examples gated): $llama_cml"
    fi
}
sanitize_cmake_for_ios

# ---------------------------------------------------------------------------
# NOTE ON KERNELS: the BitNet b1.58 2B-4T model is I2_S-quantised and runs
# entirely on the I2_S matmul kernels (always compiled; made portable by the
# patch above). The TL1/TL2 LUT kernels are a separate optimisation that is
# NOT used for this model, and their generated header (bitnet-lut-kernels.h)
# references ggml_tensor::backend / GGML_BACKEND_TYPE_CPU — both removed from
# the pinned llama.cpp fork — so enabling TL1/TL2 fails to compile. We build
# WITHOUT BITNET_ARM_TL1 / BITNET_X86_TL2 and rely on the I2_S path, matching
# how the runtime bridge (n_gpu_layers=0, I2_S) actually executes.
# ---------------------------------------------------------------------------

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# build_slice <tag> <sysroot> <arch> [extra cmake defines...]
#   tag: device-arm64 | sim-arm64 | sim-x86_64
build_slice () {
    local tag="$1"; shift
    local sysroot="$1"; shift
    local arch="$1"; shift
    local extra=("$@")

    echo "==================================================="
    echo "Building iOS slice: $tag  (sdk=$sysroot arch=$arch)"
    echo "==================================================="

    local common_cmake=(
        -G Ninja
        -DCMAKE_SYSTEM_NAME=iOS
        -DCMAKE_OSX_SYSROOT="$sysroot"
        -DCMAKE_OSX_ARCHITECTURES="$arch"
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOY"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=OFF
    )

    # 1. Build llama + ggml (static). Wipe any stale build dir so cached CMake
    #    defines (e.g. a prior TL1 run) don't leak into this configuration.
    #    NOTE: BLAS/Accelerate is deliberately OFF. When enabled, ggml routes
    #    quantized prefill matmuls through the BLAS backend, which dequantizes
    #    I2_S weights via a generic row-dequant + sgemm path never exercised on
    #    Android. I2_S stores its scale at the END of tensor data (see the
    #    portable-i2s-kernels patch), so that generic path yields garbage/flat
    #    logits and bn_prompt fails. Matching Android's CPU-only i2_s kernels.
    local build_dir="$BITNET_CPP/build-ios-$tag"
    rm -rf "$build_dir"
    cmake -B "$build_dir" -S "$BITNET_CPP" \
        "${common_cmake[@]}" \
        ${extra[@]+"${extra[@]}"} \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_COMMON=OFF \
        -DLLAMA_CURL=OFF \
        -DGGML_METAL=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_ACCELERATE=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_OPENMP=OFF
    cmake --build "$build_dir" --target llama -- -j"$JOBS"

    # 2. Stage the install prefix the bridge CMake expects.
    local stage="$BITNET_CPP/stage-ios-$tag"
    rm -rf "$stage"
    mkdir -p "$stage/lib" "$stage/include"
    # llama.cpp puts headers under 3rdparty/llama.cpp; the bridge only needs llama.h
    ln -sf "$BITNET_CPP/3rdparty/llama.cpp/include/llama.h" "$stage/include/llama.h"
    for h in "$BITNET_CPP"/3rdparty/llama.cpp/ggml/include/*.h; do
        ln -sf "$h" "$stage/include/$(basename "$h")"
    done
    # A dummy libllama so the bridge CMake's EXISTS check passes; the real
    # static archives are merged in step 4 (we don't link the bridge against
    # llama here — it's a static lib, symbols resolve at final app-link time).
    : > "$stage/lib/libllama.dylib"

    # 3. Build libbitnet_bridge.a (static) for this slice.
    local bridge_build="$PKG_ROOT/native/build-ios-$tag"
    rm -rf "$bridge_build"
    cmake -B "$bridge_build" -S "$PKG_ROOT/native" \
        "${common_cmake[@]}" \
        ${extra[@]+"${extra[@]}"} \
        -DIOS=ON \
        -DLLAMA_INSTALL_PREFIX="$stage"
    cmake --build "$bridge_build" --target bitnet_bridge_static -- -j"$JOBS"

    # 4. Merge bridge + llama + all ggml static archives into ONE archive.
    local merged_dir="$PKG_ROOT/native/merged-ios-$tag"
    rm -rf "$merged_dir"; mkdir -p "$merged_dir"
    local archives=()
    archives+=("$bridge_build/libbitnet_bridge.a")
    while IFS= read -r a; do archives+=("$a"); done < <(
        find "$build_dir" -name 'libllama.a' -o -name 'libggml*.a' | sort -u
    )
    echo "  merging ${#archives[@]} archives:"
    printf '    %s\n' "${archives[@]}"
    libtool -static -o "$merged_dir/libbitnet_bridge.a" "${archives[@]}"
    echo "  ✓ $merged_dir/libbitnet_bridge.a"
}

# arm64 device (I2_S kernels only — TL1/TL2 disabled, see note above)
build_slice "device-arm64" "iphoneos" "arm64"

# arm64 simulator
build_slice "sim-arm64" "iphonesimulator" "arm64"

# x86_64 simulator (portable scalar i2_s kernels from the patch)
build_slice "sim-x86_64" "iphonesimulator" "x86_64"

# ---------------------------------------------------------------------------
# Lipo the two simulator slices into one fat archive, then assemble the
# .xcframework (device + fat-simulator).
# ---------------------------------------------------------------------------
SIM_FAT="$PKG_ROOT/native/merged-ios-sim-fat"
rm -rf "$SIM_FAT"; mkdir -p "$SIM_FAT"
lipo -create \
    "$PKG_ROOT/native/merged-ios-sim-arm64/libbitnet_bridge.a" \
    "$PKG_ROOT/native/merged-ios-sim-x86_64/libbitnet_bridge.a" \
    -output "$SIM_FAT/libbitnet_bridge.a"

XCF="$PKG_ROOT/ios/Frameworks/bitnet_bridge.xcframework"
rm -rf "$XCF"

# xcodebuild's -headers takes a DIRECTORY, not a file. Handing it the .h
# directly makes it copy the file *to* <slice>/Headers, so each slice's
# declared HeadersPath resolves to a regular file instead of a header search
# directory and `#import <bitnet_bridge/bitnet_bridge.h>` cannot resolve.
# Nothing imports it today (the keep-alive shim in bitnet_force_link.m declares
# the bn_* prototypes extern on purpose), but the xcframework should still be
# well-formed. Stage a real directory. Named build-* so the root .gitignore's
# native/build-*/ rule already covers it.
HDR_DIR="$PKG_ROOT/native/build-ios-headers"
rm -rf "$HDR_DIR"; mkdir -p "$HDR_DIR"
cp "$PKG_ROOT/native/bitnet_bridge.h" "$HDR_DIR/bitnet_bridge.h"

xcodebuild -create-xcframework \
    -library "$PKG_ROOT/native/merged-ios-device-arm64/libbitnet_bridge.a" \
        -headers "$HDR_DIR" \
    -library "$SIM_FAT/libbitnet_bridge.a" \
        -headers "$HDR_DIR" \
    -output "$XCF"

echo
echo "✓ Built $XCF"
find "$XCF" -name '*.a' -exec lipo -info {} \;

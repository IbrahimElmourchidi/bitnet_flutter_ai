#!/usr/bin/env bash
# Build the PrismML Q2_0 backend (prism_bridge + PrismML's llama.cpp fork linked
# static) as a DYNAMIC framework for Apple iOS, packaged into an .xcframework
# that the CocoaPods podspec vendors alongside bitnet_bridge.xcframework.
#
# Why a framework (not a merged .a like the bitnet backend): both backends
# export the same seven bn_* symbols and each carries its own incompatible
# ggml/llama. Static-linking both into the app binary would collide. Shipping
# prism as a separate dynamic framework — loaded by path via
# DynamicLibrary.open('prism_bridge.framework/prism_bridge') — keeps the fork's
# ggml symbols private (two-level namespace + -exported_symbols_list) so only
# prism's bn_* are visible through that specific handle.
#
# Produces:
#   ios/Frameworks/prism_bridge.xcframework
#     ├── ios-arm64/prism_bridge.framework                 (device)
#     └── ios-arm64_x86_64-simulator/prism_bridge.framework (simulator, fat)
#
# Required tools: Xcode, cmake, ninja, lipo, xcodebuild.
#
# Optional env:
#   PRISM_LLAMA_CPP   path to a PrismML/llama.cpp clone (default: ../prism-llama.cpp)
#   IOS_DEPLOY        minimum iOS version                (default: 14.0)
#
# Usage:
#   ./tool/build_ios_prism.sh
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRISM_LLAMA_CPP="${PRISM_LLAMA_CPP:-$PKG_ROOT/../prism-llama.cpp}"
IOS_DEPLOY="${IOS_DEPLOY:-14.0}"

if [[ ! -d "$PRISM_LLAMA_CPP" ]]; then
    echo "ERROR: PRISM_LLAMA_CPP=$PRISM_LLAMA_CPP does not exist." >&2
    echo "       git clone --branch prism https://github.com/PrismML-Eng/llama.cpp.git \"$PRISM_LLAMA_CPP\"" >&2
    exit 1
fi
# Resolve to an absolute path (add_subdirectory needs it).
PRISM_LLAMA_CPP="$(cd "$PRISM_LLAMA_CPP" && pwd)"

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# build_slice <tag> <sysroot> <arch>
#   tag: device-arm64 | sim-arm64 | sim-x86_64
# Emits: $PKG_ROOT/native/build-ios-prism-<tag>/prism_bridge.framework
build_slice () {
    local tag="$1"; local sysroot="$2"; local arch="$3"

    echo "==================================================="
    echo "Building iOS prism slice: $tag (sdk=$sysroot arch=$arch)"
    echo "==================================================="

    local build_dir="$PKG_ROOT/native/build-ios-prism-$tag"
    rm -rf "$build_dir"
    cmake -B "$build_dir" -G Ninja -S "$PKG_ROOT/native/prism" \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_SYSROOT="$sysroot" \
        -DCMAKE_OSX_ARCHITECTURES="$arch" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOY" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPRISM_LLAMA_CPP="$PRISM_LLAMA_CPP"
    cmake --build "$build_dir" --target prism_bridge -- -j"$JOBS"

    if [[ ! -d "$build_dir/prism_bridge.framework" ]]; then
        echo "ERROR: prism_bridge.framework not produced in $build_dir" >&2
        exit 1
    fi
    echo "  ✓ $build_dir/prism_bridge.framework"
}

build_slice "device-arm64" "iphoneos"        "arm64"
build_slice "sim-arm64"     "iphonesimulator" "arm64"
build_slice "sim-x86_64"    "iphonesimulator" "x86_64"

# ---------------------------------------------------------------------------
# Fuse the two simulator framework binaries into one fat simulator framework.
# Start from the arm64 sim framework bundle, then lipo its binary with x86_64.
# ---------------------------------------------------------------------------
SIM_FAT="$PKG_ROOT/native/build-ios-prism-sim-fat"
rm -rf "$SIM_FAT"; mkdir -p "$SIM_FAT"
cp -R "$PKG_ROOT/native/build-ios-prism-sim-arm64/prism_bridge.framework" \
      "$SIM_FAT/prism_bridge.framework"
lipo -create \
    "$PKG_ROOT/native/build-ios-prism-sim-arm64/prism_bridge.framework/prism_bridge" \
    "$PKG_ROOT/native/build-ios-prism-sim-x86_64/prism_bridge.framework/prism_bridge" \
    -output "$SIM_FAT/prism_bridge.framework/prism_bridge"

# ---------------------------------------------------------------------------
# Assemble the xcframework (device + fat-simulator).
# ---------------------------------------------------------------------------
XCF="$PKG_ROOT/ios/Frameworks/prism_bridge.xcframework"
rm -rf "$XCF"
xcodebuild -create-xcframework \
    -framework "$PKG_ROOT/native/build-ios-prism-device-arm64/prism_bridge.framework" \
    -framework "$SIM_FAT/prism_bridge.framework" \
    -output "$XCF"

echo
echo "✓ Built $XCF"
find "$XCF" -name 'prism_bridge' -path '*framework*' -exec sh -c 'echo "--- $1"; lipo -info "$1"; nm "$1" 2>/dev/null | grep -c " T _bn_"' _ {} \;

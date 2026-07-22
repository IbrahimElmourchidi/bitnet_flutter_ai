#!/usr/bin/env bash
# Build libprism_bridge.so for Linux desktop (and host smoke tests): PrismML's
# llama.cpp fork (Q2_0 ternary kernels) statically linked into one self-
# contained shared library exporting only the bn_* C ABI.
#
# Optional env:
#   PRISM_LLAMA_CPP   path to a PrismML/llama.cpp clone (default: ../prism-llama.cpp)
#                     Clone with:
#                       git clone --branch prism https://github.com/PrismML-Eng/llama.cpp.git
#
# Usage:  ./tool/build_linux_prism.sh
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRISM_LLAMA_CPP="${PRISM_LLAMA_CPP:-$PKG_ROOT/../prism-llama.cpp}"

if [[ ! -d "$PRISM_LLAMA_CPP" ]]; then
    echo "ERROR: PRISM_LLAMA_CPP=$PRISM_LLAMA_CPP does not exist." >&2
    echo "       git clone --branch prism https://github.com/PrismML-Eng/llama.cpp.git \"$PRISM_LLAMA_CPP\"" >&2
    exit 1
fi

JOBS="$(nproc 2>/dev/null || echo 4)"
BUILD_DIR="$PKG_ROOT/native/build-linux-prism"

cmake -B "$BUILD_DIR" -G Ninja -S "$PKG_ROOT/native/prism" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPRISM_LLAMA_CPP="$PRISM_LLAMA_CPP"
cmake --build "$BUILD_DIR" -- -j"$JOBS"

OUT="$BUILD_DIR/libprism_bridge.so"
[[ -f "$OUT" ]] || { echo "ERROR: expected $OUT" >&2; exit 1; }

# Stage next to libbitnet_bridge.so for Linux desktop consumers.
mkdir -p "$PKG_ROOT/linux"
cp -f "$OUT" "$PKG_ROOT/linux/libprism_bridge.so" 2>/dev/null || true

echo
echo "Built $OUT"
echo "Exported dynamic symbols (should be bn_* only):"
nm -D --defined-only "$OUT" 2>/dev/null | grep -v ' _' | awk '{print $NF}' | sort | head -20

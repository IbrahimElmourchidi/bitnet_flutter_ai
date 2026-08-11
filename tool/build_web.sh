#!/usr/bin/env bash
# Build the BitNet WASM module for the web engine and copy the artifacts into
# example/web/ (bitnet_bridge.js, bitnet_bridge.wasm, bitnet_glue.js).
#
# Produces a single-threaded, exception-enabled WebAssembly build of
# bitnet.cpp (ggml + llama.cpp) plus the embind bridge. The i2_s matmul runs on
# the portable scalar kernels from native/patches/0001-i2s-portable-kernels.patch
# (WASM has neither AVX2 nor NEON), so the patch MUST be applied — this script
# applies it idempotently, same as tool/build_android.sh.
#
# Required:
#   emcc / em++ on PATH  (run `source <emsdk>/emsdk_env.sh` first).
#
# Optional env:
#   BITNET_CPP   path to a microsoft/BitNet clone (default: ../bitnet.cpp)
#
# Usage:
#   source ~/emsdk/emsdk_env.sh
#   ./tool/build_web.sh
set -euo pipefail

if ! command -v em++ >/dev/null 2>&1; then
    echo "ERROR: em++ not found. Install Emscripten and 'source emsdk_env.sh'." >&2
    exit 1
fi

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BITNET_CPP="${BITNET_CPP:-$PKG_ROOT/../bitnet.cpp}"

if [[ ! -d "$BITNET_CPP" ]]; then
    echo "ERROR: BITNET_CPP=$BITNET_CPP does not exist." >&2
    echo "       Clone microsoft/BitNet with --recursive and set BITNET_CPP=<path>." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Apply the portable-i2s-kernels patch (idempotent). See tool/build_android.sh
# for the full rationale; on WASM the scalar fallback is the ONLY code path,
# and the patch also carries the gemm tail-block dispatch fix required to avoid
# an out-of-bounds trap on bounds-checked WebAssembly memory.
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
    exit 1
fi

# 0003: removes the I2_S gemm prefill scratch VLA, which was sized
# (rows-per-thread * n_ubatch) floats — hundreds of KB — and is the reason
# STACK_SIZE below had to be raised to 32 MB. The kernel now accumulates
# straight into dst, so prefill stack use no longer scales with the prompt.
# (Native targets need this to avoid overrunning ggml's 512 KB–1 MB worker
# thread stacks outright; on wasm it just removes the stack pressure.)
PATCH3="$PKG_ROOT/native/patches/0003-i2s-gemm-no-stack-vla.patch"
LLAMA_SUB="$BITNET_CPP/3rdparty/llama.cpp"
if [[ -f "$PATCH3" ]]; then
    if git -C "$LLAMA_SUB" apply --check "$PATCH3" 2>/dev/null; then
        git -C "$LLAMA_SUB" apply "$PATCH3"
        echo "Applied 0003 i2s gemm no-stack-VLA patch to $LLAMA_SUB"
    elif git -C "$LLAMA_SUB" apply --check --reverse "$PATCH3" 2>/dev/null; then
        echo "0003 i2s gemm no-stack-VLA patch already applied to $LLAMA_SUB"
    else
        echo "ERROR: 0003 patch does not apply cleanly to $LLAMA_SUB." >&2
        exit 1
    fi
fi

JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
BUILD_DIR="$BITNET_CPP/build-web"

# ---------------------------------------------------------------------------
# 1. Configure + build the static libraries (libllama.a, libggml.a) for wasm32.
#    - BITNET_ARM_TL1 / X86_TL2 OFF: neither LUT kernel applies to WASM, so the
#      i2_s path goes through the scalar ggml_vec_dot_i2_i8_s kernels.
#    - GGML_NATIVE=OFF: no -march=native (meaningless for wasm).
#    - GGML_OPENMP=OFF + single-threaded at runtime: the module is built
#      without pthreads (see the STACK/threads notes below).
#    - -fexceptions: llama.cpp converts loader errors into a caught null return;
#      without it a bad/oversized GGUF aborts the whole module.
# ---------------------------------------------------------------------------
emcmake cmake -B "$BUILD_DIR" -G Ninja -S "$BITNET_CPP" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBITNET_ARM_TL1=OFF -DBITNET_X86_TL2=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_FLAGS="-fexceptions" -DCMAKE_CXX_FLAGS="-fexceptions"
cmake --build "$BUILD_DIR" --target llama -- -j"$JOBS"

LLAMA_A="$BUILD_DIR/3rdparty/llama.cpp/src/libllama.a"
GGML_A="$BUILD_DIR/3rdparty/llama.cpp/ggml/src/libggml.a"
for a in "$LLAMA_A" "$GGML_A"; do
    [[ -f "$a" ]] || { echo "ERROR: expected static lib not found: $a" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# 2. Link the embind bridge into the final module.
#    - STACK_SIZE=32MB: the i2_s gemm prefill path uses large stack buffers that
#      overflow Emscripten's tiny default (64 KB) -> memory-access-out-of-bounds.
#    - INITIAL_MEMORY=1.4GB: large enough to hold the ~1.1GB GGUF as it streams
#      into the in-memory FS without repeated memory.grow() calls (each grow
#      copies the whole heap — dozens of multi-GB copies otherwise made bn_init
#      appear to hang for a long time). Heap still grows to MAXIMUM_MEMORY for
#      the model tensors + KV cache. Must exceed STACK_SIZE.
#    - MAXIMUM_MEMORY=4GB: the 1.1 GB model + KV/compute buffers need headroom
#      (wasm32 hard ceiling is 4 GB — desktop browsers only; see README).
#    - NODEFS is linked so the module can also be exercised headlessly in Node.
# ---------------------------------------------------------------------------
OUT_DIR="$PKG_ROOT/native/build-web"
mkdir -p "$OUT_DIR"
em++ -std=c++17 --bind -O2 -fexceptions \
    -I "$PKG_ROOT/native" \
    -I "$BITNET_CPP/3rdparty/llama.cpp/include" \
    -I "$BITNET_CPP/3rdparty/llama.cpp/ggml/include" \
    "$PKG_ROOT/native/bitnet_bridge_wasm.cc" \
    "$PKG_ROOT/native/bitnet_bridge.cc" \
    "$LLAMA_A" "$GGML_A" \
    -lnodefs.js \
    -s MODULARIZE=1 -s EXPORT_NAME=createBitNetModule \
    -s ENVIRONMENT=web,worker,node \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=1476395008 \
    -s MAXIMUM_MEMORY=4294967296 \
    -s STACK_SIZE=33554432 \
    -s FORCE_FILESYSTEM=1 \
    -s EXPORTED_RUNTIME_METHODS=FS,ccall,cwrap \
    -o "$OUT_DIR/bitnet_bridge.js"

# ---------------------------------------------------------------------------
# 3. Stage the artifacts into example/web/. The Emscripten output plus the two
#    hand-written scripts: bitnet_worker.js (owns the module in a Web Worker)
#    and bitnet_glue.js (main-thread async proxy). index.html loads only the
#    glue; the worker importScripts the bridge itself.
# ---------------------------------------------------------------------------
EXAMPLE_WEB="$PKG_ROOT/example/web"
cp -f "$OUT_DIR/bitnet_bridge.js"     "$EXAMPLE_WEB/bitnet_bridge.js"
cp -f "$OUT_DIR/bitnet_bridge.wasm"   "$EXAMPLE_WEB/bitnet_bridge.wasm"
cp -f "$PKG_ROOT/web/bitnet_worker.js" "$EXAMPLE_WEB/bitnet_worker.js"
cp -f "$PKG_ROOT/web/bitnet_glue.js"   "$EXAMPLE_WEB/bitnet_glue.js"

echo
echo "WASM build complete. Artifacts staged in example/web/:"
ls -la "$EXAMPLE_WEB"/bitnet_bridge.js "$EXAMPLE_WEB"/bitnet_bridge.wasm \
       "$EXAMPLE_WEB"/bitnet_worker.js "$EXAMPLE_WEB"/bitnet_glue.js
echo
echo "NOTE: the module runs in a Web Worker (off the UI thread), single-threaded"
echo "      scalar (~1-2 tok/s for 2B). See README 'Web (WASM)' for threading +"
echo "      wasm_simd128 speed follow-ups."

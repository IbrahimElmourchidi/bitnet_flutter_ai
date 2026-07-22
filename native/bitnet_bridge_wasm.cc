/**
 * bitnet_bridge_wasm.cc
 *
 * Emscripten/embind adapter that exposes the C bridge (bitnet_bridge.h) to
 * JavaScript with the exact shape lib/src/engine/engine_web.dart expects:
 *
 *   BitNetWasm.bn_init(modelPath, nCtx, nThreads) -> handle (number) | null
 *   BitNetWasm.bn_prompt(handle, prompt, addBos)  -> int rc (0 == ok)
 *   BitNetWasm.bn_next_token(handle)              -> { done: bool, token: string|null }
 *   BitNetWasm.bn_free(handle)                    -> void
 *
 * The native C ABI uses out-parameters (out_done / out_err) and caller-owned
 * malloc'd strings; this layer marshals all of that into JS-native values so
 * the Dart side never touches the WASM heap directly.
 *
 * The opaque BnContext* is carried across the JS boundary as a `double`
 * holding the pointer value. WASM32 pointers are 32-bit, so they fit exactly
 * in a double's 53-bit mantissa — no precision loss. A null context is
 * reported to JS as `null` (not 0) so engine_web.dart's `ctx == null` check
 * works correctly.
 *
 * Build: compiled by tool/build_web.sh with `emcc --bind` and linked against
 * the WASM build of libllama / libggml (bitnet.cpp).
 */

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <cstdio>
#include <string>

#include "bitnet_bridge.h"
#include "llama.h"

using emscripten::val;

namespace {

// Decode a JS-side handle (double) back into the opaque context pointer.
BnContext* handle_to_ctx(double handle) {
    return reinterpret_cast<BnContext*>(static_cast<uintptr_t>(handle));
}

// llama.cpp is extremely chatty at INFO/WARN (the full GGUF metadata dump, a
// per-token "control token ... not marked as EOG" line for every special
// token, etc). On the web each line crosses the WASM->JS boundary as a
// console.error WITH a stack trace, which floods devtools and can freeze the
// tab on its own. Drop everything below ERROR; real errors still surface.
void wasm_quiet_log(ggml_log_level level, const char* text, void* /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_ERROR && text) {
        std::fprintf(stderr, "%s", text);
    }
}

bool g_log_installed = false;

void install_quiet_log_once() {
    if (!g_log_installed) {
        llama_log_set(wasm_quiet_log, nullptr);
        g_log_installed = true;
    }
}

// Log a bridge error to the JS console (stderr is wired to console.error by
// the Emscripten runtime) and release the malloc'd error string.
void log_and_free_err(const char* fn, char* err) {
    if (err) {
        std::fprintf(stderr, "[bitnet_wasm] %s: %s\n", fn, err);
        bn_free_string(err);
    }
}

}  // namespace

// BitNetWasm.bn_init(modelPath, nCtx, nThreads) -> handle | null
val wasm_bn_init(const std::string& model_path, int n_ctx, int n_threads) {
    install_quiet_log_once();
    char* err = nullptr;
    BnContext* ctx = bn_init(model_path.c_str(),
                             static_cast<int32_t>(n_ctx),
                             static_cast<int32_t>(n_threads),
                             &err);
    if (!ctx) {
        log_and_free_err("bn_init", err);
        return val::null();
    }
    return val(static_cast<double>(reinterpret_cast<uintptr_t>(ctx)));
}

// BitNetWasm.bn_prompt(handle, prompt, addBos) -> rc (0 == ok, -1 == error)
int wasm_bn_prompt(double handle, const std::string& prompt, int add_bos) {
    char* err = nullptr;
    const int32_t rc = bn_prompt(handle_to_ctx(handle),
                                 prompt.c_str(),
                                 static_cast<int32_t>(add_bos),
                                 &err);
    if (rc != 0) {
        log_and_free_err("bn_prompt", err);
    }
    return rc;
}

// BitNetWasm.bn_next_token(handle) -> { done: bool, token: string|null }
val wasm_bn_next_token(double handle) {
    int32_t done = 0;
    char* err = nullptr;
    char* piece = bn_next_token(handle_to_ctx(handle), &done, &err);

    val result = val::object();
    if (piece) {
        // A real token piece (may legitimately be an empty string).
        result.set("done", false);
        result.set("token", std::string(piece));
        bn_free_string(piece);
        return result;
    }

    // NULL piece: either clean end-of-generation (done==1) or an error
    // (done==0). engine_web.dart treats `done: true` as a normal stop, so we
    // surface errors as a stop too, after logging them to the console.
    if (!done) {
        log_and_free_err("bn_next_token", err);
    }
    result.set("done", true);
    result.set("token", val::null());
    return result;
}

// BitNetWasm.bn_free(handle) -> void
void wasm_bn_free(double handle) {
    bn_free(handle_to_ctx(handle));
}

EMSCRIPTEN_BINDINGS(bitnet_bridge) {
    emscripten::function("bn_init", &wasm_bn_init);
    emscripten::function("bn_prompt", &wasm_bn_prompt);
    emscripten::function("bn_next_token", &wasm_bn_next_token);
    emscripten::function("bn_free", &wasm_bn_free);
}

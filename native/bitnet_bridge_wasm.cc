/**
 * bitnet_bridge_wasm.cc
 *
 * Emscripten/embind adapter that exposes the C bridge (bitnet_bridge.h) to
 * JavaScript with the exact shape lib/src/engine/engine_web.dart expects:
 *
 *   BitNetWasm.bn_init(modelPath, nCtx, nThreads) -> handle (number) | null
 *   BitNetWasm.bn_set_sampling(handle, temp, topK, topP, repeat, lastN, seed, grammar)
 *                                                 -> { ok: bool, error: string|null }
 *   BitNetWasm.bn_prompt(handle, prompt, addBos)  -> { rc: int, error: string|null }
 *   BitNetWasm.bn_next_token(handle)
 *       -> { done: bool, bytes: Uint8Array|null, logprob: number, error: string|null }
 *   BitNetWasm.bn_token_count(handle, text, addBos) -> int (-1 on error)
 *   BitNetWasm.bn_free(handle)                    -> void
 *
 * Token pieces are returned as raw BYTES, not strings: a piece is often only a
 * fragment of a multi-byte UTF-8 character (Arabic, emoji, curly quotes), and
 * decoding it on its own yields U+FFFD. engine_web.dart reassembles them with
 * the same Utf8StreamDecoder the native engine uses.
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
#include <cstring>
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

// Takes ownership of a bridge error string.
val take_err(char* err) {
    if (!err) return val::null();
    val v{std::string(err)};
    bn_free_string(err);
    return v;
}

// BitNetWasm.bn_set_sampling(...) -> { ok, error }
val wasm_bn_set_sampling(double handle, double temperature, int top_k, double top_p,
                         double repeat_penalty, int repeat_last_n, double seed,
                         const std::string& grammar) {
    BnSamplingParams p;
    p.temperature    = static_cast<float>(temperature);
    p.top_k          = top_k;
    p.top_p          = static_cast<float>(top_p);
    p.repeat_penalty = static_cast<float>(repeat_penalty);
    p.repeat_last_n  = repeat_last_n;
    p.seed           = static_cast<uint32_t>(seed);
    char* err = nullptr;
    const int32_t rc = bn_set_sampling(handle_to_ctx(handle), &p,
                                       grammar.empty() ? nullptr : grammar.c_str(), &err);
    val result = val::object();
    result.set("ok", rc == 0);
    result.set("error", take_err(err));
    return result;
}

// BitNetWasm.bn_prompt(handle, prompt, addBos) -> { rc, error }
val wasm_bn_prompt(double handle, const std::string& prompt, int add_bos) {
    char* err = nullptr;
    const int32_t rc = bn_prompt(handle_to_ctx(handle), prompt.c_str(),
                                 static_cast<int32_t>(add_bos), &err);
    val result = val::object();
    result.set("rc", rc);
    result.set("error", take_err(err));
    return result;
}

// BitNetWasm.bn_next_token(handle) -> { done, bytes, logprob, error }
val wasm_bn_next_token(double handle) {
    int32_t done = 0;
    float logprob = 0.0f;
    char* err = nullptr;
    char* piece = bn_next_token_ex(handle_to_ctx(handle), &done, &logprob, &err);

    val result = val::object();
    result.set("logprob", logprob);
    if (piece) {
        const size_t n = std::strlen(piece);
        // Copy out of the WASM heap into a standalone Uint8Array.
        val view(emscripten::typed_memory_view(n, reinterpret_cast<const uint8_t*>(piece)));
        result.set("done", false);
        result.set("bytes", val::global("Uint8Array").new_(view));
        result.set("error", val::null());
        bn_free_string(piece);
        return result;
    }
    result.set("done", done != 0);
    result.set("bytes", val::null());
    result.set("error", take_err(err));
    return result;
}

// BitNetWasm.bn_token_count(handle, text, addBos) -> int
int wasm_bn_token_count(double handle, const std::string& text, int add_bos) {
    return bn_token_count(handle_to_ctx(handle), text.c_str(), static_cast<int32_t>(add_bos));
}

// BitNetWasm.bn_free(handle) -> void
void wasm_bn_free(double handle) {
    bn_free(handle_to_ctx(handle));
}

EMSCRIPTEN_BINDINGS(bitnet_bridge) {
    emscripten::function("bn_init", &wasm_bn_init);
    emscripten::function("bn_set_sampling", &wasm_bn_set_sampling);
    emscripten::function("bn_prompt", &wasm_bn_prompt);
    emscripten::function("bn_next_token", &wasm_bn_next_token);
    emscripten::function("bn_token_count", &wasm_bn_token_count);
    emscripten::function("bn_free", &wasm_bn_free);
}

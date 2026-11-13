/**
 * bitnet_bridge.cc
 *
 * bn_* C ABI (see bitnet_bridge.h) over Microsoft bitnet.cpp's pinned
 * llama.cpp (the OLD, late-2024 API: model-taking tokenize/piece/eog,
 * explicit-position llama_batch_get_one, llama_kv_cache_*). Everything that is
 * not API-generation specific lives in bn_core.inc, shared with
 * prism_bridge.cc.
 *
 * Build requirements:
 *   - llama.h on the include path (bitnet.cpp stage/include)
 *   - link against bitnet.cpp's libllama (I2_S / TL1 / TL2 kernels)
 *
 * Thread safety: all functions except bn_free_string and writes to the abort
 * flag are non-reentrant on the same BnContext. The Dart inference isolate
 * ensures single-threaded access.
 */

#include "bitnet_bridge.h"
#include "llama.h"

#include <cstring>
#include <string>
#include <thread>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

// llama.cpp model-load progress (0..1), forwarded to the Web Worker's main
// thread as a postMessage so the UI can show "loading model" progress. Guarded:
// postMessage exists in a Web Worker but not in plain Node (headless tests).
static bool bn_wasm_load_progress(float progress, void* /*user_data*/) {
    EM_ASM({
        if (typeof postMessage === 'function') {
            postMessage({ type: 'load_progress', frac: $0 });
        }
    }, (double)progress);
    return true;
}
#endif

// Extra end-of-turn token honoured beyond the vocab's registered EOG set.
// Some Llama-3 derivatives end assistant turns with <|eot_id|> but do not mark
// it EOG, so the model would run on and fabricate the next turn. Resolved once
// at init; LLAMA_TOKEN_NULL when the model has no such token (e.g. Falcon).
// (The shipped BitNet 2B-4T GGUF already marks it EOG — this is a safety net.)
struct BnBackendExtra {
    llama_token extra_eot_id = LLAMA_TOKEN_NULL;
};

namespace bnapi {
constexpr const char* kTag = "bitnet_bridge";
// bitnet.cpp's pinned ggml deadlocks when the abort callback fires mid-graph
// (see bn_core.inc "Cancellation"); cancel between prefill chunks instead.
constexpr bool kAbortInCompute = false;

inline int n_vocab(const BnContext* bn);
inline int tokenize(const BnContext* bn, const char* text, int len,
                    llama_token* out, int max, bool add_special);
inline int piece(const BnContext* bn, llama_token t, char* buf, int len);
inline bool is_eog(const BnContext* bn, llama_token t);
inline int decode(BnContext* bn, llama_token* toks, int n, int pos0);
inline bool kv_seq_rm(BnContext* bn, int p0);
inline void kv_clear(BnContext* bn);
inline llama_sampler* penalties(const BnContext* bn, int last_n, float repeat);
inline llama_sampler* grammar(const BnContext* bn, const char* gbnf);
inline void free_model(llama_model* m) { llama_free_model(m); }
}  // namespace bnapi

#include "bn_core.inc"

namespace bnapi {
inline int n_vocab(const BnContext* bn) { return llama_n_vocab(bn->model); }

inline int tokenize(const BnContext* bn, const char* text, int len,
                    llama_token* out, int max, bool add_special) {
    // If the caller already wrote an explicit BOS (a Llama-3 template starting
    // with "<|begin_of_text|>"), don't add a second one — double BOS confuses
    // the model.
    const bool explicit_bos = std::strncmp(text, "<|begin_of_text|>", 17) == 0;
    return llama_tokenize(bn->model, text, len, out, max,
                          add_special && !explicit_bos, /*parse_special=*/true);
}

inline int piece(const BnContext* bn, llama_token t, char* buf, int len) {
    // special=true: render special tokens as text so the Dart stop-marker
    // detector can see them.
    return llama_token_to_piece(bn->model, t, buf, len, /*lstrip=*/0, /*special=*/true);
}

inline bool is_eog(const BnContext* bn, llama_token t) {
    return llama_token_is_eog(bn->model, t) ||
           (bn->be.extra_eot_id != LLAMA_TOKEN_NULL && t == bn->be.extra_eot_id);
}

inline int decode(BnContext* bn, llama_token* toks, int n, int pos0) {
    // This llama.cpp's llama_batch_get_one takes explicit (pos_0, seq_id).
    return llama_decode(bn->ctx, llama_batch_get_one(toks, n, pos0, /*seq_id=*/0));
}

inline bool kv_seq_rm(BnContext* bn, int p0) {
    return llama_kv_cache_seq_rm(bn->ctx, /*seq_id=*/0, p0, /*p1=*/-1);
}

inline void kv_clear(BnContext* bn) { llama_kv_cache_clear(bn->ctx); }

inline llama_sampler* penalties(const BnContext* bn, int last_n, float repeat) {
    return llama_sampler_init_penalties(
        llama_n_vocab(bn->model), llama_token_eos(bn->model), llama_token_nl(bn->model),
        last_n, repeat, /*freq=*/0.0f, /*present=*/0.0f,
        /*penalize_nl=*/false, /*ignore_eos=*/false);
}

inline llama_sampler* grammar(const BnContext* bn, const char* gbnf) {
    return llama_sampler_init_grammar(bn->model, gbnf, "root");
}
}  // namespace bnapi

// Resolves the model's end-of-turn token: the vocab's declared EOT first, else
// the literal "<|eot_id|>" if it maps to exactly ONE (special) token.
static llama_token resolve_extra_eot(llama_model* model) {
    const llama_token declared = llama_token_eot(model);
    if (declared != LLAMA_TOKEN_NULL) return declared;

    static const char kEot[] = "<|eot_id|>";
    llama_token ids[8];
    const int32_t n = llama_tokenize(model, kEot, static_cast<int32_t>(sizeof(kEot) - 1),
                                     ids, 8, /*add_special=*/false, /*parse_special=*/true);
    return n == 1 ? ids[0] : LLAMA_TOKEN_NULL;
}

BnContext* bn_init(const char* model_path, int32_t n_ctx, int32_t n_threads, char** out_err) {
    if (!model_path) {
        bn_set_err(out_err, "model_path must not be NULL");
        return nullptr;
    }

    const int threads = (n_threads > 0)
        ? static_cast<int>(n_threads)
        : static_cast<int>(std::thread::hardware_concurrency());

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU-only; BitNet I2_S kernels are CPU-optimised
#ifdef __EMSCRIPTEN__
    mparams.progress_callback = bn_wasm_load_progress;
    mparams.progress_callback_user_data = nullptr;
#endif

    llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        bn_set_err(out_err, std::string("Failed to load model from: ") + model_path);
        return nullptr;
    }

    const int ctx_size = (n_ctx > 0) ? static_cast<int>(n_ctx) : llama_n_ctx_train(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = static_cast<uint32_t>(ctx_size);
    cparams.n_threads       = static_cast<uint32_t>(threads);
    cparams.n_threads_batch = static_cast<uint32_t>(threads);
    // A full-context prompt fits in one logical batch (bn_core also chunks).
    cparams.n_batch    = static_cast<uint32_t>(ctx_size);
    // The I2_S CPU path uses standard (non-fused) attention.
    cparams.flash_attn = false;
    // Micro-batch cap: bounds the ggml compute buffer and keeps each prefill
    // step short (see native/patches/0003-i2s-gemm-no-stack-vla.patch for the
    // stack-VLA history behind this knob).
    if (cparams.n_ubatch > 128) cparams.n_ubatch = 128;

    llama_context* lctx = llama_new_context_with_model(model, cparams);
    if (!lctx) {
        llama_free_model(model);
        bn_set_err(out_err, "Failed to create llama context");
        return nullptr;
    }

    BnContext* bn = new BnContext();
    bn->model = model;
    bn->ctx   = lctx;
    bn->be.extra_eot_id = resolve_extra_eot(model);
    return bn_core_attach(bn, ctx_size);
}

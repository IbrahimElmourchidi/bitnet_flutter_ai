/**
 * prism_bridge.cc
 *
 * Second native backend: the bn_* C ABI (bitnet_bridge.h) over PrismML's
 * llama.cpp fork (github.com/PrismML-Eng/llama.cpp, `prism` branch) for its
 * Q2_0 (g128) ternary models (PrismML "Ternary Bonsai", Qwen3 architecture).
 *
 * This targets the fork's MODERN API (vocab-taking tokenize/piece/eog,
 * auto-positioned 2-arg llama_batch_get_one, llama_init_from_model,
 * llama_memory_*). Everything else is shared with bitnet_bridge.cc via
 * bn_core.inc. The two stacks have incompatible ggml ABIs and MUST NOT share
 * libraries — this is linked as a single self-contained library exporting only
 * bn_* (see tool/build_*_prism.sh).
 *
 * Thread safety: non-reentrant per BnContext (the Dart inference isolate owns
 * it exclusively), except bn_free_string and writes to the abort flag.
 */

#include "bitnet_bridge.h"
#include "llama.h"

#include <string>
#include <thread>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

static bool prism_wasm_load_progress(float progress, void* /*user_data*/) {
    EM_ASM({
        if (typeof postMessage === 'function') {
            postMessage({ type: 'load_progress', frac: $0 });
        }
    }, (double)progress);
    return true;
}
#endif

struct BnBackendExtra {
    const llama_vocab* vocab = nullptr;
};

namespace bnapi {
constexpr const char* kTag = "prism_bridge";
constexpr bool kAbortInCompute = true;

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
inline void free_model(llama_model* m) { llama_model_free(m); }
}  // namespace bnapi

#include "bn_core.inc"

namespace bnapi {
inline int n_vocab(const BnContext* bn) { return llama_vocab_n_tokens(bn->be.vocab); }

inline int tokenize(const BnContext* bn, const char* text, int len,
                    llama_token* out, int max, bool add_special) {
    // No explicit-BOS sniff (a BitNet-template quirk): the tokenizer adds
    // special tokens per the model's own config.
    return llama_tokenize(bn->be.vocab, text, len, out, max, add_special,
                          /*parse_special=*/true);
}

inline int piece(const BnContext* bn, llama_token t, char* buf, int len) {
    return llama_token_to_piece(bn->be.vocab, t, buf, len, /*lstrip=*/0, /*special=*/true);
}

inline bool is_eog(const BnContext* bn, llama_token t) {
    return llama_vocab_is_eog(bn->be.vocab, t);
}

inline int decode(BnContext* bn, llama_token* toks, int n, int /*pos0*/) {
    // Modern batch: positions continue from the KV state (kv_seq_rm keeps it
    // in step with bn->kv_tokens).
    return llama_decode(bn->ctx, llama_batch_get_one(toks, n));
}

inline bool kv_seq_rm(BnContext* bn, int p0) {
    return llama_memory_seq_rm(llama_get_memory(bn->ctx), /*seq_id=*/0, p0, /*p1=*/-1);
}

inline void kv_clear(BnContext* bn) { llama_memory_clear(llama_get_memory(bn->ctx), true); }

inline llama_sampler* penalties(const BnContext* /*bn*/, int last_n, float repeat) {
    return llama_sampler_init_penalties(last_n, repeat, /*freq=*/0.0f, /*present=*/0.0f);
}

inline llama_sampler* grammar(const BnContext* bn, const char* gbnf) {
    return llama_sampler_init_grammar(bn->be.vocab, gbnf, "root");
}
}  // namespace bnapi

BnContext* bn_init(const char* model_path, int32_t n_ctx, int32_t n_threads, char** out_err) {
    if (!model_path) {
        bn_set_err(out_err, "model_path must not be NULL");
        return nullptr;
    }

    // Idempotent in llama.cpp; safe to call on every init.
    llama_backend_init();

    const int threads = (n_threads > 0)
        ? static_cast<int>(n_threads)
        : static_cast<int>(std::thread::hardware_concurrency());

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU-only
#ifdef __EMSCRIPTEN__
    mparams.progress_callback = prism_wasm_load_progress;
    mparams.progress_callback_user_data = nullptr;
#endif

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        bn_set_err(out_err, std::string("Failed to load model from: ") + model_path);
        return nullptr;
    }

    const int ctx_size = (n_ctx > 0) ? static_cast<int>(n_ctx) : llama_model_n_ctx_train(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = static_cast<uint32_t>(ctx_size);
    cparams.n_batch         = static_cast<uint32_t>(ctx_size);
    cparams.n_threads       = threads;
    cparams.n_threads_batch = threads;

    llama_context* lctx = llama_init_from_model(model, cparams);
    if (!lctx) {
        llama_model_free(model);
        bn_set_err(out_err, "Failed to create llama context");
        return nullptr;
    }

    BnContext* bn = new BnContext();
    bn->model    = model;
    bn->ctx      = lctx;
    bn->be.vocab = llama_model_get_vocab(model);
    return bn_core_attach(bn, ctx_size);
}

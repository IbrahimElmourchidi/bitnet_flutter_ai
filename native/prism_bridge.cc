/**
 * prism_bridge.cc
 *
 * Second native backend: a thin C wrapper over PrismML's llama.cpp fork
 * (github.com/PrismML-Eng/llama.cpp, `prism` branch) for its Q2_0 (g128)
 * ternary-quantized models (PrismML "Ternary Bonsai", Qwen3 architecture).
 *
 * It exposes the SAME `bn_*` C ABI as bitnet_bridge.h, so the existing Dart FFI
 * bindings (lib/src/ffi/bindings.dart + native_library.dart) drive it with no
 * changes — only the shared-library name differs (libprism_bridge.so).
 *
 * Unlike bitnet_bridge.cc (which targets bitnet.cpp's OLD pinned llama.cpp API)
 * this targets the fork's MODERN API: vocab-taking tokenize/piece/eog, 2-arg
 * llama_batch_get_one (auto-positioned from KV state), llama_init_from_model,
 * llama_memory_clear. The two stacks have incompatible ggml ABIs and MUST NOT
 * share libraries — this is linked as a single self-contained .so exporting
 * only bn_* (see tool/build_*_prism.sh).
 *
 * Thread safety: non-reentrant per BnContext (the Dart inference isolate owns
 * it exclusively), except bn_free_string.
 */

#include "bitnet_bridge.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

struct BnContext {
    llama_model*        model   = nullptr;
    llama_context*      ctx     = nullptr;
    const llama_vocab*  vocab   = nullptr;
    llama_sampler*      sampler = nullptr;
    int                 n_past  = 0;
    int                 n_ctx   = 0;
    char                meta_buf[1024] = {};
};

static char* make_err(const std::string& msg) {
    char* buf = static_cast<char*>(std::malloc(msg.size() + 1));
    if (buf) std::memcpy(buf, msg.c_str(), msg.size() + 1);
    return buf;
}

static void set_err(char** out_err, const std::string& msg) {
    if (out_err) *out_err = make_err(msg);
}

// Web-only: forward model-load progress to the Web Worker's main thread (guard
// postMessage — absent in plain Node). Mirrors bitnet_bridge.cc.
#ifdef __EMSCRIPTEN__
static bool prism_wasm_load_progress(float progress, void* /*user_data*/) {
    EM_ASM({
        if (typeof postMessage === 'function') {
            postMessage({ type: 'load_progress', frac: $0 });
        }
    }, (double)progress);
    return true;
}
#endif

BnContext* bn_init(
    const char* model_path,
    int32_t     n_ctx,
    int32_t     n_threads,
    char**      out_err
) {
    if (!model_path) {
        set_err(out_err, "model_path must not be NULL");
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
        set_err(out_err, std::string("Failed to load model from: ") + model_path);
        return nullptr;
    }

    const int ctx_size = (n_ctx > 0)
        ? static_cast<int>(n_ctx)
        : llama_model_n_ctx_train(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = static_cast<uint32_t>(ctx_size);
    cparams.n_threads       = threads;
    cparams.n_threads_batch = threads;

    llama_context* lctx = llama_init_from_model(model, cparams);
    if (!lctx) {
        llama_model_free(model);
        set_err(out_err, "Failed to create llama context");
        return nullptr;
    }

    llama_sampler* sampler =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    BnContext* bn = new BnContext();
    bn->model   = model;
    bn->ctx     = lctx;
    bn->vocab   = llama_model_get_vocab(model);
    bn->sampler = sampler;
    bn->n_past  = 0;
    bn->n_ctx   = ctx_size;
    return bn;
}

void bn_free(BnContext* ctx) {
    if (!ctx) return;
    if (ctx->sampler) llama_sampler_free(ctx->sampler);
    if (ctx->ctx)     llama_free(ctx->ctx);
    if (ctx->model)   llama_model_free(ctx->model);
    delete ctx;
}

void bn_free_string(char* str) {
    std::free(str);
}

int32_t bn_prompt(
    BnContext*  ctx,
    const char* prompt,
    int32_t     add_bos,
    char**      out_err
) {
    if (!ctx || !prompt) {
        set_err(out_err, "ctx and prompt must not be NULL");
        return -1;
    }

    const std::string p(prompt);
    const int max_tokens = ctx->n_ctx;
    std::vector<llama_token> tokens(max_tokens);

    // No explicit-BOS text sniff (that is a BitNet-template quirk). Let the
    // tokenizer add special tokens per the model's own config via add_special.
    const int n_tokens = llama_tokenize(
        ctx->vocab,
        prompt,
        static_cast<int32_t>(p.size()),
        tokens.data(),
        max_tokens,
        add_bos != 0,  // add_special
        true           // parse_special: render <|...|> markers as special ids
    );

    if (n_tokens < 0) {
        set_err(out_err, "Tokenisation failed: prompt may be too long");
        return -1;
    }
    tokens.resize(static_cast<size_t>(n_tokens));

    std::fprintf(stderr,
        "[prism_bridge] bn_prompt: chars=%zu tokens=%d add_special=%d\n",
        p.size(), n_tokens, add_bos != 0 ? 1 : 0);

    // Fresh conversation turn: clear the KV cache so positions restart at 0.
    llama_memory_clear(llama_get_memory(ctx->ctx), true);
    ctx->n_past = 0;

    // Modern batch: pos=NULL, decode auto-positions from KV state.
    llama_batch batch = llama_batch_get_one(tokens.data(),
                                            static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx->ctx, batch) != 0) {
        set_err(out_err, "llama_decode failed during prompt prefill");
        return -1;
    }

    // Degenerate-logits guard: catch a broken kernel build (flat/non-finite
    // logits) loudly instead of streaming garbage.
    const float* logits = llama_get_logits_ith(ctx->ctx, -1);
    if (logits) {
        const int n_vocab = llama_vocab_n_tokens(ctx->vocab);
        bool flat = true;
        bool finite = true;
        for (int i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(logits[i])) { finite = false; break; }
            if (logits[i] != logits[0]) { flat = false; }
        }
        if (!finite || flat) {
            set_err(out_err,
                "Prism produced degenerate logits (flat or non-finite) after "
                "prompt prefill. The linked libggml was likely built without "
                "functional Q2_0 kernels for this CPU/ABI.");
            return -1;
        }
    }

    ctx->n_past = static_cast<int>(tokens.size());
    return 0;
}

char* bn_next_token(
    BnContext* ctx,
    int32_t*   out_done,
    char**     out_err
) {
    if (out_done) *out_done = 0;

    if (!ctx) {
        set_err(out_err, "ctx must not be NULL");
        return nullptr;
    }

    if (ctx->n_past >= ctx->n_ctx) {
        if (out_done) *out_done = 1;
        return nullptr;
    }

    const llama_token token_id = llama_sampler_sample(ctx->sampler, ctx->ctx, -1);

    if (llama_vocab_is_eog(ctx->vocab, token_id)) {
        if (out_done) *out_done = 1;
        return nullptr;
    }

    char piece_buf[256] = {};
    const int piece_len = llama_token_to_piece(
        ctx->vocab,
        token_id,
        piece_buf,
        static_cast<int32_t>(sizeof(piece_buf) - 1),
        0,      // lstrip
        true    // special: render special tokens as text (for stop-marker detn)
    );
    if (piece_len < 0) {
        set_err(out_err, "llama_token_to_piece failed");
        return nullptr;
    }
    piece_buf[piece_len] = '\0';

    llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(&token_id), 1);
    if (llama_decode(ctx->ctx, batch) != 0) {
        set_err(out_err, "llama_decode failed during generation step");
        return nullptr;
    }
    ctx->n_past++;

    char* result = static_cast<char*>(std::malloc(static_cast<size_t>(piece_len) + 1));
    if (!result) {
        set_err(out_err, "malloc failed for token piece");
        return nullptr;
    }
    std::memcpy(result, piece_buf, static_cast<size_t>(piece_len) + 1);
    return result;
}

int32_t bn_context_size(const BnContext* ctx) {
    if (!ctx) return -1;
    return static_cast<int32_t>(ctx->n_ctx);
}

const char* bn_meta(const BnContext* ctx, const char* key) {
    if (!ctx || !key) return nullptr;
    BnContext* mut = const_cast<BnContext*>(ctx);
    const int32_t n = llama_model_meta_val_str(
        ctx->model, key, mut->meta_buf, sizeof(mut->meta_buf)
    );
    if (n < 0) return nullptr;
    return mut->meta_buf;
}

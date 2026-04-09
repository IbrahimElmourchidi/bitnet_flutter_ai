/**
 * bitnet_bridge.cc
 *
 * Thin C wrapper over the llama.cpp C API for use with Dart FFI.
 *
 * Build requirements:
 *   - llama.h must be on the include path (from the prebuilt bitnet.cpp/llama.cpp headers)
 *   - Link against the prebuilt libllama (.so/.dylib/.dll) that contains the
 *     BitNet b1.58 I2_S kernels from Microsoft's bitnet.cpp fork.
 *
 * The prebuilt libraries are distributed via GitHub Releases and are NOT
 * vendored in this repository. The CI/CD pipeline fetches them at build time.
 *
 * Thread safety: all functions except bn_free_string are non-reentrant on
 * the same BnContext. The Dart inference isolate ensures single-threaded access.
 */

#include "bitnet_bridge.h"
#include "llama.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

/* ------------------------------------------------------------------
 * Internal context struct
 * ------------------------------------------------------------------ */

struct BnContext {
    llama_model*   model    = nullptr;
    llama_context* ctx      = nullptr;
    llama_sampler* sampler  = nullptr;
    int            n_past   = 0;    // tokens already in the KV cache
    int            n_ctx    = 0;    // context window size
};

/* ------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------ */

static char* make_err(const std::string& msg) {
    char* buf = static_cast<char*>(std::malloc(msg.size() + 1));
    if (buf) std::memcpy(buf, msg.c_str(), msg.size() + 1);
    return buf;
}

static void set_err(char** out_err, const std::string& msg) {
    if (out_err) *out_err = make_err(msg);
}

/* ------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------ */

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

    // Resolve thread count
    const int threads = (n_threads > 0)
        ? static_cast<int>(n_threads)
        : static_cast<int>(std::thread::hardware_concurrency());

    // Load model from GGUF
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU-only; BitNet I2_S kernels are CPU-optimised

    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        set_err(out_err, std::string("Failed to load model from: ") + model_path);
        return nullptr;
    }

    // Resolve context window size
    const int ctx_size = (n_ctx > 0)
        ? static_cast<int>(n_ctx)
        : llama_model_n_ctx_train(model);

    // Create inference context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = static_cast<uint32_t>(ctx_size);
    cparams.n_threads  = static_cast<uint32_t>(threads);
    cparams.n_threads_batch = static_cast<uint32_t>(threads);

    llama_context* lctx = llama_new_context_with_model(model, cparams);
    if (!lctx) {
        llama_model_free(model);
        set_err(out_err, "Failed to create llama context");
        return nullptr;
    }

    // Build sampler chain (greedy by default — deterministic, lowest latency)
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    BnContext* bn = new BnContext();
    bn->model   = model;
    bn->ctx     = lctx;
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

/* ------------------------------------------------------------------
 * Prompt ingestion
 * ------------------------------------------------------------------ */

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

    // Tokenize
    const int max_tokens = ctx->n_ctx;
    std::vector<llama_token> tokens(max_tokens);

    const int n_tokens = llama_tokenize(
        ctx->model,
        prompt,
        static_cast<int32_t>(std::strlen(prompt)),
        tokens.data(),
        max_tokens,
        add_bos != 0,   // add BOS
        false            // special tokens
    );

    if (n_tokens < 0) {
        set_err(out_err, "Tokenisation failed: prompt may be too long");
        return -1;
    }

    tokens.resize(static_cast<size_t>(n_tokens));

    // Reset KV cache position for a fresh conversation turn
    llama_kv_cache_clear(ctx->ctx);
    ctx->n_past = 0;

    // Prefill: batch all prompt tokens in one decode call
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx->ctx, batch) != 0) {
        set_err(out_err, "llama_decode failed during prompt prefill");
        return -1;
    }

    ctx->n_past = static_cast<int>(tokens.size());
    return 0;
}

/* ------------------------------------------------------------------
 * Token generation
 * ------------------------------------------------------------------ */

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

    // Check context window exhaustion
    if (ctx->n_past >= ctx->n_ctx) {
        if (out_done) *out_done = 1;
        return nullptr;
    }

    // Sample the next token from the logits of the last decode
    const llama_token token_id = llama_sampler_sample(ctx->sampler, ctx->ctx, -1);

    // Check for end-of-generation
    if (llama_token_is_eog(ctx->model, token_id)) {
        if (out_done) *out_done = 1;
        return nullptr;
    }

    // Decode token id → UTF-8 piece
    // llama_token_to_piece returns the number of bytes written (may be negative on error)
    char piece_buf[256] = {};
    const int piece_len = llama_token_to_piece(
        ctx->model,
        token_id,
        piece_buf,
        static_cast<int32_t>(sizeof(piece_buf) - 1),
        0,      // lstrip
        true    // special — render special tokens as text
    );

    if (piece_len < 0) {
        set_err(out_err, "llama_token_to_piece failed");
        return nullptr;
    }
    piece_buf[piece_len] = '\0';

    // Feed the sampled token back for the next step
    llama_batch batch = llama_batch_get_one(&token_id, 1);
    if (llama_decode(ctx->ctx, batch) != 0) {
        set_err(out_err, "llama_decode failed during generation step");
        return nullptr;
    }
    ctx->n_past++;

    // Return a heap-allocated copy for the Dart side to bn_free_string()
    char* result = static_cast<char*>(std::malloc(static_cast<size_t>(piece_len) + 1));
    if (!result) {
        set_err(out_err, "malloc failed for token piece");
        return nullptr;
    }
    std::memcpy(result, piece_buf, static_cast<size_t>(piece_len) + 1);
    return result;
}

/* ------------------------------------------------------------------
 * Diagnostics
 * ------------------------------------------------------------------ */

int32_t bn_context_size(const BnContext* ctx) {
    if (!ctx) return -1;
    return static_cast<int32_t>(ctx->n_ctx);
}

const char* bn_meta(const BnContext* ctx, const char* key) {
    if (!ctx || !key) return nullptr;
    return llama_model_meta_val_str(ctx->model, key);
}

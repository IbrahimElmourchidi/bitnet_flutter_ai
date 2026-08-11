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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

// llama.cpp model-load progress (0..1), forwarded to the Web Worker's main
// thread as a postMessage so the UI can show "loading model" progress. Without
// this, bn_init is a long opaque blocking call and the page looks frozen after
// the download hits 100%.
static bool bn_wasm_load_progress(float progress, void* /*user_data*/) {
    // Guard postMessage: it exists in a Web Worker but not in a plain Node
    // context (used by the headless tests), where calling it would throw.
    EM_ASM({
        if (typeof postMessage === 'function') {
            postMessage({ type: 'load_progress', frac: $0 });
        }
    }, (double)progress);
    return true; // continue loading
}
#endif

/* ------------------------------------------------------------------
 * Internal context struct
 * ------------------------------------------------------------------ */

struct BnContext {
    llama_model*   model    = nullptr;
    llama_context* ctx      = nullptr;
    llama_sampler* sampler  = nullptr;
    int            n_past   = 0;    // tokens already in the KV cache
    int            n_ctx    = 0;    // context window size
    // Extra end-of-generation token to honour beyond the vocab's registered EOG
    // set. The BitNet b1.58 2B-4T GGUF ends assistant turns with <|eot_id|> but
    // registers only <|end_of_text|> as EOS, so llama_token_is_eog() never fires
    // on <|eot_id|> and the model runs on, fabricating the next turn. Resolved
    // once at init (see bn_init); LLAMA_TOKEN_NULL (-1) when the model has no
    // such token (e.g. Falcon), in which case it is ignored.
    llama_token    extra_eot_id = LLAMA_TOKEN_NULL;
    // Scratch buffer for bn_meta() lookups. The pointer returned by
    // bn_meta points into this buffer and is only valid until the next
    // bn_meta() call on the same context.
    char           meta_buf[1024] = {};
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
 * End-of-turn token resolution
 * ------------------------------------------------------------------ */

// Resolves the model's end-of-turn token (<|eot_id|> on BitNet/Llama-3 style
// vocabs) so bn_next_token can stop on it even when the GGUF failed to register
// it in the EOG set. Returns LLAMA_TOKEN_NULL when the model has no such token.
//
// NOTE: on the shipped BitNet 2B-4T GGUF this is a belt-and-braces safety net,
// not the actual fix — that GGUF ALREADY marks both <|end_of_text|> and
// <|eot_id|> as EOG (llama_token_is_eog returns true for both), so the base
// check below already catches them. The real cause of the model rambling past
// its turn was a PROMPT-FORMAT bug: the chat template must separate turns with
// <|eot_id|>, not "\n\n", or the model never emits an end-of-turn token in the
// first place (see _formatBitnetLabeled in lib/src/session/chat_templates.dart).
// This resolver still earns its keep for other Llama-3 derivatives whose GGUF
// omits <|eot_id|> from the EOG set.
//
// Strategy, most-authoritative first:
//   1) llama_token_eot(model) — the vocab's declared end-of-turn id, if any.
//   2) Tokenize the literal "<|eot_id|>" with parse_special=true; accept it only
//      if it maps to EXACTLY ONE token (a real special token, not byte-fallback
//      fragments). This catches vocabs that carry the token but don't tag it as
//      "eot". Models without the token tokenize the string into several pieces
//      and are correctly rejected.
static llama_token resolve_extra_eot(llama_model* model) {
    const llama_token declared = llama_token_eot(model);
    if (declared != LLAMA_TOKEN_NULL) {
        return declared;
    }

    static const char kEot[] = "<|eot_id|>";
    llama_token ids[8];
    const int32_t n = llama_tokenize(
        model,
        kEot,
        static_cast<int32_t>(sizeof(kEot) - 1),
        ids,
        static_cast<int32_t>(sizeof(ids) / sizeof(ids[0])),
        /*add_special=*/false,
        /*parse_special=*/true);
    if (n == 1) {
        return ids[0];
    }
    return LLAMA_TOKEN_NULL;
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
#ifdef __EMSCRIPTEN__
    // Report tensor-load progress to the UI (web only).
    mparams.progress_callback = bn_wasm_load_progress;
    mparams.progress_callback_user_data = nullptr;
#endif

    llama_model* model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        set_err(out_err, std::string("Failed to load model from: ") + model_path);
        return nullptr;
    }

    // Resolve context window size
    const int ctx_size = (n_ctx > 0)
        ? static_cast<int>(n_ctx)
        : llama_n_ctx_train(model);

    // Create inference context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = static_cast<uint32_t>(ctx_size);
    cparams.n_threads  = static_cast<uint32_t>(threads);
    cparams.n_threads_batch = static_cast<uint32_t>(threads);
    // Allow a full-context prompt to be submitted in a single logical batch.
    // Dart passes n_ctx up to 4096; llama.cpp's default n_batch is 2048, so a
    // prefill longer than 2048 tokens would be rejected by llama_decode. Sizing
    // n_batch to the context window removes that ceiling. (n_ubatch below still
    // bounds the actual per-step compute chunk.)
    cparams.n_batch = static_cast<uint32_t>(ctx_size);
    // Keep Flash Attention off. This llama.cpp's I2_S CPU path uses the standard
    // (non-fused) attention; leaving flash_attn=false is both the default and the
    // verified-correct path for these ternary kernels.
    cparams.flash_attn = false;
    // Cap the micro-batch. This is now purely a memory/latency knob: it bounds
    // the ggml compute buffer and keeps a single prefill step short. It used to
    // double as a crash guard, because the I2_S gemm prefill path sized a stack
    // VLA as (rows-per-thread * n_ubatch) floats and blew ggml's worker-thread
    // stacks (512 KB by default on Darwin) — 128 was NOT low enough: 1152 rows
    // x 128 tokens x 4 B = 576 KB still overran and took SIGBUS in the stack
    // guard page. That VLA is gone (native/patches/0003-i2s-gemm-no-stack-vla.patch
    // makes the kernel accumulate straight into dst), so prefill stack use no
    // longer scales with the prompt at all.
    if (cparams.n_ubatch > 128) {
        cparams.n_ubatch = 128;
    }

    llama_context* lctx = llama_new_context_with_model(model, cparams);
    if (!lctx) {
        llama_free_model(model);
        set_err(out_err, "Failed to create llama context");
        return nullptr;
    }

    // Build sampler chain. Pure greedy (argmax) on these small 1-2B ternary
    // models degenerates into runaway repetition ("1. … 2. … 3. …", "It is 3.
    // It is 3."), and a repetition penalty alone in front of greedy is not
    // enough to escape it — the argmax just walks to the next-most-repetitive
    // token. The robust fix is real stochastic sampling, matching what
    // bitnet.cpp's own llama-cli uses:
    //   penalties → top_k → top_p → temperature → dist(seed)
    // Order matters: penalize repeats first, then narrow the candidate set,
    // then temperature-scale, then draw. A fixed seed keeps runs reproducible
    // (important for the smoke tests); pass LLAMA_DEFAULT_SEED for nondeterism.
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler,
        llama_sampler_init_penalties(
            llama_n_vocab(model),        // n_vocab
            llama_token_eos(model),      // special_eos_id
            llama_token_nl(model),       // linefeed_id
            /*penalty_last_n=*/256,
            /*penalty_repeat=*/1.3f,
            /*penalty_freq=*/0.0f,
            /*penalty_present=*/0.0f,
            /*penalize_nl=*/false,
            /*ignore_eos=*/false));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(/*k=*/40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(/*p=*/0.95f, /*min_keep=*/1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(/*t=*/0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(/*seed=*/0x5eed5eedu));

    BnContext* bn = new BnContext();
    bn->model   = model;
    bn->ctx     = lctx;
    bn->sampler = sampler;
    bn->n_past  = 0;
    bn->n_ctx   = ctx_size;
    // Resolve the model's end-of-turn token so generation stops natively even
    // when the GGUF omitted it from the EOG set (BitNet 2B-4T <|eot_id|>).
    bn->extra_eot_id = resolve_extra_eot(model);

    return bn;
}

void bn_free(BnContext* ctx) {
    if (!ctx) return;
    if (ctx->sampler) llama_sampler_free(ctx->sampler);
    if (ctx->ctx)     llama_free(ctx->ctx);
    if (ctx->model)   llama_free_model(ctx->model);
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

    // If the caller already prefixed an explicit BOS token (e.g. the
    // Llama-3 chat template starts with "<|begin_of_text|>"), don't add a
    // second one — that double-BOS confuses the model.
    const std::string p(prompt);
    const bool has_explicit_bos =
        p.compare(0, 17, "<|begin_of_text|>") == 0;
    const bool effective_add_bos = (add_bos != 0) && !has_explicit_bos;

    // This llama.cpp takes the model (not a separate vocab handle) for
    // tokenize / token_to_piece / vocab-size / eog queries.
    const int n_tokens = llama_tokenize(
        ctx->model,
        prompt,
        static_cast<int32_t>(p.size()),
        tokens.data(),
        max_tokens,
        effective_add_bos,
        true   // parse_special: render <|...|> markers as their special token IDs
    );

    if (n_tokens < 0) {
        set_err(out_err, "Tokenisation failed: prompt may be too long");
        return -1;
    }

    tokens.resize(static_cast<size_t>(n_tokens));

    std::fprintf(stderr,
        "[bitnet_bridge] bn_prompt: chars=%zu tokens=%d add_bos=%d "
        "parse_special=1 first_ids=[%d,%d,%d]\n",
        p.size(), n_tokens, effective_add_bos ? 1 : 0,
        n_tokens > 0 ? tokens[0] : -1,
        n_tokens > 1 ? tokens[1] : -1,
        n_tokens > 2 ? tokens[2] : -1);

    // Reset the KV cache for a fresh conversation turn.
    llama_kv_cache_clear(ctx->ctx);
    ctx->n_past = 0;

    // Prefill: batch all prompt tokens in one decode call. This llama.cpp's
    // llama_batch_get_one takes explicit (tokens, n_tokens, pos_0, seq_id); we
    // start at position 0 and advance ctx->n_past ourselves for the generation
    // steps below.
    llama_batch batch = llama_batch_get_one(
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        /*pos_0=*/0,
        /*seq_id=*/0
    );
    if (llama_decode(ctx->ctx, batch) != 0) {
        set_err(out_err, "llama_decode failed during prompt prefill");
        return -1;
    }

    // Guard against a libggml that was built without functional I2_S kernels
    // for this CPU/ABI (e.g. Android x86_64 compiled with no AVX2 and no
    // scalar fallback). In that case every matmul writes nothing, the logits
    // come out flat/non-finite, and greedy sampling would emit a runaway
    // stream of a single token ("!"). Detect it here and fail loudly instead.
    const float* logits = llama_get_logits_ith(ctx->ctx, -1);
    if (logits) {
        const int n_vocab = llama_n_vocab(ctx->model);
        bool flat = true;
        bool finite = true;
        for (int i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(logits[i])) { finite = false; break; }
            if (logits[i] != logits[0]) { flat = false; }
        }
        if (!finite || flat) {
            set_err(out_err,
                "BitNet produced degenerate logits (flat or non-finite) after "
                "prompt prefill. The linked libggml was likely built without "
                "functional I2_S kernels for this CPU/ABI.");
            return -1;
        }
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

    // Check for end-of-generation. Honour BOTH the vocab's registered EOG set
    // AND the resolved end-of-turn token: the BitNet 2B-4T GGUF ends turns with
    // <|eot_id|> but registers only <|end_of_text|> as EOS, so llama_token_is_eog
    // alone lets the model run past its turn and fabricate the next one. When the
    // model has no such token, extra_eot_id is LLAMA_TOKEN_NULL and never matches.
    if (llama_token_is_eog(ctx->model, token_id) ||
        (ctx->extra_eot_id != LLAMA_TOKEN_NULL && token_id == ctx->extra_eot_id)) {
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

    // Feed the sampled token back for the next step at the current position.
    // ctx->n_past is the next free KV slot (prefill left it at the prompt length).
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token*>(&token_id),
        1,
        /*pos_0=*/ctx->n_past,
        /*seq_id=*/0
    );
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
    // bitnet.cpp's pinned llama.cpp returns the value into a caller-owned
    // buffer and returns its length. We stash it in the context's scratch
    // buffer; the pointer is valid until the next bn_meta() call.
    BnContext* mut = const_cast<BnContext*>(ctx);
    const int32_t n = llama_model_meta_val_str(
        ctx->model, key, mut->meta_buf, sizeof(mut->meta_buf)
    );
    if (n < 0) return nullptr;
    return mut->meta_buf;
}

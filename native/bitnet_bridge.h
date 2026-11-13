#pragma once

/**
 * bitnet_bridge.h
 *
 * Thin C ABI over the llama.cpp C API, exposing only what bitnet_flutter_ai
 * needs for streaming inference with the BitNet b1.58 2B-4T I2_S GGUF model.
 *
 * All strings are null-terminated UTF-8.
 * All functions are non-reentrant on the same BnContext; call them from a
 * single thread (the Dart inference isolate owns the context exclusively).
 * bn_free_string() is the only function safe to call from any thread.
 *
 * Error strings written to out_err are malloc-allocated. The caller must
 * release them with bn_free_string(). Passing NULL for out_err is valid;
 * the error string is simply discarded.
 *
 * ABI version 2 adds: bn_abi_version, bn_set_sampling (configurable sampler +
 * optional GBNF grammar), bn_next_token_ex (per-token log-probability),
 * bn_token_count, bn_set_abort_flag (cross-thread cancellation of prefill and
 * decode) and bn_last_prompt_reused. bn_prompt now reuses the longest common
 * KV-cache prefix with the previous prompt instead of re-prefilling it.
 * Both native backends (bitnet_bridge.cc, prism_bridge.cc) implement the full
 * v2 ABI through the shared core in bn_core.inc.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------
 * Opaque context handle
 * ------------------------------------------------------------------ */

/** Opaque inference context. One instance per loaded model. */
typedef struct BnContext BnContext;

/** Version of this C ABI. Bumped on any incompatible change. */
#define BN_ABI_VERSION 2

/**
 * Returns BN_ABI_VERSION of the compiled library. The Dart side refuses to
 * drive a library whose version differs (a stale prebuilt binary).
 */
int32_t bn_abi_version(void);

/**
 * Sampler configuration for bn_set_sampling().
 *
 * temperature    <= 0 selects greedy (argmax) decoding; top_k/top_p/seed are
 *                then ignored. Greedy + grammar is the right choice for
 *                structured extraction.
 * top_k          <= 0 disables top-k.
 * top_p          >= 1 disables nucleus sampling.
 * repeat_penalty 1.0 disables the repetition penalty.
 * repeat_last_n  window (tokens) the penalty looks back over.
 * seed           RNG seed for stochastic sampling.
 */
typedef struct BnSamplingParams {
    float    temperature;
    int32_t  top_k;
    float    top_p;
    float    repeat_penalty;
    int32_t  repeat_last_n;
    uint32_t seed;
} BnSamplingParams;

/* ------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------ */

/**
 * Allocate and initialise a BitNet inference context.
 *
 * Loads the GGUF model from disk, allocates the KV cache, and prepares
 * the sampler chain. This call may take several seconds on first run.
 *
 * @param model_path  Null-terminated UTF-8 path to the .gguf file.
 * @param n_ctx       KV-cache context window in tokens. Pass 0 for the
 *                    model's default (clamped to 4096 for 2B-4T).
 * @param n_threads   CPU thread count for inference. Pass 0 to use
 *                    std::thread::hardware_concurrency().
 * @param out_err     If non-NULL and init fails, receives a malloc-allocated
 *                    error string. Caller must bn_free_string() it.
 * @return            Opaque context pointer, or NULL on failure.
 */
BnContext* bn_init(
    const char*  model_path,
    int32_t      n_ctx,
    int32_t      n_threads,
    char**       out_err
);

/**
 * Free all resources owned by the context.
 * Calling bn_free(NULL) is a no-op.
 * The pointer is invalid after this call.
 */
void bn_free(BnContext* ctx);

/**
 * Free a string returned by bn_init or bn_next_token.
 * Calling bn_free_string(NULL) is a no-op.
 * Thread-safe.
 */
void bn_free_string(char* str);

/* ------------------------------------------------------------------
 * Prompt ingestion
 * ------------------------------------------------------------------ */

/**
 * Tokenize prompt and run a prefill forward pass, populating the KV cache.
 *
 * Resets the generation cursor to the end of the prompt. Must be called
 * once per conversation turn before streaming tokens with bn_next_token().
 *
 * @param ctx      Valid non-NULL context.
 * @param prompt   Null-terminated UTF-8 prompt text.
 * @param add_bos  1 → prepend BOS token; 0 → do not.
 * @param out_err  Optional; receives error string on failure.
 * @return         0 on success, -1 on failure, BN_ABORTED if the abort flag
 *                 stopped the prefill.
 *
 * KV-cache reuse: the longest token prefix shared with the previous prompt
 * (plus the tokens generated after it) is kept and only the remainder is
 * decoded, so a fixed system prompt / few-shot block is prefilled once.
 */
int32_t bn_prompt(
    BnContext*   ctx,
    const char*  prompt,
    int32_t      add_bos,
    char**       out_err
);

/* ------------------------------------------------------------------
 * Token generation — call repeatedly after bn_prompt()
 * ------------------------------------------------------------------ */

/**
 * Run one auto-regressive step and return the decoded token piece.
 *
 * Return value semantics:
 *   non-NULL string  → one token piece (UTF-8). Caller must bn_free_string().
 *                      The string may be empty for whitespace-only tokens.
 *   NULL + *out_done == 1  → EOS or max-context reached; generation complete.
 *   NULL + *out_done == 0  → error; check *out_err.
 *
 * @param ctx       Valid non-NULL context.
 * @param out_done  Set to 1 on EOS/completion, 0 otherwise.
 * @param out_err   Optional; receives error string on failure.
 * @return          Caller-owned UTF-8 token piece, or NULL (see above).
 */
char* bn_next_token(
    BnContext*   ctx,
    int32_t*     out_done,
    char**       out_err
);

/* ------------------------------------------------------------------
 * Diagnostics
 * ------------------------------------------------------------------ */

/**
 * Return the context window size of the loaded model.
 * Returns -1 if ctx is NULL.
 */
int32_t bn_context_size(const BnContext* ctx);

/**
 * Return model metadata by GGUF key (e.g. "general.name").
 * The returned pointer points into a per-context scratch buffer: it is valid
 * only until the next bn_meta() call on the same context (or bn_free).
 * Returns NULL if the key is not found.
 */
const char* bn_meta(const BnContext* ctx, const char* key);

/* ------------------------------------------------------------------
 * ABI v2
 * ------------------------------------------------------------------ */

/**
 * Replace the sampler chain used by subsequent generations.
 *
 * @param params  Sampler settings; NULL restores the library defaults
 *                (temp 0.7, top_k 40, top_p 0.95, repeat 1.3 over 256).
 * @param grammar Optional GBNF grammar (root rule "root") constraining the
 *                output; NULL or "" disables grammar constraints.
 * @param out_err Optional; receives an error string on failure.
 * @return        0 on success, -1 on failure (previous sampler kept).
 */
int32_t bn_set_sampling(
    BnContext*              ctx,
    const BnSamplingParams* params,
    const char*             grammar,
    char**                  out_err
);

/**
 * bn_next_token plus the natural-log probability the model assigned to the
 * emitted token (raw softmax over the full vocabulary, before temperature,
 * top-k/p or grammar). Basis for confidence scores. out_logprob may be NULL.
 */
char* bn_next_token_ex(
    BnContext*   ctx,
    int32_t*     out_done,
    float*       out_logprob,
    char**       out_err
);

/**
 * Number of tokens [text] tokenizes to (special markers parsed), including a
 * BOS when add_bos != 0. Returns -1 on error.
 */
int32_t bn_token_count(const BnContext* ctx, const char* text, int32_t add_bos);

/**
 * Register a cancellation flag. While *flag != 0, bn_prompt and
 * bn_next_token(_ex) stop at the next compute-graph boundary: bn_prompt
 * returns BN_ABORTED and bn_next_token(_ex) reports completion
 * (*out_done == 1). The flag may be written from any thread; the caller owns
 * its memory and must keep it alive until bn_free or until it is replaced.
 * Pass NULL to unregister.
 */
void bn_set_abort_flag(BnContext* ctx, volatile int32_t* flag);

/** bn_prompt return code when the abort flag stopped the prefill. */
#define BN_ABORTED (-2)

/**
 * Diagnostics: how many leading prompt tokens the last bn_prompt reused from
 * the KV cache (i.e. did not re-prefill). -1 if ctx is NULL.
 */
int32_t bn_last_prompt_reused(const BnContext* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

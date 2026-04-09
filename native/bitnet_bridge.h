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
 * @return         0 on success, -1 on failure.
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
 * The returned pointer is owned by the context; valid until bn_free(ctx).
 * Returns NULL if the key is not found.
 */
const char* bn_meta(const BnContext* ctx, const char* key);

#ifdef __cplusplus
} /* extern "C" */
#endif

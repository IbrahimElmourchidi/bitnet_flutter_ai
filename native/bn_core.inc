/**
 * bn_core.inc — backend-independent implementation of the bn_* C ABI (v2).
 *
 * Included (not compiled on its own) by bitnet_bridge.cc and prism_bridge.cc.
 * The two llama.cpp generations they target differ only in a handful of calls
 * (tokenize / piece / eog take a model vs a vocab, batch positioning, KV-cache
 * removal, sampler constructors). Each bridge defines, BEFORE including this
 * file:
 *
 *   struct BnBackendExtra { ... };          // per-backend context fields
 *   namespace bnapi {
 *     constexpr const char* kTag;            // log prefix, e.g. "bitnet_bridge"
 *     int  n_vocab   (const BnContext*);
 *     int  tokenize  (const BnContext*, const char*, int len, llama_token*, int max,
 *                     bool add_special);     // parse_special always true;
 *                                            // returns -needed when max too small
 *     int  piece     (const BnContext*, llama_token, char* buf, int len);
 *     bool is_eog    (const BnContext*, llama_token);
 *     int  decode    (BnContext*, llama_token*, int n, int pos0);
 *     bool kv_seq_rm (BnContext*, int p0);   // drop positions [p0, end)
 *     void kv_clear  (BnContext*);
 *     llama_sampler* penalties(const BnContext*, int last_n, float repeat);
 *     llama_sampler* grammar  (const BnContext*, const char* gbnf);
 *     void free_model(llama_model*);
 *     constexpr bool kAbortInCompute;        // see "Cancellation" below
 *   }
 *
 * and implements bn_init itself (model/context creation differs), finishing
 * with bn_core_attach().
 *
 * Design notes
 *  - KV prefix reuse: kv_tokens mirrors exactly which tokens occupy KV
 *    positions [0, n). A new prompt keeps the longest common prefix and decodes
 *    only the rest, so a fixed system prompt + few-shot block (Fino's
 *    transaction extractor) is prefilled once, not on every voice entry.
 *  - Sampling is done by hand (not llama_sampler_sample) so that we can
 *    (a) report the raw log-probability of the chosen token for confidence
 *    scores and (b) apply a GBNF grammar cheaply: sample unconstrained first
 *    and only run the (vocab-sized) grammar pass when that token is rejected —
 *    the same trick llama.cpp's common_sampler uses.
 *  - Cancellation: an abort flag owned by the caller is checked between
 *    prefill chunks and before every generated token. When the backend's ggml
 *    supports it (kAbortInCompute), the flag is also polled inside graph
 *    compute via llama's abort callback, so even one long chunk stops early.
 *    bitnet.cpp's pinned ggml (OpenMP build) deadlocks its worker barrier when
 *    that callback fires (verified: every thread parks in futex_wait), so that
 *    backend instead prefills in micro-batch-sized chunks and checks between
 *    them. After any aborted decode the KV state is treated as unknown and
 *    discarded.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct BnContext {
    llama_model*   model   = nullptr;
    llama_context* ctx     = nullptr;
    BnBackendExtra be;

    llama_sampler* chain   = nullptr;   // penalties → (greedy | top_k → top_p → temp → dist)
    llama_sampler* grammar = nullptr;   // optional GBNF constraint

    // Tokens whose K/V currently occupy positions [0, kv_tokens.size()).
    std::vector<llama_token> kv_tokens;
    int n_ctx   = 0;
    int n_batch = 0;
    int last_reused = 0;

    volatile int32_t* abort_flag = nullptr;

    std::vector<llama_token_data> cur;  // sampling scratch, n_vocab entries
    std::string piece_buf;              // token_to_piece scratch

    // Scratch buffer for bn_meta(); valid until the next bn_meta() call.
    char meta_buf[1024] = {};
};

/* ------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------ */

static char* bn_make_err(const std::string& msg) {
    char* buf = static_cast<char*>(std::malloc(msg.size() + 1));
    if (buf) std::memcpy(buf, msg.c_str(), msg.size() + 1);
    return buf;
}

static void bn_set_err(char** out_err, const std::string& msg) {
    if (out_err) *out_err = bn_make_err(msg);
}

static bool bn_aborted(const BnContext* bn) {
    return bn->abort_flag != nullptr &&
           __atomic_load_n(bn->abort_flag, __ATOMIC_RELAXED) != 0;
}

static bool bn_abort_cb(void* data) {
    return bn_aborted(static_cast<const BnContext*>(data));
}

// KV contents are unknown (aborted / failed decode): forget all of it so the
// next prompt does a full, correct prefill.
static void bn_invalidate_kv(BnContext* bn) {
    bnapi::kv_clear(bn);
    bn->kv_tokens.clear();
}

static const BnSamplingParams kBnDefaultSampling = {
    /*temperature=*/0.7f,
    /*top_k=*/40,
    /*top_p=*/0.95f,
    /*repeat_penalty=*/1.3f,
    /*repeat_last_n=*/256,
    /*seed=*/0x5eed5eedu,
};

static llama_sampler* bn_build_chain(const BnContext* bn, const BnSamplingParams& p) {
    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (p.repeat_penalty != 1.0f && p.repeat_last_n != 0) {
        llama_sampler_chain_add(chain,
            bnapi::penalties(bn, p.repeat_last_n, p.repeat_penalty));
    }
    if (p.temperature <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        if (p.top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(p.top_k));
        if (p.top_p < 1.0f) llama_sampler_chain_add(chain, llama_sampler_init_top_p(p.top_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(p.temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(p.seed));
    }
    return chain;
}

// Shared tail of bn_init: wire the sampler, abort callback and bookkeeping.
static BnContext* bn_core_attach(BnContext* bn, int ctx_size) {
    bn->n_ctx   = ctx_size;
    bn->n_batch = static_cast<int>(llama_n_batch(bn->ctx));
    if (bn->n_batch <= 0) bn->n_batch = 512;
    if (!bnapi::kAbortInCompute) {
        // Cancellation granularity = one prefill chunk; a micro-batch-sized
        // chunk costs nothing extra (llama_decode splits into ubatches anyway).
        const int ub = static_cast<int>(llama_n_ubatch(bn->ctx));
        if (ub > 0 && ub < bn->n_batch) bn->n_batch = ub;
    }
    bn->chain   = bn_build_chain(bn, kBnDefaultSampling);
    bn->cur.resize(static_cast<size_t>(bnapi::n_vocab(bn)));
    if (bnapi::kAbortInCompute) llama_set_abort_callback(bn->ctx, bn_abort_cb, bn);
    return bn;
}

// Tokenizes [text] into [out]. Returns false (with *needed set) on failure.
static bool bn_tokenize_all(const BnContext* bn, const char* text, bool add_special,
                            std::vector<llama_token>& out) {
    const int len = static_cast<int>(std::strlen(text));
    int n = bnapi::tokenize(bn, text, len, nullptr, 0, add_special);
    if (n < 0) n = -n;
    out.resize(static_cast<size_t>(n));
    if (n == 0) return true;
    const int got = bnapi::tokenize(bn, text, len, out.data(), n, add_special);
    if (got < 0) return false;
    out.resize(static_cast<size_t>(got));
    return true;
}

/* ------------------------------------------------------------------
 * ABI
 * ------------------------------------------------------------------ */

int32_t bn_abi_version(void) { return BN_ABI_VERSION; }

void bn_free(BnContext* bn) {
    if (!bn) return;
    if (bn->grammar) llama_sampler_free(bn->grammar);
    if (bn->chain)   llama_sampler_free(bn->chain);
    if (bn->ctx)     llama_free(bn->ctx);
    if (bn->model)   bnapi::free_model(bn->model);
    delete bn;
}

void bn_free_string(char* str) { std::free(str); }

int32_t bn_set_sampling(BnContext* bn, const BnSamplingParams* params,
                        const char* grammar, char** out_err) {
    if (!bn) {
        bn_set_err(out_err, "ctx must not be NULL");
        return -1;
    }
    llama_sampler* g = nullptr;
    if (grammar && grammar[0] != '\0') {
        g = bnapi::grammar(bn, grammar);
        // Newer llama.cpp returns NULL on a parse error; bitnet.cpp's older one
        // returns a sampler with no grammar inside, which silently constrains
        // nothing. Probe it: a real grammar must reject at least one token at
        // the start position. (A grammar that rejects nothing is a no-op
        // either way, so refusing it loses nothing.)
        if (g) {
            for (size_t i = 0; i < bn->cur.size(); ++i) {
                bn->cur[i] = llama_token_data{static_cast<llama_token>(i), 0.0f, 0.0f};
            }
            llama_token_data_array arr = {bn->cur.data(), bn->cur.size(), -1, false};
            llama_sampler_apply(g, &arr);
            bool rejects_any = false;
            for (size_t i = 0; i < arr.size; ++i) {
                if (!std::isfinite(arr.data[i].logit)) { rejects_any = true; break; }
            }
            if (!rejects_any) {
                llama_sampler_free(g);
                g = nullptr;
            }
        }
        if (!g) {
            bn_set_err(out_err, "Invalid GBNF grammar (parse error, or root rule "
                                "missing / unconstrained)");
            return -1;
        }
    }
    llama_sampler* chain = bn_build_chain(bn, params ? *params : kBnDefaultSampling);
    if (bn->chain)   llama_sampler_free(bn->chain);
    if (bn->grammar) llama_sampler_free(bn->grammar);
    bn->chain   = chain;
    bn->grammar = g;
    return 0;
}

void bn_set_abort_flag(BnContext* bn, volatile int32_t* flag) {
    if (bn) bn->abort_flag = flag;
}

int32_t bn_last_prompt_reused(const BnContext* bn) {
    return bn ? bn->last_reused : -1;
}

int32_t bn_token_count(const BnContext* bn, const char* text, int32_t add_bos) {
    if (!bn || !text) return -1;
    std::vector<llama_token> toks;
    if (!bn_tokenize_all(bn, text, add_bos != 0, toks)) return -1;
    return static_cast<int32_t>(toks.size());
}

int32_t bn_prompt(BnContext* bn, const char* prompt, int32_t add_bos, char** out_err) {
    if (!bn || !prompt) {
        bn_set_err(out_err, "ctx and prompt must not be NULL");
        return -1;
    }

    std::vector<llama_token> toks;
    if (!bn_tokenize_all(bn, prompt, add_bos != 0, toks)) {
        bn_set_err(out_err, "Tokenisation failed");
        return -1;
    }
    if (toks.empty()) {
        bn_set_err(out_err, "Prompt tokenized to zero tokens");
        return -1;
    }
    // Leave at least one slot to generate into. The "prompt_too_long:" prefix
    // is matched by the Dart side to raise BitNetPromptTooLongException.
    if (static_cast<int>(toks.size()) >= bn->n_ctx) {
        bn_set_err(out_err, "prompt_too_long: prompt is " +
                   std::to_string(toks.size()) + " tokens; context window is " +
                   std::to_string(bn->n_ctx));
        return -1;
    }

    // Fresh generation: forget the previous turn's sampling history.
    llama_sampler_reset(bn->chain);
    if (bn->grammar) llama_sampler_reset(bn->grammar);

    // Longest common prefix with what is already in the KV cache. Always
    // re-decode at least the final prompt token so fresh logits exist.
    size_t keep = 0;
    const size_t max_keep = toks.size() - 1;
    while (keep < max_keep && keep < bn->kv_tokens.size() &&
           bn->kv_tokens[keep] == toks[keep]) {
        ++keep;
    }
    if (keep < bn->kv_tokens.size()) {
        if (!bnapi::kv_seq_rm(bn, static_cast<int>(keep))) {
            bn_invalidate_kv(bn);
            keep = 0;
        }
        bn->kv_tokens.resize(keep);
    }
    bn->last_reused = static_cast<int>(keep);

#ifdef BN_DEBUG_LOG
    std::fprintf(stderr, "[%s] bn_prompt: tokens=%zu reused=%zu\n",
                 bnapi::kTag, toks.size(), keep);
#endif

    // Prefill the remainder in n_batch-sized chunks (a single oversized batch
    // is rejected by llama_decode), checking the abort flag between chunks.
    for (size_t i = keep; i < toks.size();) {
        if (bn_aborted(bn)) {
            bn_invalidate_kv(bn);
            return BN_ABORTED;
        }
        const int n = static_cast<int>(
            std::min(static_cast<size_t>(bn->n_batch), toks.size() - i));
        const int rc = bnapi::decode(bn, toks.data() + i, n,
                                     static_cast<int>(bn->kv_tokens.size()));
        if (bn_aborted(bn)) {
            bn_invalidate_kv(bn);
            return BN_ABORTED;
        }
        if (rc != 0) {
            bn_invalidate_kv(bn);
            bn_set_err(out_err, "llama_decode failed during prompt prefill");
            return -1;
        }
        bn->kv_tokens.insert(bn->kv_tokens.end(), toks.begin() + i, toks.begin() + i + n);
        i += static_cast<size_t>(n);
    }

    // Guard against a libggml built without functional kernels for this
    // CPU/ABI: logits come out flat or non-finite and sampling would stream a
    // single token forever. Fail loudly instead.
    const float* logits = llama_get_logits_ith(bn->ctx, -1);
    if (logits) {
        const int n_vocab = bnapi::n_vocab(bn);
        bool flat = true;
        bool finite = true;
        for (int i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(logits[i])) { finite = false; break; }
            if (logits[i] != logits[0]) flat = false;
        }
        if (!finite || flat) {
            bn_invalidate_kv(bn);
            bn_set_err(out_err, std::string(bnapi::kTag) +
                ": degenerate logits (flat or non-finite) after prompt prefill. "
                "The linked libggml was likely built without functional kernels "
                "for this CPU/ABI.");
            return -1;
        }
    }
    return 0;
}

// Picks the next token from the current logits and returns its raw logprob.
static llama_token bn_sample(BnContext* bn, float* out_logprob) {
    const float* logits = llama_get_logits_ith(bn->ctx, -1);
    const int n_vocab = static_cast<int>(bn->cur.size());

    auto fill = [&]() {
        for (int i = 0; i < n_vocab; ++i) bn->cur[i] = llama_token_data{i, logits[i], 0.0f};
    };
    auto run = [&](llama_sampler* first, llama_sampler* second) -> llama_token {
        llama_token_data_array arr = {bn->cur.data(), static_cast<size_t>(n_vocab), -1, false};
        if (first) llama_sampler_apply(first, &arr);
        llama_sampler_apply(second, &arr);
        if (arr.selected < 0 || arr.selected >= static_cast<int64_t>(arr.size)) {
            return arr.data[0].id;  // defensive; greedy/dist always select
        }
        return arr.data[arr.selected].id;
    };

    fill();
    llama_token tok = run(nullptr, bn->chain);

    if (bn->grammar) {
        // Fast path: is the unconstrained choice already grammatical?
        llama_token_data single = {tok, 1.0f, 0.0f};
        llama_token_data_array one = {&single, 1, -1, false};
        llama_sampler_apply(bn->grammar, &one);
        if (!std::isfinite(single.logit)) {
            fill();
            tok = run(bn->grammar, bn->chain);
        }
    }

    if (out_logprob) {
        float mx = logits[0];
        for (int i = 1; i < n_vocab; ++i) mx = logits[i] > mx ? logits[i] : mx;
        double sum = 0.0;
        for (int i = 0; i < n_vocab; ++i) sum += std::exp(static_cast<double>(logits[i] - mx));
        *out_logprob = static_cast<float>(logits[tok] - mx - std::log(sum));
    }
    return tok;
}

char* bn_next_token_ex(BnContext* bn, int32_t* out_done, float* out_logprob, char** out_err) {
    if (out_done) *out_done = 0;
    if (out_logprob) *out_logprob = 0.0f;
    if (!bn) {
        bn_set_err(out_err, "ctx must not be NULL");
        return nullptr;
    }
    // Cancelled, or context window full: a clean stop.
    if (bn_aborted(bn) || static_cast<int>(bn->kv_tokens.size()) >= bn->n_ctx) {
        if (out_done) *out_done = 1;
        return nullptr;
    }

    const llama_token tok = bn_sample(bn, out_logprob);
    if (bnapi::is_eog(bn, tok)) {
        if (out_done) *out_done = 1;
        return nullptr;
    }

    // token → UTF-8 bytes (may be a fragment of a multi-byte character).
    bn->piece_buf.resize(64);
    int len = bnapi::piece(bn, tok, &bn->piece_buf[0], static_cast<int>(bn->piece_buf.size()));
    if (len < 0) {
        bn->piece_buf.resize(static_cast<size_t>(-len));
        len = bnapi::piece(bn, tok, &bn->piece_buf[0], static_cast<int>(bn->piece_buf.size()));
    }
    if (len < 0) {
        bn_set_err(out_err, "llama_token_to_piece failed");
        return nullptr;
    }

    if (bn->grammar) llama_sampler_accept(bn->grammar, tok);
    llama_sampler_accept(bn->chain, tok);

    llama_token t = tok;
    const int rc = bnapi::decode(bn, &t, 1, static_cast<int>(bn->kv_tokens.size()));
    if (bn_aborted(bn)) {
        bn_invalidate_kv(bn);
        if (out_done) *out_done = 1;
        return nullptr;
    }
    if (rc != 0) {
        bn_invalidate_kv(bn);
        bn_set_err(out_err, "llama_decode failed during generation step");
        return nullptr;
    }
    bn->kv_tokens.push_back(tok);

    char* result = static_cast<char*>(std::malloc(static_cast<size_t>(len) + 1));
    if (!result) {
        bn_set_err(out_err, "malloc failed for token piece");
        return nullptr;
    }
    std::memcpy(result, bn->piece_buf.data(), static_cast<size_t>(len));
    result[len] = '\0';
    return result;
}

char* bn_next_token(BnContext* bn, int32_t* out_done, char** out_err) {
    return bn_next_token_ex(bn, out_done, nullptr, out_err);
}

int32_t bn_context_size(const BnContext* bn) {
    return bn ? static_cast<int32_t>(bn->n_ctx) : -1;
}

const char* bn_meta(const BnContext* bn, const char* key) {
    if (!bn || !key) return nullptr;
    BnContext* mut = const_cast<BnContext*>(bn);
    const int32_t n = llama_model_meta_val_str(bn->model, key, mut->meta_buf, sizeof(mut->meta_buf));
    return n < 0 ? nullptr : mut->meta_buf;
}

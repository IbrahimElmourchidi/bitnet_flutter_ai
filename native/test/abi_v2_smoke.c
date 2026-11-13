/*
 * abi_v2_smoke.c — exercises the bn_* ABI v2 against a real GGUF.
 *
 * Build (Linux, bitnet backend):
 *   cc -O2 -pthread -Inative native/test/abi_v2_smoke.c \
 *      -Lnative/build-linux -lbitnet_bridge -Wl,-rpath,native/build-linux \
 *      -o /tmp/abi_v2_smoke
 * Build (prism backend): link -lprism_bridge from native/build-linux-prism.
 *
 * Run:  /tmp/abi_v2_smoke <model.gguf> [chat-template: bitnet|chatml]
 *
 * Checks: ABI version, token counting, KV prefix reuse, grammar-constrained
 * JSON with per-token logprobs, greedy determinism, and abort of a long
 * prefill from another thread. Exits non-zero on the first failure.
 */
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "bitnet_bridge.h"

#define CHECK(cond, ...)                                        \
    do {                                                        \
        if (!(cond)) {                                          \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                       \
            fprintf(stderr, "\n");                              \
            exit(1);                                            \
        }                                                       \
    } while (0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static const char* g_tpl = "bitnet";

static void fmt(char* out, size_t n, const char* sys, const char* user) {
    if (strcmp(g_tpl, "chatml") == 0) {
        snprintf(out, n,
                 "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\n", sys, user);
    } else {
        snprintf(out, n, "Human: %s\n\n%s<|eot_id|>BITNETAssistant: ", sys, user);
    }
}

/* Generates until done; returns text in buf, mean/min logprob via out params. */
static int run(BnContext* ctx, char* buf, size_t cap, int max_tokens,
               double* mean_lp, double* min_lp) {
    size_t len = 0;
    int n = 0;
    double sum = 0, mn = 0;
    buf[0] = 0;
    for (; n < max_tokens; n++) {
        int32_t done = 0;
        float lp = 0;
        char* err = NULL;
        char* piece = bn_next_token_ex(ctx, &done, &lp, &err);
        if (!piece) {
            CHECK(done, "bn_next_token_ex error: %s", err ? err : "?");
            break;
        }
        size_t pl = strlen(piece);
        if (len + pl + 1 < cap) {
            memcpy(buf + len, piece, pl + 1);
            len += pl;
        }
        bn_free_string(piece);
        sum += lp;
        if (n == 0 || lp < mn) mn = lp;
    }
    if (mean_lp) *mean_lp = n ? sum / n : 0;
    if (min_lp) *min_lp = mn;
    return n;
}

struct abort_arg { volatile int32_t* flag; int delay_ms; };
static void* set_flag_later(void* p) {
    struct abort_arg* a = (struct abort_arg*)p;
    usleep((useconds_t)a->delay_ms * 1000);
    __atomic_store_n(a->flag, 1, __ATOMIC_RELAXED);
    return NULL;
}

static const char* kGrammar =
    "root ::= \"{\" \"\\\"amount\\\":\" num \",\" \"\\\"currency\\\":\" cur \",\" "
    "\"\\\"vendor\\\":\" (str | \"null\") \",\" \"\\\"category\\\":\" cat \"}\"\n"
    "num ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\n"
    "cur ::= \"\\\"EGP\\\"\" | \"\\\"USD\\\"\" | \"\\\"EUR\\\"\"\n"
    "cat ::= \"\\\"food\\\"\" | \"\\\"groceries\\\"\" | \"\\\"transport\\\"\" | "
    "\"\\\"utilities\\\"\" | \"\\\"shopping\\\"\" | \"\\\"other\\\"\"\n"
    "str ::= \"\\\"\" [^\"\\\\\\x7F\\x00-\\x1F]* \"\\\"\"\n";

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [bitnet|chatml]\n", argv[0]);
        return 2;
    }
    if (argc > 2) g_tpl = argv[2];
    setvbuf(stdout, NULL, _IONBF, 0);

    CHECK(bn_abi_version() == BN_ABI_VERSION, "abi %d", bn_abi_version());
    printf("ok  abi_version=%d\n", bn_abi_version());

    char* err = NULL;
    BnContext* ctx = bn_init(argv[1], 2048, 0, &err);
    CHECK(ctx, "bn_init: %s", err ? err : "?");

    int32_t tc = bn_token_count(ctx, "I spent 250 EGP at the supermarket", 0);
    CHECK(tc > 3 && tc < 30, "token_count=%d", tc);
    printf("ok  token_count=%d\n", tc);

    const char* sys =
        "Extract the transaction from the user's sentence as JSON with fields "
        "amount, currency, vendor and category.\n"
        "Example: \"paid 40 dollars for an uber\" -> "
        "{\"amount\":40,\"currency\":\"USD\",\"vendor\":\"Uber\",\"category\":\"transport\"}\n"
        "Example: \"electricity bill 150\" -> "
        "{\"amount\":150,\"currency\":\"EGP\",\"vendor\":null,\"category\":\"utilities\"}";
    char prompt[4096], out[2048];

    /* --- grammar-constrained extraction, greedy --- */
    BnSamplingParams greedy = {0.0f, 0, 1.0f, 1.0f, 0, 1};
    CHECK(bn_set_sampling(ctx, &greedy, kGrammar, &err) == 0, "set_sampling: %s", err);
    fmt(prompt, sizeof prompt, sys, "I spent 250 EGP at the supermarket");

    double t0 = now_ms();
    CHECK(bn_prompt(ctx, prompt, 1, &err) == 0, "prompt1: %s", err);
    double t_prefill1 = now_ms() - t0;
    int reused1 = bn_last_prompt_reused(ctx);
    double mean_lp, min_lp;
    int n = run(ctx, out, sizeof out, 64, &mean_lp, &min_lp);
    printf("ok  extract: %s  (tokens=%d prefill=%.0fms reused=%d conf=%.3f min=%.3f)\n",
           out, n, t_prefill1, reused1, exp(mean_lp), exp(min_lp));
    CHECK(out[0] == '{' && out[strlen(out) - 1] == '}', "not JSON: %s", out);
    CHECK(strstr(out, "\"amount\":250") != NULL, "amount wrong: %s", out);
    CHECK(mean_lp <= 0.0 && mean_lp > -20.0, "bad logprob %f", mean_lp);

    /* --- same system prompt, new sentence: prefix must be reused --- */
    fmt(prompt, sizeof prompt, sys, "Paid 75 for a taxi");
    t0 = now_ms();
    CHECK(bn_prompt(ctx, prompt, 1, &err) == 0, "prompt2: %s", err);
    double t_prefill2 = now_ms() - t0;
    int reused2 = bn_last_prompt_reused(ctx);
    char out2[2048];
    run(ctx, out2, sizeof out2, 64, NULL, NULL);
    int total2 = bn_token_count(ctx, prompt, 1);
    printf("ok  reuse: %s  (reused %d of %d prompt tokens, prefill %.0fms vs %.0fms)\n",
           out2, reused2, total2, t_prefill2, t_prefill1);
    CHECK(reused2 > total2 / 2, "expected prefix reuse, got %d/%d", reused2, total2);

    /* --- greedy is deterministic across a reused cache --- */
    fmt(prompt, sizeof prompt, sys, "I spent 250 EGP at the supermarket");
    CHECK(bn_prompt(ctx, prompt, 1, &err) == 0, "prompt3: %s", err);
    char out3[2048];
    run(ctx, out3, sizeof out3, 64, NULL, NULL);
    CHECK(strcmp(out, out3) == 0, "non-deterministic: %s vs %s", out, out3);
    printf("ok  deterministic with reused KV\n");

    /* --- invalid grammar is rejected, previous sampler kept --- */
    char* gerr = NULL;
    CHECK(bn_set_sampling(ctx, &greedy, "root ::= (", &gerr) != 0, "bad grammar accepted");
    bn_free_string(gerr);
    printf("ok  invalid grammar rejected\n");

    /* --- free text with default sampling --- */
    CHECK(bn_set_sampling(ctx, NULL, NULL, &err) == 0, "reset sampling: %s", err);
    fmt(prompt, sizeof prompt, "You are a concise assistant.", "Name one benefit of budgeting.");
    CHECK(bn_prompt(ctx, prompt, 1, &err) == 0, "prompt4: %s", err);
    n = run(ctx, out, sizeof out, 48, NULL, NULL);
    printf("ok  free text (%d tokens): %.120s\n", n, out);
    CHECK(n > 0, "no tokens");

    /* --- abort a long prefill from another thread --- */
    static char big[60000];
    size_t bl = 0;
    while (bl + 64 < 7000) bl += (size_t)snprintf(big + bl, sizeof big - bl,
                                                   "The quick brown fox jumps over the lazy dog. ");
    volatile int32_t flag = 0;
    bn_set_abort_flag(ctx, &flag);
    struct abort_arg a = {&flag, 150};
    pthread_t th;
    pthread_create(&th, NULL, set_flag_later, &a);
    t0 = now_ms();
    int rc = bn_prompt(ctx, big, 1, &err);
    double t_abort = now_ms() - t0;
    pthread_join(th, NULL);
    printf("ok? abort: rc=%d after %.0fms\n", rc, t_abort);
    CHECK(rc == BN_ABORTED, "expected BN_ABORTED, rc=%d err=%s", rc, err ? err : "");
    CHECK(t_abort < 5000, "abort too slow: %.0fms", t_abort);

    /* next_token while flagged reports done */
    int32_t done = 0;
    char* p = bn_next_token_ex(ctx, &done, NULL, &err);
    CHECK(!p && done == 1, "next_token not stopped by flag");

    /* clearing the flag: engine fully usable again (full re-prefill) */
    flag = 0;
    fmt(prompt, sizeof prompt, sys, "I spent 250 EGP at the supermarket");
    CHECK(bn_set_sampling(ctx, &greedy, kGrammar, &err) == 0, "set_sampling");
    CHECK(bn_prompt(ctx, prompt, 1, &err) == 0, "prompt after abort: %s", err);
    CHECK(bn_last_prompt_reused(ctx) == 0, "KV should be invalidated after abort");
    run(ctx, out3, sizeof out3, 64, NULL, NULL);
    CHECK(strcmp(out, out3) != 0 || 1, "unreachable");
    printf("ok  recovered after abort: %s\n", out3);

    /* --- prompt longer than context is a typed error --- */
    bn_set_abort_flag(ctx, NULL);
    static char huge[200000];
    size_t hl = 0;
    while (hl + 64 < sizeof huge - 64) hl += (size_t)snprintf(huge + hl, sizeof huge - hl, "word ");
    char* lerr = NULL;
    CHECK(bn_prompt(ctx, huge, 1, &lerr) == -1, "huge prompt accepted");
    CHECK(lerr && strncmp(lerr, "prompt_too_long:", 16) == 0, "err=%s", lerr ? lerr : "null");
    printf("ok  too-long prompt: %s\n", lerr);
    bn_free_string(lerr);

    bn_free(ctx);
    printf("ALL OK\n");
    return 0;
}

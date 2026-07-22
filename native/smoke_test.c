/*
 * Standalone C smoke test for libbitnet_bridge.
 *
 * Build:
 *   cd native
 *   gcc smoke_test.c -o build-linux/smoke_test \
 *       -I. -L./build-linux -lbitnet_bridge \
 *       -Wl,-rpath,'$ORIGIN' \
 *       -Wl,-rpath,'$ORIGIN/../../bitnet.cpp/build-linux/3rdparty/llama.cpp/src' \
 *       -Wl,-rpath,'$ORIGIN/../../bitnet.cpp/build-linux/3rdparty/llama.cpp/ggml/src'
 *
 * Run:
 *   ./build-linux/smoke_test /path/to/ggml-model-i2_s.gguf
 */

#include "bitnet_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt]\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* prompt = (argc > 2) ? argv[2] : "Hello, who are you?";
    const int   max_new = 32;

    char* err = NULL;

    fprintf(stderr, "[smoke] init...\n");
    BnContext* ctx = bn_init(model_path, /*n_ctx=*/512, /*n_threads=*/4, &err);
    if (!ctx) {
        fprintf(stderr, "[smoke] bn_init failed: %s\n", err ? err : "(null)");
        if (err) bn_free_string(err);
        return 2;
    }
    fprintf(stderr, "[smoke] context size = %d\n", bn_context_size(ctx));

    fprintf(stderr, "[smoke] prompt: %s\n", prompt);
    if (bn_prompt(ctx, prompt, /*add_bos=*/1, &err) != 0) {
        fprintf(stderr, "[smoke] bn_prompt failed: %s\n", err ? err : "(null)");
        if (err) bn_free_string(err);
        bn_free(ctx);
        return 3;
    }

    fprintf(stderr, "[smoke] streaming up to %d tokens:\n", max_new);
    fprintf(stdout, "RESPONSE: ");
    fflush(stdout);

    for (int i = 0; i < max_new; i++) {
        int32_t done = 0;
        char* tok = bn_next_token(ctx, &done, &err);
        if (!tok) {
            if (done) {
                fprintf(stderr, "\n[smoke] EOS at token %d\n", i);
                break;
            }
            fprintf(stderr, "\n[smoke] bn_next_token error: %s\n", err ? err : "(null)");
            if (err) bn_free_string(err);
            bn_free(ctx);
            return 4;
        }
        fputs(tok, stdout);
        fflush(stdout);
        bn_free_string(tok);
    }
    fputc('\n', stdout);

    bn_free(ctx);
    fprintf(stderr, "[smoke] OK\n");
    return 0;
}

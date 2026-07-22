/* Smoke test for the PrismML (Q2_0) backend: drives the bn_* C ABI exported by
 * libprism_bridge.so against a real Ternary Bonsai GGUF, using the chatmlQwen3
 * prompt (empty think-block prefill). Confirms Q2_0/qwen3 loads + generates
 * coherently through the prism fork on this host.
 *
 *   gcc -I native native/prism_smoke.c \
 *       -L native/build-linux-prism -lprism_bridge -o /tmp/prism_smoke
 *   LD_LIBRARY_PATH=native/build-linux-prism /tmp/prism_smoke <model.gguf>
 */
#include <stdio.h>
#include <stdlib.h>
#include "bitnet_bridge.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const char* model_path = argv[1];

    /* chatmlQwen3 template (see chat_templates.dart _formatChatmlQwen3). */
    const char* prompt =
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n";

    char* err = NULL;
    BnContext* ctx = bn_init(model_path, 512, 4, &err);
    if (!ctx) {
        fprintf(stderr, "bn_init FAILED: %s\n", err ? err : "(null)");
        bn_free_string(err);
        return 1;
    }
    printf("bn_init OK (ctx_size=%d)\n", bn_context_size(ctx));

    if (bn_prompt(ctx, prompt, 1, &err) != 0) {
        fprintf(stderr, "bn_prompt FAILED: %s\n", err ? err : "(null)");
        bn_free_string(err);
        bn_free(ctx);
        return 1;
    }
    printf("bn_prompt OK\ngenerated: ");

    int produced = 0;
    for (int i = 0; i < 24; ++i) {
        int done = 0;
        char* piece = bn_next_token(ctx, &done, &err);
        if (!piece) {
            if (done) { printf(" [EOG at %d]", i); break; }
            fprintf(stderr, "\nbn_next_token FAILED: %s\n", err ? err : "(null)");
            bn_free_string(err);
            bn_free(ctx);
            return 1;
        }
        fputs(piece, stdout);
        fflush(stdout);
        bn_free_string(piece);
        produced++;
    }
    printf("\n");
    bn_free(ctx);

    if (produced == 0) {
        fprintf(stderr, "FAIL: no tokens produced\n");
        return 1;
    }
    printf("PASS: prism backend produced %d tokens\n", produced);
    return 0;
}

/*
 * i2s_kernel_test.c — correctness test for the BitNet I2_S vec_dot kernel.
 *
 * Links against the built libggml and drives the exported dispatcher
 * ggml_vec_dot_i2_i8_s through BOTH of its live paths:
 *   - nrc == 1              -> _1x1  (weight-rows semantics; the gemv/decode path)
 *   - nrc  % PARALLEL_SIZE  -> _Nx1  (activation-columns semantics; the gemm path)
 *
 * For each it compares the kernel result — after the same act_sum/act_scale
 * correction the real caller applies — against an independent float reference,
 * and cross-checks the on-disk packing with dequantize_row_i2_s.
 *
 * Build (against the AVX2 desktop build):
 *   gcc native/test/i2s_kernel_test.c -o /tmp/i2s_test \
 *       -I<bitnet.cpp>/3rdparty/llama.cpp/ggml/src \
 *       -L<ggml_dir> -lggml -lm -Wl,-rpath,<ggml_dir>
 *
 * Build (against a scalar build with GGML_NATIVE=OFF) — same, pointing -L at
 * that build's libggml. Exit code 0 = all cases within tolerance.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Exported by libggml (BitNet fork). */
extern void ggml_vec_dot_i2_i8_s(int n, float *s, size_t bs, const void *vx,
                                 size_t bx, const void *vy, size_t by, int nrc);
extern void dequantize_row_i2_s(const uint8_t *x, float *y, int64_t n, float i2_scale);
extern void quantize_row_i8_s(const float *x, void *y, int64_t n,
                              float *act_scales, int32_t *act_sums);

/* Canonical i2_s packer: ternary {-1,0,1} -> code {0,1,2}; each 128-weight
 * block is 32 bytes, byte b holds weights b,32+b,64+b,96+b at shifts 6,4,2,0. */
static void pack_row(const int8_t *w, uint8_t *out, int n) {
    const int nb = n / 128;
    memset(out, 0, (size_t)nb * 32);
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < 128; ++j) {
            int g = j / 32, b = j % 32;
            uint8_t code = (uint8_t)(w[i * 128 + j] + 1);
            out[i * 32 + b] |= (uint8_t)(code << (6 - 2 * g));
        }
}

static int fails = 0;

/* Independent scalar reference for the kernel's RAW output: the kernel returns
 * the integer sum of (unsigned) codes * int8 activations. This must match the
 * kernel EXACTLY (no quantization noise involved — same int8 inputs). */
static int64_t ref_raw(const uint8_t *packed, const int8_t *y, int n) {
    int64_t sum = 0;
    const int nb = n / 128;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *px = packed + (size_t)i * 32;
        const int8_t  *py = y + (size_t)i * 128;
        for (int b = 0; b < 32; ++b) {
            const uint8_t w = px[b];
            sum += (int64_t)((w >> 6) & 3) * py[b];
            sum += (int64_t)((w >> 4) & 3) * py[b + 32];
            sum += (int64_t)((w >> 2) & 3) * py[b + 64];
            sum += (int64_t)( w       & 3) * py[b + 96];
        }
    }
    return sum;
}

/* Bit-exact check of the kernel's raw sum against the scalar reference. */
static void check_raw(const char *name, int n, float kernel_raw, int64_t ref) {
    int64_t got = (int64_t)kernel_raw;
    int ok = (got == ref);
    printf("%-14s n=%-5d raw=%-12lld ref=%-12lld %s\n",
           name, n, (long long)got, (long long)ref, ok ? "OK" : "FAIL");
    if (!ok) fails++;
}

/* Loose sanity check: corrected kernel value ~ exact float dot (quantization
 * error grows with n, so tolerance scales with n). */
static void check_val(const char *name, int n, double got, double ref) {
    double tol = 0.05 * fabs(ref) + 0.5 + 0.002 * n;
    double err = fabs(got - ref);
    printf("%-14s n=%-5d got=%-12.3f ref=%-12.3f %s\n",
           name, n, got, ref, err <= tol ? "OK" : "FAIL");
    if (err > tol) fails++;
}

int main(void) {
    srand(20240722);
    const int dims[] = {128, 256, 2560, 6912};

    for (int di = 0; di < 4; ++di) {
        const int n = dims[di];

        int8_t  *w      = malloc((size_t)n);
        uint8_t *packed = malloc((size_t)n / 4);
        float   *act    = malloc((size_t)n * sizeof(float));
        int8_t  *y      = malloc((size_t)n);          /* one activation row */
        float   *deq    = malloc((size_t)n * sizeof(float));

        for (int i = 0; i < n; ++i) w[i] = (int8_t)((rand() % 3) - 1);
        for (int i = 0; i < n; ++i) act[i] = (float)((rand() % 2001) - 1000) / 250.0f;
        pack_row(w, packed, n);

        /* cross-check packing: dequantize (scale 1) must reproduce w */
        dequantize_row_i2_s(packed, deq, n, 1.0f);
        int deq_ok = 1;
        for (int i = 0; i < n; ++i) if ((int)deq[i] != (int)w[i]) { deq_ok = 0; break; }
        printf("dequant-check  n=%-5d %s\n", n, deq_ok ? "OK" : "FAIL");
        if (!deq_ok) fails++;

        float   act_scale = 0;
        int32_t act_sum   = 0;
        quantize_row_i8_s(act, y, n, &act_scale, &act_sum);

        double ref = 0;
        for (int i = 0; i < n; ++i) ref += (double)w[i] * (double)act[i];
        int64_t raw = ref_raw(packed, y, n);

        /* --- _1x1 path: nrc=1, one weight row, output s[0] --- */
        float s1 = 0;
        ggml_vec_dot_i2_i8_s(n, &s1, 1, packed, (size_t)n, y, 0, 1);
        check_raw("_1x1", n, s1, raw);
        check_val("_1x1-val", n, (double)(s1 - act_sum) / act_scale, ref);

        /* --- _Nx1 path: nrc=4, one weight row, 4 activation columns --- */
        /* Build 4 activation rows; column c lives at y4 + c*n. Reuse the same
         * quantized row in all 4 columns so every output must equal raw. */
        const int NC = 4;
        int8_t *y4 = malloc((size_t)n * NC);
        for (int c = 0; c < NC; ++c) memcpy(y4 + (size_t)c * n, y, (size_t)n);
        float s4[4] = {0, 0, 0, 0};
        ggml_vec_dot_i2_i8_s(n, s4, 1, packed, (size_t)n, y4, (size_t)n, NC);
        for (int c = 0; c < NC; ++c) {
            char lbl[16];
            snprintf(lbl, sizeof(lbl), "_Nx1[c=%d]", c);
            check_raw(lbl, n, s4[c], raw);
        }

        free(w); free(packed); free(act); free(y); free(deq); free(y4);
    }

    printf(fails ? "\nRESULT: %d FAILURES\n" : "\nRESULT: ALL OK\n", fails);
    return fails ? 1 : 0;
}

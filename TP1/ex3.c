#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#define N 1000

float *alloc_matrix(int n) {
    size_t elems = (size_t)n * (size_t)n;
    float *p = malloc(elems * sizeof(float));
    if (!p) {
        fprintf(stderr, "malloc failed for %zu bytes\n", elems * sizeof(float));
        exit(1);
    }
    return p;
}

void init_matrix(float *M, int n) {
    for (size_t i = 0; i < (size_t)n * n; i++)
        M[i] = (float)(rand() % 10);
}

#define IDX(i, j, n) ((size_t)(i) * (size_t)(n) + (size_t)(j))

// Returns checksum of C so result cannot be optimized away.
double multiply_blocked(float *A, float *B, int n, int BS) {
    size_t elems = (size_t)n * n;
    float *C = calloc(elems, sizeof(float));
    if (!C) { fprintf(stderr, "calloc failed\n"); exit(1); }

    for (int ii = 0; ii < n; ii += BS)
        for (int kk = 0; kk < n; kk += BS)
            for (int jj = 0; jj < n; jj += BS)

                for (int i = ii; i < ii + BS && i < n; i++)
                    for (int k = kk; k < kk + BS && k < n; k++) {
                        float a = A[IDX(i, k, n)];
                        for (int j = jj; j < jj + BS && j < n; j++)
                            C[IDX(i, j, n)] += a * B[IDX(k, j, n)];
                    }

    double sum = 0.0;
    for (size_t i = 0; i < elems; i++) sum += C[i];

    free(C);
    return sum;
}

static double timespec_to_sec(const struct timespec *t) {
    return (double)t->tv_sec + (double)t->tv_nsec / 1e9;
}

int main() {
    int block_sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 9000};
    int num_blocks = sizeof(block_sizes) / sizeof(block_sizes[0]);

    srand((unsigned)time(NULL));

    float *A = alloc_matrix(N);
    float *B = alloc_matrix(N);

    init_matrix(A, N);
    init_matrix(B, N);

    printf("block_size,time(ms),gflops\n");

    for (int b = 0; b < num_blocks; b++) {
        int BS = block_sizes[b];

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        double checksum = multiply_blocked(A, B, N, BS);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double time_sec = timespec_to_sec(&t1) - timespec_to_sec(&t0);
        double time_ms  = time_sec * 1000.0;

        // flops for naive matmul: 2 * n^3 (one mul + one add per innermost iteration)
        double flops = 2.0 * (double)N * (double)N * (double)N;
        double gflops = (flops / time_sec) / 1e9;

        printf("%d,%.3f,%.3f\n", BS, time_ms, gflops);
    }

    free(A);
    free(B);
    return 0;
}

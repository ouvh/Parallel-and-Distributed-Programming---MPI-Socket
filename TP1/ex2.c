#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float **alloc_matrix(int n) {
    float **M = malloc(n * sizeof *M);
    for (int i = 0; i < n; i++)
        M[i] = malloc(n * sizeof *M[i]);
    return M;
}

void free_matrix(float **M, int n) {
    for (int i = 0; i < n; i++)
        free(M[i]);
    free(M);
}

void init_matrix(float **M, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = (float)(rand() % 10);
}

void multiply(float **A, float **B, float **C, int n) {

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0f;
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }

}

void multiply_optimize(float **A, float **B, float **C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0.0f;

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i][j] += A[i][k] * B[k][j];

}

int main() {
    int sizes[] = {2, 4, 5, 10, 25, 100, 500, 1000};
    int count = 8;

    srand(time(NULL));
    printf("matrix_size,time(ms),rate(MB/s)\n");

    for (int i = 0; i < count; i++) {
        int n = sizes[i];

        float **A = alloc_matrix(n);
        float **B = alloc_matrix(n);
        float **C = alloc_matrix(n);

        init_matrix(A, n);
        init_matrix(B, n);

        clock_t start = clock();
        multiply_optimize(A, B, C, n);   
        clock_t end = clock();

        double time_sec = (double)(end - start) / CLOCKS_PER_SEC;
        double time_ms  = time_sec * 1000.0;

        double bytes = 4.0 * n * n * n * sizeof(float);
        double rate = (bytes / time_sec) / (1024.0 * 1024.0);

        printf("%d,%.3f,%.2f\n", n, time_ms, rate);

        free_matrix(A, n);
        free_matrix(B, n);
        free_matrix(C, n);

    }

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef M_SIZE
#define M_SIZE 1000
#endif

#ifndef N_SIZE
#define N_SIZE 1000
#endif

int main() {
    int m = M_SIZE;
    int n = N_SIZE;
    
    // Allocate memory dynamically
    double *a = (double *)malloc(m * n * sizeof(double));
    double *b = (double *)malloc(n * m * sizeof(double));
    double *c = (double *)malloc(m * m * sizeof(double));
    
    if (!a || !b || !c) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize matrices
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = (i + 1) + (j + 1);
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[i * m + j] = (i + 1) - (j + 1);
        }
    }
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            c[i * m + j] = 0;
        }
    }
    
    double start_time = omp_get_wtime();
    
    // Matrix multiplication with OpenMP collapse directive
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < n; k++) {
                c[i * m + j] += a[i * n + k] * b[k * m + j];
            }
        }
    }
    
    double end_time = omp_get_wtime();
    
    printf("Matrix size: %d x %d\n", m, n);
    printf("Number of threads: %d\n", omp_get_max_threads());
    printf("Execution time: %f seconds\n", end_time - start_time);
    
    free(a);
    free(b);
    free(c);
    
    return 0;
}

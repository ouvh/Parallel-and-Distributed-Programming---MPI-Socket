#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#ifndef M_SIZE
#define M_SIZE 1000
#endif

#ifndef N_SIZE
#define N_SIZE 1000
#endif

void run_matrix_mult(char* schedule_type, int chunk_size) {
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
    
    // Matrix multiplication with different schedule types
    if (strcmp(schedule_type, "static") == 0) {
        if (chunk_size > 0) {
            #pragma omp parallel for collapse(2) schedule(static, chunk_size)
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    for (int k = 0; k < n; k++) {
                        c[i * m + j] += a[i * n + k] * b[k * m + j];
                    }
                }
            }
        } else {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    for (int k = 0; k < n; k++) {
                        c[i * m + j] += a[i * n + k] * b[k * m + j];
                    }
                }
            }
        }
    } else if (strcmp(schedule_type, "dynamic") == 0) {
        if (chunk_size > 0) {
            #pragma omp parallel for collapse(2) schedule(dynamic, chunk_size)
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    for (int k = 0; k < n; k++) {
                        c[i * m + j] += a[i * n + k] * b[k * m + j];
                    }
                }
            }
        } else {
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    for (int k = 0; k < n; k++) {
                        c[i * m + j] += a[i * n + k] * b[k * m + j];
                    }
                }
            }
        }
    } else if (strcmp(schedule_type, "guided") == 0) {
        if (chunk_size > 0) {
            #pragma omp parallel for collapse(2) schedule(guided, chunk_size)
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    for (int k = 0; k < n; k++) {
                        c[i * m + j] += a[i * n + k] * b[k * m + j];
                    }
                }
            }
        } else {
            #pragma omp parallel for collapse(2) schedule(guided)
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    for (int k = 0; k < n; k++) {
                        c[i * m + j] += a[i * n + k] * b[k * m + j];
                    }
                }
            }
        }
    }
    
    double end_time = omp_get_wtime();
    
    printf("%s,%d,%d,%f\n", schedule_type, chunk_size, omp_get_max_threads(), end_time - start_time);
    
    free(a);
    free(b);
    free(c);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <schedule_type> [chunk_size]\n", argv[0]);
        fprintf(stderr, "schedule_type: static, dynamic, or guided\n");
        fprintf(stderr, "chunk_size: optional chunk size (default: 0 = auto)\n");
        return 1;
    }
    
    char *schedule_type = argv[1];
    int chunk_size = (argc >= 3) ? atoi(argv[2]) : 0;
    
    run_matrix_mult(schedule_type, chunk_size);
    
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 1000

void init_matrix(int n, double *A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i*n + j] = (double)(i + j);
        }
    }
}

void print_matrix(int n, double *A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.1f ", A[i*n + j]);
        }
        printf("\n");
    }
}

double sum_matrix(int n, double *A) {
    double sum = 0.0;
    for (int i = 0; i < n*n; i++) {
        sum += A[i];
    }
    return sum;
}

int main() {
    double *A;
    double sum;
    double start, end;

    A = (double*) malloc(N * N * sizeof(double));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    printf("=== Serial Version ===\n");
    start = omp_get_wtime();
    
    init_matrix(N, A);
    if (N <= 10) print_matrix(N, A);
    sum = sum_matrix(N, A);
    
    end = omp_get_wtime();
    
    printf("Sum = %lf\n", sum);
    printf("Execution time = %lf seconds\n\n", end - start);

    // Reset matrix
    for (int i = 0; i < N*N; i++) A[i] = 0.0;

    printf("=== Parallel Version (Master for init, Single for print) ===\n");
    start = omp_get_wtime();

    #pragma omp parallel
    {
        // Master thread initializes matrix
        #pragma omp master
        {
            printf("[Thread %d] Master: Initializing matrix\n", omp_get_thread_num());
            init_matrix(N, A);
        }

        // Implicit barrier here ensures matrix is initialized before printing
        
        // Single thread prints matrix
        #pragma omp single
        {
            printf("[Thread %d] Single: Printing matrix\n", omp_get_thread_num());
            if (N <= 10) print_matrix(N, A);
        }

        // All threads compute sum in parallel
        #pragma omp for reduction(+:sum)
        for (int i = 0; i < N*N; i++) {
            sum += A[i];
        }

        #pragma omp single
        {
            printf("[Thread %d] Computing sum with %d threads\n", 
                   omp_get_thread_num(), omp_get_num_threads());
        }
    }

    end = omp_get_wtime();

    printf("Sum = %lf\n", sum);
    printf("Execution time = %lf seconds\n\n", end - start);

    // Alternative: Using single with nowait
    for (int i = 0; i < N*N; i++) A[i] = 0.0;
    
    printf("=== Parallel Version (All Single with nowait) ===\n");
    start = omp_get_wtime();

    sum = 0.0;
    #pragma omp parallel
    {
        // Single thread initializes (with nowait - no barrier)
        #pragma omp single nowait
        {
            printf("[Thread %d] Single-nowait: Initializing matrix\n", omp_get_thread_num());
            init_matrix(N, A);
        }

        // Barrier needed before accessing matrix
        #pragma omp barrier

        // Single thread prints
        #pragma omp single nowait
        {
            printf("[Thread %d] Single-nowait: Printing matrix\n", omp_get_thread_num());
            if (N <= 10) print_matrix(N, A);
        }

        // All threads compute sum
        #pragma omp for reduction(+:sum)
        for (int i = 0; i < N*N; i++) {
            sum += A[i];
        }

        #pragma omp single
        {
        }
    }

    end = omp_get_wtime();

    printf("Sum = %lf\n", sum);
    printf("Execution time = %lf seconds\n", end - start);

    free(A);
    return 0;
}

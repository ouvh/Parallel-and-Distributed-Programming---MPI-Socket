#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Version 1: With implicit barrier
void dmvm_v1(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel for
    for (int c = 0; c < n; ++c) {
        int offset = m * c;
        for (int r = 0; r < m; ++r)
            lhs[r] += mat[r + offset] * rhs[c];
    }
    // Implicit barrier at end of parallel for
}

// Version 2: Dynamic scheduling with nowait
void dmvm_v2(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic) nowait
        for (int c = 0; c < n; ++c) {
            int offset = m * c;
            for (int r = 0; r < m; ++r)
                lhs[r] += mat[r + offset] * rhs[c];
        }
        // No barrier! Can be dangerous if lhs is used immediately after
    }
}

// Version 3: Static scheduling with nowait
void dmvm_v3(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (int c = 0; c < n; ++c) {
            int offset = m * c;
            for (int r = 0; r < m; ++r)
                lhs[r] += mat[r + offset] * rhs[c];
        }
    }
}

// Version 4: With atomic operations (safe but slow)
void dmvm_v4(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < n; ++c) {
        int offset = m * c;
        for (int r = 0; r < m; ++r) {
            #pragma omp atomic
            lhs[r] += mat[r + offset] * rhs[c];
        }
    }
}

// Version 5: Thread-local accumulation (most efficient)
void dmvm_v5(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        
        // Thread-local buffer
        double *local_lhs = (double*)calloc(m, sizeof(double));
        
        #pragma omp for schedule(static) nowait
        for (int c = 0; c < n; ++c) {
            int offset = m * c;
            for (int r = 0; r < m; ++r)
                local_lhs[r] += mat[r + offset] * rhs[c];
        }
        
        // Reduce local results to global
        #pragma omp critical
        {
            for (int r = 0; r < m; ++r)
                lhs[r] += local_lhs[r];
        }
        
        free(local_lhs);
    }
}

int main() {
    const int n = 40000; // columns
    const int m = 600;    // rows
    
    // Calculate FLOPs: 2*n*m (one multiply and one add per element)
    long long flops = 2LL * n * m;

    double *mat = malloc(n * m * sizeof(double));
    double *rhs = malloc(n * sizeof(double));
    double *lhs = malloc(m * sizeof(double));

    if (!mat || !rhs || !lhs) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialization
    for (int c = 0; c < n; ++c) {
        rhs[c] = 1.0;
        for (int r = 0; r < m; ++r)
            mat[r + c*m] = 1.0;
    }

    printf("Matrix-Vector Multiplication: %d x %d\n", m, n);
    printf("Total FLOPs: %lld\n\n", flops);

    // Test different versions with different thread counts
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_tests = sizeof(thread_counts) / sizeof(int);

    printf("Version,Threads,Time,Speedup,Efficiency,MFLOPS\n");

    double serial_time = 0.0;

    for (int t = 0; t < num_tests; t++) {
        int threads = thread_counts[t];
        omp_set_num_threads(threads);

        // Version 1: Implicit barrier
        for (int r = 0; r < m; ++r) lhs[r] = 0.0;
        double start = omp_get_wtime();
        dmvm_v1(n, m, lhs, rhs, mat);
        double end = omp_get_wtime();
        double time_v1 = end - start;
        
        if (threads == 1) serial_time = time_v1;
        double speedup = serial_time / time_v1;
        double efficiency = (speedup / threads) * 100.0;
        double mflops = (flops / time_v1) / 1e6;
        
        printf("V1_Barrier,%d,%.6f,%.2f,%.2f,%.2f\n", 
               threads, time_v1, speedup, efficiency, mflops);

        // Version 3: Static nowait
        for (int r = 0; r < m; ++r) lhs[r] = 0.0;
        start = omp_get_wtime();
        dmvm_v3(n, m, lhs, rhs, mat);
        #pragma omp barrier  // Ensure completion before measuring time
        end = omp_get_wtime();
        double time_v3 = end - start;
        
        speedup = serial_time / time_v3;
        efficiency = (speedup / threads) * 100.0;
        mflops = (flops / time_v3) / 1e6;
        
        printf("V3_Static_Nowait,%d,%.6f,%.2f,%.2f,%.2f\n", 
               threads, time_v3, speedup, efficiency, mflops);

        // Version 5: Thread-local (best performance)
        for (int r = 0; r < m; ++r) lhs[r] = 0.0;
        start = omp_get_wtime();
        dmvm_v5(n, m, lhs, rhs, mat);
        end = omp_get_wtime();
        double time_v5 = end - start;
        
        speedup = serial_time / time_v5;
        efficiency = (speedup / threads) * 100.0;
        mflops = (flops / time_v5) / 1e6;
        
        printf("V5_ThreadLocal,%d,%.6f,%.2f,%.2f,%.2f\n", 
               threads, time_v5, speedup, efficiency, mflops);
    }

    // Verify result
    printf("\nVerification (first 10 elements):\n");
    for (int r = 0; r < 10; ++r)
        printf("lhs[%d] = %.1f\n", r, lhs[r]);

    free(mat);
    free(rhs);
    free(lhs);
    return 0;
}

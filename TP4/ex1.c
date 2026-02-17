#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N 1000000

int main() {
    double *A = malloc(N * sizeof(double));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    double sum = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double max = 0.0;
    double start_time, end_time;

    /* Synchronization flag: 0 => sum not ready, 1 => sum ready */
    int sum_ready = 0;

    srand(0);
    for (int i = 0; i < N; i++)
        A[i] = (double)rand() / RAND_MAX;

    start_time = omp_get_wtime();

    /* parallel sections: each section may be executed by a different thread */
    #pragma omp parallel sections
    {
        /* Section 1: Compute sum */
        #pragma omp section
        {
            double local_sum = 0.0;
            for (int i = 0; i < N; i++) {
                local_sum += A[i];
            }

            /* publish sum then set the ready flag atomically and flush to ensure visibility */
            sum = local_sum;

            #pragma omp atomic write
            sum_ready = 1;

            /* ensure both sum and sum_ready are visible to other threads */
            #pragma omp flush(sum, sum_ready)

            printf("[Thread %d] Section 1: Computing sum\n", omp_get_thread_num());
        }

        /* Section 2: Compute max */
        #pragma omp section
        {
            double local_max = A[0];
            for (int i = 1; i < N; i++) {
                if (A[i] > local_max)
                    local_max = A[i];
            }
            max = local_max;
            printf("[Thread %d] Section 2: Computing max\n", omp_get_thread_num());
        }

        /* Section 3: Compute stddev using sum computed by Section 1 (no recompute) */
        #pragma omp section
        {
            /* wait until Section 1 sets sum_ready == 1 */
            int ready = 0;
            do {
                #pragma omp atomic read
                ready = sum_ready;
                /* small pause could be added here if desired to reduce busy-spin */
            } while (!ready);


            double local_mean = sum / N;
            double local_stddev = 0.0;
            for (int i = 0; i < N; i++) {
                double d = A[i] - local_mean;
                local_stddev += d * d;
            }
            mean = local_mean;
            stddev = sqrt(local_stddev / N);

            printf("[Thread %d] Section 3: Computing std dev\n", omp_get_thread_num());
        }
    } /* end parallel sections */

    end_time = omp_get_wtime();

    /* Print results */
    printf("\nResults:\n");
    printf("Sum     = %f\n", sum);
    printf("Mean    = %f\n", mean);
    printf("Max     = %f\n", max);
    printf("Std Dev = %f\n", stddev);
    printf("Execution time = %f seconds\n", end_time - start_time);
    printf("Number of threads = %d\n", omp_get_max_threads());

    free(A);
    return 0;
}

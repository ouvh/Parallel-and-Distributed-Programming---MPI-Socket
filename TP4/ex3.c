#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void task_light(int N) {
    double x = 0.0;
    for (int i = 0; i < N; i++) {
        x += sin(i * 0.001);
    }
    // Prevent optimization
    if (x < -1000) printf("%f\n", x);
}

void task_moderate(int N) {
    double x = 0.0;
    for (int i = 0; i < 5*N; i++) {
        x += sqrt(i * 0.5) * cos(i * 0.001);
    }
    if (x < -1000) printf("%f\n", x);
}

void task_heavy(int N) {
    double x = 0.0;
    for (int i = 0; i < 20*N; i++) {
        x += sqrt(i * 0.5) * cos(i * 0.001) * sin(i * 0.0001);
    }
    if (x < -1000) printf("%f\n", x);
}

int main() {
    int N = 1000000;
    double start, end;

    printf("=== Version 1: Serial Execution ===\n");
    start = omp_get_wtime();
    task_light(N);
    task_moderate(N);
    task_heavy(N);
    end = omp_get_wtime();
    printf("Execution time: %f seconds\n\n", end - start);

    printf("=== Version 2: Parallel Sections (Unbalanced) ===\n");
    start = omp_get_wtime();
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            double t1 = omp_get_wtime();
            task_light(N);
            double t2 = omp_get_wtime();
            printf("[Thread %d] Task Light: %f seconds\n", omp_get_thread_num(), t2-t1);
        }

        #pragma omp section
        {
            double t1 = omp_get_wtime();
            task_moderate(N);
            double t2 = omp_get_wtime();
            printf("[Thread %d] Task Moderate: %f seconds\n", omp_get_thread_num(), t2-t1);
        }

        #pragma omp section
        {
            double t1 = omp_get_wtime();
            task_heavy(N);
            double t2 = omp_get_wtime();
            printf("[Thread %d] Task Heavy: %f seconds\n", omp_get_thread_num(), t2-t1);
        }
    }
    
    end = omp_get_wtime();
    printf("Total parallel time: %f seconds\n", end - start);
    printf("Note: Load imbalance! Heavy task dominates.\n\n");

    printf("=== Version 3: Balanced with Task Parallelism ===\n");
    start = omp_get_wtime();
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Create tasks that can be scheduled dynamically
            #pragma omp task
            {
                double t1 = omp_get_wtime();
                task_light(N);
                double t2 = omp_get_wtime();
                printf("[Thread %d] Task Light: %f seconds\n", omp_get_thread_num(), t2-t1);
            }

            #pragma omp task
            {
                double t1 = omp_get_wtime();
                task_moderate(N);
                double t2 = omp_get_wtime();
                printf("[Thread %d] Task Moderate: %f seconds\n", omp_get_thread_num(), t2-t1);
            }

            #pragma omp task
            {
                double t1 = omp_get_wtime();
                task_heavy(N);
                double t2 = omp_get_wtime();
                printf("[Thread %d] Task Heavy: %f seconds\n", omp_get_thread_num(), t2-t1);
            }
        }
    }
    
    end = omp_get_wtime();
    printf("Total parallel time: %f seconds\n\n", end - start);

    printf("=== Version 4: Multiple Heavy Tasks Split ===\n");
    // Split heavy task into smaller chunks for better load balancing
    start = omp_get_wtime();
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            double t1 = omp_get_wtime();
            task_light(N);
            task_moderate(N);  // Light thread helps with moderate
            double t2 = omp_get_wtime();
            printf("[Thread %d] Light + Moderate: %f seconds\n", omp_get_thread_num(), t2-t1);
        }

        #pragma omp section
        {
            double t1 = omp_get_wtime();
            task_heavy(N);
            double t2 = omp_get_wtime();
            printf("[Thread %d] Heavy: %f seconds\n", omp_get_thread_num(), t2-t1);
        }
    }
    
    end = omp_get_wtime();
    printf("Total parallel time: %f seconds\n", end - start);
    printf("Note: Better balance by combining light and moderate tasks.\n\n");

    printf("=== Version 5: All with Dynamic For Loop ===\n");
    // Convert to loop-based parallelism for automatic load balancing
    start = omp_get_wtime();
    
    #pragma omp parallel for schedule(dynamic)
    for (int task = 0; task < 3; task++) {
        double t1 = omp_get_wtime();
        if (task == 0) {
            task_light(N);
            double t2 = omp_get_wtime();
            printf("[Thread %d] Task Light: %f seconds\n", omp_get_thread_num(), t2-t1);
        } else if (task == 1) {
            task_moderate(N);
            double t2 = omp_get_wtime();
            printf("[Thread %d] Task Moderate: %f seconds\n", omp_get_thread_num(), t2-t1);
        } else {
            task_heavy(N);
            double t2 = omp_get_wtime();
            printf("[Thread %d] Task Heavy: %f seconds\n", omp_get_thread_num(), t2-t1);
        }
    }
    
    end = omp_get_wtime();
    printf("Total parallel time: %f seconds\n", end - start);

    return 0;
}

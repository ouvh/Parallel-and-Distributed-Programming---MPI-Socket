#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;

int main() {
    double x, pi, sum = 0.0;
    double start_time, end_time;
    
    step = 1.0 / (double) num_steps;
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        double partial_sum = 0.0;
        
        for (int i = id; i < num_steps; i += nthreads) {
            x = (i + 0.5) * step;
            partial_sum += 4.0 / (1.0 + x * x);
        }
        
        #pragma omp critical
        sum += partial_sum;
    }
    
    pi = step * sum;
    
    end_time = omp_get_wtime();
    
    printf("PI = %.15f\n", pi);
    printf("Execution time: %f seconds\n", end_time - start_time);
    
    return 0;
}

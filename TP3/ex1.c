#include <stdio.h>
#include <omp.h>

int main(){
    int count = 0;

    #pragma omp parallel 
    {
        int rank = omp_get_thread_num();
        printf("Hello from the rank %d thread\n", rank);
        
        #pragma omp single
        {
            count = omp_get_num_threads();
        }
    }

    printf("Parallel execution of hello_world with %d threads\n", count);

    return 0;
}
/*
 * Exercise 1: Hello World with MPI
 * 
 * Questions:
 * 1. Print "Hello World"
 * 2. Each process prints its rank and total number of processes
 * 3. Only rank 0 prints a message
 * 4. What happens if MPI_Finalize is omitted?
 */

#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Get process rank and total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Question 1: Simple Hello World
    printf("Hello World from process %d\n", rank);
    
    // Question 2: Print rank and size
    printf("I am process %d of %d total processes\n", rank, size);
    
    // Question 3: Only rank 0 prints
    if (rank == 0) {
        printf("\n=== Master Process Message ===\n");
        printf("This message is printed by rank 0 only\n");
        printf("Total number of MPI processes: %d\n", size);
        printf("==============================\n\n");
    }
    
    // Question 4: What happens if MPI_Finalize is omitted?
    // Answer: The program may hang, leave resources uncleaned, or produce warnings.
    // MPI_Finalize is essential for proper cleanup of MPI resources.
    // Some implementations may not terminate properly without it.
    
    MPI_Finalize();
    
    return 0;
}

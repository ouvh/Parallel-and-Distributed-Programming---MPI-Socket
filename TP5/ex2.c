/*
 * Exercise 2: Sharing Data
 * 
 * Objective:
 * - Read integer from terminal (rank 0)
 * - Distribute to all processes
 * - Each process prints rank and received value
 * - Continue until negative integer is entered
 */

#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int value;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    while (1) {
        // Rank 0 reads the value
        if (rank == 0) {
            printf("\nEnter an integer (negative to quit): ");
            fflush(stdout);
            scanf("%d", &value);
        }
        
        // Broadcast the value to all processes
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Check for termination condition
        if (value < 0) {
            if (rank == 0) {
                printf("Negative value received. Exiting...\n");
            }
            break;
        }
        
        // Each process prints the received value
        printf("Process %d got %d\n", rank, value);
        fflush(stdout);
        
        // Small delay to ensure ordered output
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}

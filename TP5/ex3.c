/*
 * Exercise 3: Sending in a Ring (Broadcast by Ring)
 * 
 * Objective:
 * - Process 0 reads a value
 * - Sends to process 1, which adds its rank and forwards to process 2
 * - Each process receives, adds its rank, prints, and forwards
 * - Last process doesn't forward
 */

#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int value = 0;
    int received_value;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        // Process 0 reads the initial value
        printf("Enter an integer value: ");
        fflush(stdout);
        scanf("%d", &value);
        
        printf("Process %d: Initial value = %d\n", rank, value);
        
        // Add rank and send to next process
        value += rank;
        
        if (size > 1) {
            MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            printf("Process %d: Sent value %d to process 1\n", rank, value);
        } else {
            printf("Process %d: Final accumulated value = %d\n", rank, value);
        }
    } else {
        // Receive from previous process
        MPI_Recv(&received_value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        printf("Process %d: Received value %d from process %d\n", 
               rank, received_value, rank - 1);
        
        // Add this process's rank
        value = received_value + rank;
        printf("Process %d: After adding rank, value = %d\n", rank, value);
        
        // Forward to next process if not the last
        if (rank < size - 1) {
            MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            printf("Process %d: Sent value %d to process %d\n", rank, value, rank + 1);
        } else {
            printf("Process %d: Final accumulated value = %d\n", rank, value);
        }
    }
    
    MPI_Finalize();
    return 0;
}

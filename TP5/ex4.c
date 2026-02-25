/*
 * Exercise 4: Matrix-Vector Product with MPI
 * 
 * Objective:
 * 1. Implement MPI version of matrix-vector multiplication
 * 2. Plot Speedup/Efficiency
 * 3. Handle case when N is not divisible by number of processes
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void matrixVectorMult(double* A, double* b, double* x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] = 0.0;
        for (int j = 0; j < size; ++j) {
            x[i] += A[i * size + j] * b[j];
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    double start_time, end_time, elapsed_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    int N = atoi(argv[1]);
    if (N <= 0) {
        if (rank == 0) {
            printf("Matrix size must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Calculate rows per process (handle non-divisible case)
    int rows_per_proc = N / size;
    int remainder = N % size;
    
    // Processes with rank < remainder get one extra row
    int local_rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;
    int start_row = rank * rows_per_proc + ((rank < remainder) ? rank : remainder);
    
    // Arrays for the full problem (only on rank 0)
    double *A = NULL;
    double *b = NULL;
    double *x_parallel = NULL;
    double *x_serial = NULL;
    
    // Local arrays for each process
    double *local_A = malloc(local_rows * N * sizeof(double));
    double *local_x = malloc(local_rows * sizeof(double));
    
    // Vector b needed by all processes
    b = malloc(N * sizeof(double));
    
    // Rank 0 initializes the data
    if (rank == 0) {
        A = malloc(N * N * sizeof(double));
        x_parallel = malloc(N * sizeof(double));
        x_serial = malloc(N * sizeof(double));
        
        if (!A || !b || !x_parallel || !x_serial) {
            printf("Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        srand48(42);
        
        // Fill A[0][:100] with random values
        int limit = (N < 100) ? N : 100;
        for (int j = 0; j < limit; ++j)
            A[0 * N + j] = drand48();
        
        // Copy A[0][:100] into A[1][100:200] if possible
        if (N > 1 && N > 100) {
            int copy_len = (N - 100 < 100) ? (N - 100) : 100;
            for (int j = 0; j < copy_len; ++j)
                A[1 * N + (100 + j)] = A[0 * N + j];
        }
        
        // Set diagonal
        for (int i = 0; i < N; ++i)
            A[i * N + i] = drand48();
        
        // Fill vector b
        for (int i = 0; i < N; ++i)
            b[i] = drand48();
        
        // Compute serial version for verification
        printf("Computing serial version...\n");
        double serial_start = MPI_Wtime();
        matrixVectorMult(A, b, x_serial, N);
        double serial_time = MPI_Wtime() - serial_start;
        printf("Serial time: %.6f seconds\n", serial_time);
    }
    
    // Broadcast vector b to all processes
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Prepare send counts and displacements for Scatterv
    int *sendcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        
        for (int i = 0; i < size; i++) {
            int proc_rows = (i < remainder) ? rows_per_proc + 1 : rows_per_proc;
            sendcounts[i] = proc_rows * N;
            displs[i] = (i * rows_per_proc + ((i < remainder) ? i : remainder)) * N;
        }
    }
    
    // Start timing parallel computation
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Scatter matrix rows to all processes
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Each process computes its portion of the result
    for (int i = 0; i < local_rows; i++) {
        local_x[i] = 0.0;
        for (int j = 0; j < N; j++) {
            local_x[i] += local_A[i * N + j] * b[j];
        }
    }
    
    // Prepare receive counts and displacements for Gatherv
    int *recvcounts = NULL;
    int *recvdispls = NULL;
    
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        recvdispls = malloc(size * sizeof(int));
        
        for (int i = 0; i < size; i++) {
            recvcounts[i] = (i < remainder) ? rows_per_proc + 1 : rows_per_proc;
            recvdispls[i] = i * rows_per_proc + ((i < remainder) ? i : remainder);
        }
    }
    
    // Gather results back to rank 0
    MPI_Gatherv(local_x, local_rows, MPI_DOUBLE,
                x_parallel, recvcounts, recvdispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    // Rank 0 verifies and reports results
    if (rank == 0) {
        // Compare parallel and serial results
        double max_error = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = fabs(x_parallel[i] - x_serial[i]);
            if (diff > max_error)
                max_error = diff;
        }
        
        printf("\nParallel computation complete!\n");
        printf("Matrix size: %d x %d\n", N, N);
        printf("Number of processes: %d\n", size);
        printf("Parallel time: %.6f seconds\n", elapsed_time);
        printf("Maximum error: %e\n", max_error);
        
        if (max_error < 1e-10) {
            printf("✓ Verification PASSED\n");
        } else {
            printf("✗ Verification FAILED\n");
        }
        
        free(A);
        free(x_parallel);
        free(x_serial);
        free(sendcounts);
        free(displs);
        free(recvcounts);
        free(recvdispls);
    }
    
    free(local_A);
    free(local_x);
    free(b);
    
    MPI_Finalize();
    return 0;
}

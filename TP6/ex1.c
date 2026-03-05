#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 4
#define COLS 5

void print_matrix_4x5(double a[ROWS][COLS], const char* label) {
    printf("%s\n", label);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%6.1f ", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrix_5x4(double at[COLS][ROWS], const char* label) {
    printf("%s\n", label);
    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < ROWS; j++) {
            printf("%6.1f ", at[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    int rank, size;
    double a[ROWS][COLS];     
    double at[COLS][ROWS];     
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) {
        if (rank == 0) {
            printf("This program requires at least 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {

        int count = 1;
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                a[i][j] = count++;
            }
        }
        
        printf("Process 0 - Matrix a:\n");
        print_matrix_4x5(a, "");
        
        MPI_Send(&a[0][0], ROWS * COLS, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        
        printf("Process 0: Matrix sent to process 1\n\n");
    }
    else if (rank == 1) {
      
        
    
        MPI_Datatype column_type;
        MPI_Type_vector(COLS,           // count: number of blocks
                       1,               // blocklength: elements per block
                       ROWS,            // stride: spacing between blocks (in elements)
                       MPI_DOUBLE,      // old type
                       &column_type);   // new type
        
        // Step 2: Create a hvector type that represents all columns
        // We have COLS columns, each separated by sizeof(double) bytes
        MPI_Datatype transpose_type;
        MPI_Type_create_hvector(ROWS,               // count: number of columns
                               1,                   // blocklength: columns per block
                               sizeof(double),      // stride: bytes between columns
                               column_type,         // old type
                               &transpose_type);    // new type
        
        // Commit the derived type
        MPI_Type_commit(&transpose_type);
        
        // Receive the matrix using the transpose type
        MPI_Datatype new_type;
        MPI_Type_create_resized(column_type,0,sizeof(double),&new_type);
        MPI_Type_commit(&new_type);

        MPI_Recv(&at[0][0], ROWS, new_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Display transposed matrix
        printf("Process 1 - Matrix transpose at:\n");
        print_matrix_5x4(at, "");
        
        // Free the derived types
        MPI_Type_free(&transpose_type);
        MPI_Type_free(&column_type);
        
    }
    
    MPI_Finalize();
    return 0;
}

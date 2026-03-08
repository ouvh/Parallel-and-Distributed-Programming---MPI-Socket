/*
 * Exercise 1: Conway's Game of Life with MPI 2D Cartesian Topology
 * Uses MPI_Cart_create for process organization and ghost cell exchange
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Global grid dimensions
#define GLOBAL_NX 64
#define GLOBAL_NY 64
#define GENERATIONS 100

// Cell states
#define DEAD 0
#define ALIVE 1

// Function prototypes
void initialize_grid(int** grid, int local_nx, int local_ny, int rank, int coord_x, int coord_y);
void exchange_ghost_cells(int** grid, int local_nx, int local_ny, 
                         MPI_Comm cart_comm, int north, int south, int east, int west);
void apply_game_of_life(int** grid, int** new_grid, int local_nx, int local_ny);
int count_neighbors(int** grid, int i, int j);
void print_local_grid(int** grid, int local_nx, int local_ny, int rank, int gen);

int main(int argc, char* argv[]) {
    int rank, size;
    int dims[2] = {0, 0};        // Process grid dimensions (will be computed)
    int periods[2] = {1, 1};     // Periodic boundary conditions
    int reorder = 1;             // Allow rank reordering
    int coord[2];                // Process coordinates in 2D grid
    
    // Neighbor ranks
    int north, south, east, west;
    
    MPI_Comm cart_comm;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double start_time = MPI_Wtime();
    
    // Create 2D Cartesian topology
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    
    // Get new rank in Cartesian communicator
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coord);
    
    // Determine local subdomain size
    int local_nx = GLOBAL_NX / dims[0];
    int local_ny = GLOBAL_NY / dims[1];
    
    // Find neighbors (with periodic boundaries)
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);  // Shift in x-direction
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);    // Shift in y-direction
    
    if (rank == 0) {
        printf("=== Conway's Game of Life - MPI 2D Cartesian Topology ===\n");
        printf("Global grid: %d x %d\n", GLOBAL_NX, GLOBAL_NY);
        printf("Number of processes: %d\n", size);
        printf("Process grid: %d x %d\n", dims[0], dims[1]);
        printf("Local grid per process: %d x %d (plus ghost cells)\n", local_nx, local_ny);
        printf("Generations: %d\n", GENERATIONS);
        printf("Boundary conditions: periodic (wrap-around)\n\n");
    }
    
    // Allocate local grid with ghost cells (halo)
    // Grid is (local_nx + 2) x (local_ny + 2) to include boundaries
    int** grid = (int**)malloc((local_nx + 2) * sizeof(int*));
    int** new_grid = (int**)malloc((local_nx + 2) * sizeof(int*));
    for (int i = 0; i < local_nx + 2; i++) {
        grid[i] = (int*)calloc(local_ny + 2, sizeof(int));
        new_grid[i] = (int*)calloc(local_ny + 2, sizeof(int));
    }
    
    // Initialize grid (only internal cells, not ghost cells)
    initialize_grid(grid, local_nx, local_ny, rank, coord[0], coord[1]);
    
    // Print initial configuration for rank 0 (optional)
    if (rank == 0 && local_nx <= 20) {
        printf("Rank %d - Initial configuration (internal cells):\n", rank);
        for (int i = 1; i <= local_nx; i++) {
            for (int j = 1; j <= local_ny; j++) {
                printf("%c ", grid[i][j] ? 'O' : '.');
            }
            printf("\n");
        }
        printf("\n");
    }
    
    MPI_Barrier(cart_comm);
    double compute_start = MPI_Wtime();
    
    // Main simulation loop
    for (int gen = 0; gen < GENERATIONS; gen++) {
        // Exchange ghost cells with neighbors
        exchange_ghost_cells(grid, local_nx, local_ny, cart_comm, 
                            north, south, east, west);
        
        // Apply Game of Life rules
        apply_game_of_life(grid, new_grid, local_nx, local_ny);
        
        // Swap grids
        
        int** temp = grid;
        grid = new_grid;
        new_grid = temp;
        
        // Optional: Print intermediate generations for debugging
        // if (rank == 0 && gen % 10 == 0 && local_nx <= 20) {
        //     print_local_grid(grid, local_nx, local_ny, rank, gen);
        // }
    }
    
    double compute_end = MPI_Wtime();
    
    // Print final configuration for rank 0 (optional)
    if (rank == 0 && local_nx <= 20) {
        printf("Rank %d - Generation %d (final):\n", rank, GENERATIONS);
        for (int i = 1; i <= local_nx; i++) {
            for (int j = 1; j <= local_ny; j++) {
                printf("%c ", grid[i][j] ? 'O' : '.');
            }
            printf("\n");
        }
        printf("\n");
    }
    
    // Count total alive cells
    int local_alive = 0;
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            if (grid[i][j] == ALIVE) local_alive++;
        }
    }
    
    int global_alive = 0;
    MPI_Reduce(&local_alive, &global_alive, 1, MPI_INT, MPI_SUM, 0, cart_comm);
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Simulation complete!\n");
        printf("Final alive cells: %d / %d (%.2f%%)\n", 
               global_alive, GLOBAL_NX * GLOBAL_NY, 
               100.0 * global_alive / (GLOBAL_NX * GLOBAL_NY));
        printf("Total time: %.6f seconds\n", end_time - start_time);
        printf("Computation time: %.6f seconds\n", compute_end - compute_start);
        printf("Communication overhead: %.6f seconds\n", 
               (end_time - start_time) - (compute_end - compute_start));
    }
    
    // Cleanup
    for (int i = 0; i < local_nx + 2; i++) {
        free(grid[i]);
        free(new_grid[i]);
    }
    free(grid);
    free(new_grid);
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}

void initialize_grid(int** grid, int local_nx, int local_ny, int rank, int coord_x, int coord_y) {
    // Initialize with a known pattern or random
    // Here we use a glider pattern in the center for rank 0's subdomain
    
    // Random initialization
    srand(rank + time(NULL));
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            // 30% probability of being alive
            grid[i][j] = (rand() % 100 < 30) ? ALIVE : DEAD;
        }
    }
    
    // Add a glider pattern if this is the center process
    if (coord_x == 0 && coord_y == 0 && local_nx >= 5 && local_ny >= 5) {
        // Classic glider pattern
        int cx = local_nx / 2;
        int cy = local_ny / 2;
        
        grid[cx][cy] = DEAD;
        grid[cx][cy+1] = ALIVE;
        grid[cx][cy+2] = DEAD;
        
        grid[cx+1][cy] = DEAD;
        grid[cx+1][cy+1] = DEAD;
        grid[cx+1][cy+2] = ALIVE;
        
        grid[cx+2][cy] = ALIVE;
        grid[cx+2][cy+1] = ALIVE;
        grid[cx+2][cy+2] = ALIVE;
    }
}

void exchange_ghost_cells(int** grid, int local_nx, int local_ny,
                         MPI_Comm cart_comm, int north, int south, int east, int west) {
    MPI_Status status;
    
    // Exchange with north and south (rows)
    // Send top row to north, receive from south
    MPI_Sendrecv(&grid[1][1], local_ny, MPI_INT, north, 0,
                 &grid[local_nx+1][1], local_ny, MPI_INT, south, 0,
                 cart_comm, &status);
    
    // Send bottom row to south, receive from north
    MPI_Sendrecv(&grid[local_nx][1], local_ny, MPI_INT, south, 1,
                 &grid[0][1], local_ny, MPI_INT, north, 1,
                 cart_comm, &status);
    
    // For east-west exchange, we need to send columns
    // Create temporary buffers for columns
    int* send_col = (int*)malloc(local_nx * sizeof(int));
    int* recv_col = (int*)malloc(local_nx * sizeof(int));
    
    // Exchange with west and east (columns)
    // Send left column to west, receive from east
    for (int i = 0; i < local_nx; i++) {
        send_col[i] = grid[i+1][1];
    }
    MPI_Sendrecv(send_col, local_nx, MPI_INT, west, 2,
                 recv_col, local_nx, MPI_INT, east, 2,
                 cart_comm, &status);
    for (int i = 0; i < local_nx; i++) {
        grid[i+1][local_ny+1] = recv_col[i];
    }
    
    // Send right column to east, receive from west
    for (int i = 0; i < local_nx; i++) {
        send_col[i] = grid[i+1][local_ny];
    }
    MPI_Sendrecv(send_col, local_nx, MPI_INT, east, 3,
                 recv_col, local_nx, MPI_INT, west, 3,
                 cart_comm, &status);
    for (int i = 0; i < local_nx; i++) {
        grid[i+1][0] = recv_col[i];
    }
    
    // Exchange corners (needed for accurate neighbor counting)
    // Top-left corner
    int corner;
    MPI_Sendrecv(&grid[1][1], 1, MPI_INT, north, 4,
                 &corner, 1, MPI_INT, south, 4, cart_comm, &status);
    MPI_Sendrecv(&corner, 1, MPI_INT, west, 5,
                 &grid[local_nx+1][local_ny+1], 1, MPI_INT, east, 5,
                 cart_comm, &status);
    
    free(send_col);
    free(recv_col);
}

void apply_game_of_life(int** grid, int** new_grid, int local_nx, int local_ny) {
    // Apply Conway's Game of Life rules to internal cells
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            int neighbors = count_neighbors(grid, i, j);
            
            if (grid[i][j] == ALIVE) {
                // Cell is alive
                if (neighbors < 2 || neighbors > 3) {
                    new_grid[i][j] = DEAD;  // Dies
                } else {
                    new_grid[i][j] = ALIVE; // Survives
                }
            } else {
                // Cell is dead
                if (neighbors == 3) {
                    new_grid[i][j] = ALIVE; // Reproduction
                } else {
                    new_grid[i][j] = DEAD;  // Stays dead
                }
            }
        }
    }
}

int count_neighbors(int** grid, int i, int j) {
    // Count the 8 neighbors (including diagonals)
    int count = 0;
    
    count += grid[i-1][j-1]; // Top-left
    count += grid[i-1][j];   // Top
    count += grid[i-1][j+1]; // Top-right
    count += grid[i][j-1];   // Left
    count += grid[i][j+1];   // Right
    count += grid[i+1][j-1]; // Bottom-left
    count += grid[i+1][j];   // Bottom
    count += grid[i+1][j+1]; // Bottom-right
    
    return count;
}

void print_local_grid(int** grid, int local_nx, int local_ny, int rank, int gen) {
    printf("Rank %d - Generation %d:\n", rank, gen);
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            printf("%c ", grid[i][j] ? 'O' : '.');
        }
        printf("\n");
    }
    printf("\n");
}

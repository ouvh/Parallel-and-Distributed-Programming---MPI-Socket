/*
 * Exercise 2: Poisson Equation Solver using Jacobi Iteration
 * with MPI 2D Cartesian Topology
 * 
 * Uses compute.c for initialization and Jacobi iteration
 * Solves: Δu(x,y) = f(x,y) = 2·(x²-x+y²-y) on [0,1]×[0,1]
 * Boundary condition: u = 0 on ∂Ω
 * Exact solution: u(x,y) = x·y·(x-1)·(y-1)
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define TOLERANCE 1.0e-6
#define MAX_ITERATIONS 10000

// Global variables used by compute.c
int sx, sy, ex, ey;  // Subdomain boundaries (start x, start y, end x, end y)
int ntx, nty;        // Total grid dimensions

// Index macro to match compute.c
#define IDX(i, j) ( ((i)-(sx-1))*(ey-sy+3) + (j)-(sy-1) )



void initialization(double **pu, double **pu_new, double **pu_exact) {
  double hx, hy;
  int iterx, itery;
  double x, y;

  *pu = calloc((ex-sx+3) * (ey-sy+3), sizeof(double));
  *pu_new = calloc((ex-sx+3) * (ey-sy+3), sizeof(double));
  *pu_exact = calloc((ex-sx+3) * (ey-sy+3), sizeof(double));
  f = calloc((ex-sx+3) * (ey-sy+3), sizeof(double));

  /* Grid spacing */
  hx = 1 / (ntx + 1.0);
  hy = 1 / (nty + 1.0);

  /* Compute coefficients */
  coef[0] = (0.5 * hx * hx * hy * hy) / (hx * hx + hy * hy);
  coef[1] = 1.0 / (hx * hx);
  coef[2] = 1.0 / (hy * hy);

  /* Initialize right-hand side and compute exact solution */
  for (iterx = sx; iterx < ex + 1; iterx++) {
    for (itery = sy; itery < ey + 1; itery++) {
      x = iterx * hx;
      y = itery * hy;
      f[IDX(iterx, itery)] = 2 * (x * x - x + y * y - y);
      (*pu_exact)[IDX(iterx, itery)] = x * y * (x - 1) * (y - 1);
    }
  }
}

/*
 * Compute the solution u_new at iteration n+1
 */
void compute(double *u, double *u_new) {
  int iterx, itery;

  for (iterx = sx; iterx < ex + 1; iterx++) {
    for (itery = sy; itery < ey + 1; itery++) {
      u_new[IDX(iterx, itery)] =
        coef[0] * (
          coef[1] * (u[IDX(iterx + 1, itery)] + u[IDX(iterx - 1, itery)]) +
          coef[2] * (u[IDX(iterx, itery + 1)] + u[IDX(iterx, itery - 1)]) -
          f[IDX(iterx, itery)]
        );
    }
  }
}

/*
 * Print results
 */
void output_results(double *u, double *u_exact) {
  int itery;

  printf("Exact solution u_exact - Computed solution u\n");
  for (itery = sy; itery < ey + 1; itery++)
    printf("%12.5e - %12.5e\n", u_exact[IDX(1, itery)], u[IDX(1, itery)]);
}


// Exchange ghost cells with neighbors
void exchange_boundaries(double *u, MPI_Comm cart_comm,
                        int north, int south, int east, int west) {
    MPI_Status status;
    int local_nx = ex - sx + 1;
    int local_ny = ey - sy + 1;
    
    // Create buffers for column exchange
    double *send_col = (double*)malloc((local_nx + 2) * sizeof(double));
    double *recv_col = (double*)malloc((local_nx + 2) * sizeof(double));
    
    // Exchange with north (send top row, receive from north into ghost row)
    if (north != MPI_PROC_NULL) {
        MPI_Sendrecv(&u[IDX(sx, sy)], local_ny, MPI_DOUBLE, north, 0,
                     &u[IDX(sx-1, sy)], local_ny, MPI_DOUBLE, north, 1,
                     cart_comm, &status);
    }
    
    // Exchange with south (send bottom row, receive from south into ghost row)
    if (south != MPI_PROC_NULL) {
        MPI_Sendrecv(&u[IDX(ex, sy)], local_ny, MPI_DOUBLE, south, 1,
                     &u[IDX(ex+1, sy)], local_ny, MPI_DOUBLE, south, 0,
                     cart_comm, &status);
    }
    
    // Exchange with west (send left column, receive from west into ghost column)
    if (west != MPI_PROC_NULL) {
        for (int i = sx-1; i <= ex+1; i++) {
            send_col[i-(sx-1)] = u[IDX(i, sy)];
        }
        MPI_Sendrecv(send_col, local_nx+2, MPI_DOUBLE, west, 2,
                     recv_col, local_nx+2, MPI_DOUBLE, west, 3,
                     cart_comm, &status);
        for (int i = sx-1; i <= ex+1; i++) {
            u[IDX(i, sy-1)] = recv_col[i-(sx-1)];
        }
    }
    
    // Exchange with east (send right column, receive from east into ghost column)
    if (east != MPI_PROC_NULL) {
        for (int i = sx-1; i <= ex+1; i++) {
            send_col[i-(sx-1)] = u[IDX(i, ey)];
        }
        MPI_Sendrecv(send_col, local_nx+2, MPI_DOUBLE, east, 3,
                     recv_col, local_nx+2, MPI_DOUBLE, east, 2,
                     cart_comm, &status);
        for (int i = sx-1; i <= ex+1; i++) {
            u[IDX(i, ey+1)] = recv_col[i-(sx-1)];
        }
    }
    
    free(send_col);
    free(recv_col);
}

// Compute local residual
double compute_residual(double *u, double *u_new) {
    double local_residual = 0.0;
    
    for (int i = sx; i <= ex; i++) {
        for (int j = sy; j <= ey; j++) {
            double diff = u_new[IDX(i, j)] - u[IDX(i, j)];
            local_residual += diff * diff;
        }
    }
    
    return local_residual;
}

int main(int argc, char* argv[]) {
    int rank, size;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};  // Non-periodic boundaries
    int reorder = 1;
    int coord[2];
    
    int north, south, east, west;
    MPI_Comm cart_comm;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Grid size (default or from command line)
    ntx = 12;  // Number of interior points in x (default)
    nty = 10;  // Number of interior points in y (default)
    
    if (argc >= 3) {
        ntx = atoi(argv[1]);
        nty = atoi(argv[2]);
    }
    
    // Create 2D Cartesian topology
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coord);
    
    // Compute local subdomain indices (1-based, matching compute.c convention)
    int base_nx = ntx / dims[0];
    int base_ny = nty / dims[1];
    int extra_x = ntx % dims[0];
    int extra_y = nty % dims[1];
    
    // Calculate starting indices (1-based)
    sx = 1 + coord[0] * base_nx + (coord[0] < extra_x ? coord[0] : extra_x);
    sy = 1 + coord[1] * base_ny + (coord[1] < extra_y ? coord[1] : extra_y);
    
    // Calculate local sizes
    int local_nx = base_nx + (coord[0] < extra_x ? 1 : 0);
    int local_ny = base_ny + (coord[1] < extra_y ? 1 : 0);
    
    // Calculate ending indices
    ex = sx + local_nx - 1;
    ey = sy + local_ny - 1;
    
    // Find neighbors
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);
    
    if (rank == 0) {
        printf("Poisson execution with %d MPI processes\n", size);
        printf("Domain size: ntx=%d nty=%d\n", ntx, nty);
        printf("Topology dimensions: %d along x, %d along y\n", dims[0], dims[1]);
        printf("-----------------------------------------\n");
    }
    
    // Print topology information for each process
    for (int r = 0; r < size; r++) {
        if (rank == r) {
            printf("Rank in the topology: %d Array indices: x from %d to %d, y from %d to %d\n",
                   rank, sx, ex, sy, ey);
            printf("Process %d has neighbors: N %d E %d S %d W %d\n",
                   rank, north, east, south, west);
            fflush(stdout);
        }
        MPI_Barrier(cart_comm);
    }
    
    // Allocate arrays using compute.c initialization
    double *u = NULL;
    double *u_new = NULL;
    double *u_exact = NULL;
    
    initialization(&u, &u_new, &u_exact);
    
    if (rank == 0) {
        printf("-----------------------------------------\n");
    }
    
    MPI_Barrier(cart_comm);
    double start_time = MPI_Wtime();
    
    // Jacobi iteration
    int iteration = 0;
    double global_residual = TOLERANCE + 1.0;
    
    while (iteration < MAX_ITERATIONS && global_residual > TOLERANCE) {
        // Exchange ghost cells
        exchange_boundaries(u, cart_comm, north, south, east, west);
        
        // Perform Jacobi iteration (compute.c)
        compute(u, u_new);
        
        // Compute residual
        double local_residual = compute_residual(u, u_new);
        
        // Global reduction
        MPI_Allreduce(&local_residual, &global_residual, 1, MPI_DOUBLE,
                     MPI_SUM, cart_comm);
        global_residual = sqrt(global_residual);
        
        // Swap arrays
        double *temp = u;
        u = u_new;
        u_new = temp;
        
        iteration++;
        
        // Print progress every 100 iterations
        if (rank == 0 && iteration % 100 == 0) {
            printf("Iteration %d global_error = %.5e\n", iteration, global_residual);
            fflush(stdout);
        }
    }
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        if (global_residual <= TOLERANCE) {
            printf("Converged after %d iterations in %.6f seconds\n",
                   iteration, end_time - start_time);
        } else {
            printf("Did not converge after %d iterations (residual = %.5e)\n",
                   iteration, global_residual);
        }
    }
    
    // Output results from process 0 (using compute.c function)
    if (rank == 0) {
        output_results(u, u_exact);
    }
    
    // Cleanup
    free(u);
    free(u_new);
    free(u_exact);
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}

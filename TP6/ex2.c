#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_FEATURES 5
#define LEARNING_RATE 0.01
#define MAX_EPOCHS 10000
#define CONVERGENCE_THRESHOLD 0.01

// Training sample structure
typedef struct {
    double x[N_FEATURES];
    double y;
} Sample;

// Generate synthetic dataset: y = 2*x[0] - x[1] + 0.5*x[2] + noise
void generate_data(Sample* data, int n_samples, unsigned int seed) {
    srand(seed);
    
    double true_weights[N_FEATURES] = {2.0, -1.0, 0.5, 0.0, 0.0};
    
    for (int i = 0; i < n_samples; i++) {
        // Generate random features
        for (int j = 0; j < N_FEATURES; j++) {
            data[i].x[j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // [-1, 1]
        }
        
        // Generate label based on true weights plus noise
        data[i].y = 0.0;
        for (int j = 0; j < N_FEATURES; j++) {
            data[i].y += true_weights[j] * data[i].x[j];
        }
        
        // Add Gaussian noise
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.2;
        data[i].y += noise;
    }
}

// Predict value: y_pred = w[0]*x[0] + w[1]*x[1] + ... + w[n-1]*x[n-1]
double predict(double* weights, double* features) {
    double result = 0.0;
    for (int i = 0; i < N_FEATURES; i++) {
        result += weights[i] * features[i];
    }
    return result;
}

// Compute local loss (MSE) and gradient
void compute_local_gradient(Sample* local_data, int local_n, double* weights,
                           double* local_gradient, double* local_loss) {
    // Initialize gradient to zero
    for (int i = 0; i < N_FEATURES; i++) {
        local_gradient[i] = 0.0;
    }
    *local_loss = 0.0;
    
    // Compute gradient and loss for local samples
    for (int i = 0; i < local_n; i++) {
        double prediction = predict(weights, local_data[i].x);
        double error = prediction - local_data[i].y;
        
        // Accumulate loss (MSE)
        *local_loss += error * error;
        
        // Accumulate gradient: dL/dw_j = 2 * error * x_j
        for (int j = 0; j < N_FEATURES; j++) {
            local_gradient[j] += 2.0 * error * local_data[i].x[j];
        }
    }
}

// Create MPI derived type for Sample structure
MPI_Datatype create_sample_type() {
    MPI_Datatype sample_type;
    
    int block_lengths[2] = {N_FEATURES, 1};
    MPI_Aint displacements[2];
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    
    // Calculate displacements
    Sample dummy_sample;
    MPI_Aint base_address;
    MPI_Get_address(&dummy_sample, &base_address);
    MPI_Get_address(&dummy_sample.x, &displacements[0]);
    MPI_Get_address(&dummy_sample.y, &displacements[1]);
    
    // Make displacements relative
    displacements[0] -= base_address;
    displacements[1] -= base_address;
    
    // Create and commit the datatype
    MPI_Type_create_struct(2, block_lengths, displacements, types, &sample_type);
    MPI_Type_commit(&sample_type);
    
    return sample_type;
}

int main(int argc, char* argv[]) {
    int rank, size;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    int n_samples = 1000;
    if (argc > 1) {
        n_samples = atoi(argv[1]);
    }
    
    if (rank == 0) {
        printf("=== Distributed Gradient Descent ===\n");
        printf("Total samples: %d\n", n_samples);
        printf("Number of processes: %d\n", size);
        printf("Features: %d\n", N_FEATURES);
        printf("Learning rate: %.4f\n", LEARNING_RATE);
        printf("Convergence threshold: %.6f\n\n", CONVERGENCE_THRESHOLD);
    }
    
    // Create MPI derived type for Sample
    MPI_Datatype sample_type = create_sample_type();
    
    // Full dataset (only on rank 0)
    Sample* full_data = NULL;
    
    // Calculate samples per process
    int samples_per_proc = n_samples / size;
    int remainder = n_samples % size;
    int local_n = (rank < remainder) ? samples_per_proc + 1 : samples_per_proc;
    
    // Local data for each process
    Sample* local_data = (Sample*)malloc(local_n * sizeof(Sample));
    
    // Rank 0 generates and distributes data
    int* sendcounts = NULL;
    int* displs = NULL;
    
    if (rank == 0) {
        full_data = (Sample*)malloc(n_samples * sizeof(Sample));
        generate_data(full_data, n_samples, 42);
        
        printf("Dataset generated. First sample:\n");
        printf("  x = [%.3f, %.3f, %.3f, %.3f, %.3f], y = %.3f\n\n",
               full_data[0].x[0], full_data[0].x[1], full_data[0].x[2],
               full_data[0].x[3], full_data[0].x[4], full_data[0].y);
        
        // Prepare scatter counts and displacements
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = (i < remainder) ? samples_per_proc + 1 : samples_per_proc;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    
    // Scatter data to all processes
    MPI_Scatterv(full_data, sendcounts, displs, sample_type,
                 local_data, local_n, sample_type,
                 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Data distributed. Starting training...\n\n");
    }
    
    // Initialize weights to zero
    double weights[N_FEATURES] = {0.0};
    double local_gradient[N_FEATURES];
    double global_gradient[N_FEATURES];
    double local_loss, global_loss;
    
    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Gradient descent loop
    int epoch;
    for (epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        // Compute local gradient and loss
        compute_local_gradient(local_data, local_n, weights,
                              local_gradient, &local_loss);
        
        // Aggregate gradients from all processes
        MPI_Allreduce(local_gradient, global_gradient, N_FEATURES,
                     MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Aggregate losses
        MPI_Allreduce(&local_loss, &global_loss, 1,
                     MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Compute average loss (MSE)
        global_loss /= n_samples;
        
        // Update weights (all processes have same weights)
        for (int i = 0; i < N_FEATURES; i++) {
            weights[i] -= LEARNING_RATE * global_gradient[i] / n_samples;
        }
        
        // Print progress every 10 epochs
        if (rank == 0 && (epoch + 1) % 10 == 0) {
            printf("Epoch %4d | Loss (MSE): %.6f | w: [", epoch + 1, global_loss);
            for (int i = 0; i < N_FEATURES; i++) {
                printf("%.4f", weights[i]);
                if (i < N_FEATURES - 1) printf(", ");
            }
            printf("]\n");
        }
        
        // Check convergence
        if (global_loss < CONVERGENCE_THRESHOLD) {
            if (rank == 0) {
                printf("\nEarly stopping at epoch %d — loss %.6f < %.2e\n",
                       epoch + 1, global_loss, CONVERGENCE_THRESHOLD);
            }
            break;
        }
    }
    
    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("\nTraining complete!\n");
        printf("Final weights: [");
        for (int i = 0; i < N_FEATURES; i++) {
            printf("%.4f", weights[i]);
            if (i < N_FEATURES - 1) printf(", ");
        }
        printf("]\n");
        printf("Final loss: %.6f\n", global_loss);
        printf("Training time: %.6f seconds (MPI with %d processes)\n", 
               end_time - start_time, size);
        printf("\nExpected weights: [2.0, -1.0, 0.5, 0.0, 0.0]\n");
    }
    
    // Cleanup
    free(local_data);
    if (rank == 0) {
        free(full_data);
        free(sendcounts);
        free(displs);
    }
    MPI_Type_free(&sample_type);
    
    MPI_Finalize();
    return 0;
}

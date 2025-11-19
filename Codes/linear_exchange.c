#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#define MAXPROCS 256

// Helper function for reduction (assuming MPI_INT and MPI_SUM)
void your_reduction_function(int* target, int* source, int count) {
    for (int i = 0; i < count; i++) {
        target[i] += source[i];
    }
}

// Linear Exchange Allreduce with Instrumentation
void linear_allreduce(int* sendbuf, int* result, int count, int rank, int size, long long* local_comm_row) {
    // int chunk_size = count / size;
    // size_t chunk_bytes = chunk_size * sizeof(int);
    
    int* tempbuf = (int*)malloc(sizeof(int) * count);
    memcpy(result, sendbuf, count * sizeof(int)); // Start with own data

    MPI_Request send_req[MAXPROCS], recv_req[MAXPROCS];

    // ========== REDUCE-SCATTER PHASE ==========
    for (int step = 1; step < size; step++) {
        int dest = (rank + step) % size;
        int source = (rank - step + size) % size;

        // We send our complete buffer and receive a complete buffer from a partner
        local_comm_row[dest] += count * sizeof(int);

        MPI_Isend(sendbuf, count, MPI_INT, dest, 0, MPI_COMM_WORLD, &send_req[step-1]);
        MPI_Irecv(tempbuf, count, MPI_INT, source, 0, MPI_COMM_WORLD, &recv_req[step-1]);
        
        MPI_Wait(&send_req[step-1], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req[step-1], MPI_STATUS_IGNORE);

        // Perform local reduction on the entire received buffer
        your_reduction_function(result, tempbuf, count);
    }
    
    // At this point, `result` holds the final reduced data for every process.
    // The "allgather" is implicit because every process received and reduced all data.

    free(tempbuf);
}


// Main function structure for instrumentation and latency measurement
int main(int argc, char** argv) {
    if (argc != 2) {
        if (argc > 1) { // Only rank 0 prints usage
             int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
             if (rank == 0) fprintf(stderr, "Usage: %s <buffer_size_bytes>\n", argv[0]);
        }
        MPI_Init(&argc, &argv);
        MPI_Finalize();
        return 1;
    }
    long buffer_size_bytes = atol(argv[1]);
    
    MPI_Init(&argc, &argv);

    int r, p;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &r);
    MPI_Comm_size(comm, &p);

    if (buffer_size_bytes <= 0) {
        if (r == 0) fprintf(stderr, "Invalid buffer size.\n");
        MPI_Finalize();
        return 1;
    }

    const int TOTAL_ELEMENTS = buffer_size_bytes / sizeof(int);
    if (TOTAL_ELEMENTS == 0 || TOTAL_ELEMENTS % p != 0) {
        if (r == 0) fprintf(stderr, "Error: Total elements (%d) must be non-zero and divisible by process count (%d).\n", TOTAL_ELEMENTS, p);
        MPI_Finalize();
        return 1;
    }

    int* input_V = (int*)malloc(TOTAL_ELEMENTS * sizeof(int));
    int* result_ar = (int*)malloc(TOTAL_ELEMENTS * sizeof(int)); 
    long long* local_comm_row_ar = (long long*)calloc(p, sizeof(long long));

    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        input_V[i] = r;
    }
    
    // --- WARM-UP CALL ---
    linear_allreduce(input_V, result_ar, TOTAL_ELEMENTS, r, p, local_comm_row_ar);
    
    // --- RESET COMM COUNTER ---
    memset(local_comm_row_ar, 0, p * sizeof(long long));

    double start_time, end_time, elapsed_time;
    
    MPI_Barrier(comm); 
    start_time = MPI_Wtime();

    linear_allreduce(input_V, result_ar, TOTAL_ELEMENTS, r, p, local_comm_row_ar);

    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    double max_latency;
    MPI_Reduce(&elapsed_time, &max_latency, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (r == 0) {
        long buffer_size_kb = buffer_size_bytes / 1024;
        printf("linear_exchange,%d,%ld,%.10f\n", p, buffer_size_kb, max_latency);
    }

    long long* full_comm_matrix_ar = NULL;
    if (r == 0) {
        full_comm_matrix_ar = (long long*)malloc(p * p * sizeof(long long));
    }

    MPI_Gather(local_comm_row_ar, p, MPI_LONG_LONG, 
               full_comm_matrix_ar, p, MPI_LONG_LONG, 0, comm);

    if (r == 0) {
        char filename[100];
        snprintf(filename, sizeof(filename), "linear_exchange_%d.csv", p);
        FILE* f_ar = fopen(filename, "w");
        if (f_ar != NULL) {
            for (int i = 0; i < p; i++) {
                for (int j = 0; j < p; j++) {
                    fprintf(f_ar, "%lld%s", full_comm_matrix_ar[i * p + j], (j < p - 1) ? "," : "");
                }
                fprintf(f_ar, "\n");
            }
            fclose(f_ar);
        }
        free(full_comm_matrix_ar);
    }

    free(input_V);
    free(result_ar);
    free(local_comm_row_ar);

    MPI_Finalize();
    return 0;
}
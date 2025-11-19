#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// Helper function for reduction
void your_reduction_function(int* target, int* source, int count) {
    for (int i = 0; i < count; i++) {
        target[i] += source[i];
    }
}

// Ring-Based Allreduce
void ring_allreduce(int* sendbuf, int* result, int count, int rank, int size, long long* local_comm_row) {
    int chunk_size = count / size;
    size_t chunk_bytes = chunk_size * sizeof(int);
    
    memcpy(result, sendbuf, sizeof(int) * count);
    int* temp_chunk = (int*)malloc(chunk_bytes);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    // Phase 1: Reduce-Scatter
    for (int i = 0; i < size - 1; ++i) {
        int send_block_idx = (rank - i + size) % size;
        int recv_block_idx = (rank - i - 1 + size) % size;

        local_comm_row[next] += chunk_bytes;
        MPI_Sendrecv(result + send_block_idx * chunk_size, chunk_size, MPI_INT, next, 0,
                     temp_chunk, chunk_size, MPI_INT, prev, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        your_reduction_function(result + recv_block_idx * chunk_size, temp_chunk, chunk_size);
    }

    // Phase 2: Allgather
    for (int i = 0; i < size - 1; ++i) {
        int send_block_idx = (rank - i + 1 + size) % size;
        int recv_block_idx = (rank - i + size) % size;

        local_comm_row[next] += chunk_bytes;
        MPI_Sendrecv(result + send_block_idx * chunk_size, chunk_size, MPI_INT, next, 1,
                     result + recv_block_idx * chunk_size, chunk_size, MPI_INT, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    free(temp_chunk);
}

// Main function
int main(int argc, char** argv) {
    if (argc != 2) {
        MPI_Init(&argc, &argv);
        int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) fprintf(stderr, "Usage: %s <buffer_size_bytes>\n", argv[0]);
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

    for (int i = 0; i < TOTAL_ELEMENTS; i++) input_V[i] = r;
    
    // --- WARM-UP CALL ---
    ring_allreduce(input_V, result_ar, TOTAL_ELEMENTS, r, p, local_comm_row_ar);
    
    // --- RESET COMM COUNTER ---
    memset(local_comm_row_ar, 0, p * sizeof(long long));

    double start_time, end_time, elapsed_time;
    
    MPI_Barrier(comm); 
    start_time = MPI_Wtime();

    ring_allreduce(input_V, result_ar, TOTAL_ELEMENTS, r, p, local_comm_row_ar);

    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    double max_latency;
    MPI_Reduce(&elapsed_time, &max_latency, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (r == 0) {
        long buffer_size_kb = buffer_size_bytes / 1024;
        printf("ring,%d,%ld,%.10f\n", p, buffer_size_kb, max_latency);
    }

    long long* full_comm_matrix_ar = NULL;
    if (r == 0) full_comm_matrix_ar = (long long*)malloc(p * p * sizeof(long long));

    MPI_Gather(local_comm_row_ar, p, MPI_LONG_LONG, full_comm_matrix_ar, p, MPI_LONG_LONG, 0, comm);

    if (r == 0) {
        char filename[100];
        snprintf(filename, sizeof(filename), "ring_%d.csv", p);
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
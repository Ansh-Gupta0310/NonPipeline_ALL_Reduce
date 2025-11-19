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

// Non-Pipeline Allreduce based on instrumented_allreduce.c
void non_pipeline_allreduce(
    void* input_V, void* result_W, int m_total_elements, 
    MPI_Comm comm, long long* local_comm_row) 
{
    int r, p;
    MPI_Comm_rank(comm, &r);
    MPI_Comm_size(comm, &p);

    int blk_size = m_total_elements / p;
    size_t blk_bytes = blk_size * sizeof(int);
    
    size_t r_buf_bytes = p * blk_bytes;
    int* partial_R = (int*)malloc(r_buf_bytes);
    
    size_t t_buf_bytes = (int)ceil((double)p / 2.0) * blk_bytes;
    int* recv_T = (int*)malloc(t_buf_bytes);

    int stack[64];
    int stack_ptr = 0;

    // Phase 1: Reduce-Scatter
    for (int i = 0; i < p; i++) {
        memcpy(partial_R + (i * blk_size), 
               (char*)input_V + (((r + i) % p) * blk_bytes), 
               blk_bytes);
    }
    
    int s = p;
    while (s > 1) {
        stack[stack_ptr++] = s;
        int s_prime = s;
        s = (int)ceil((double)s / 2.0);
        int num_blocks = s_prime - s;
        size_t comm_bytes = num_blocks * blk_bytes;
        int to_proc = (r + s) % p;
        int from_proc = (r - s + p) % p;

        local_comm_row[to_proc] += comm_bytes;

        MPI_Sendrecv(partial_R + (s * blk_size), num_blocks * blk_size, MPI_INT, to_proc, 0,
                     recv_T, num_blocks * blk_size, MPI_INT, from_proc, 0,
                     comm, MPI_STATUS_IGNORE);

        for (int i = 0; i < num_blocks; i++) {
            your_reduction_function(partial_R + (i * blk_size), 
                                    recv_T + (i * blk_size), 
                                    blk_size);
        }
    }

    // Phase 2: Allgather
    s = 1;
    while (stack_ptr > 0) {
        int s_prime = stack[--stack_ptr];
        int num_blocks = s_prime - s;
        size_t comm_bytes = num_blocks * blk_bytes;
        int to_proc = (r - s + p) % p;
        int from_proc = (r + s) % p;

        local_comm_row[to_proc] += comm_bytes;

        MPI_Sendrecv(partial_R, num_blocks * blk_size, MPI_INT, to_proc, 1,
                     partial_R + (s * blk_size), num_blocks * blk_size, MPI_INT, from_proc, 1,
                     comm, MPI_STATUS_IGNORE);
        
        s = s_prime;
    }

    // Final Data Rotation
    for (int i = 0; i < p; i++) {
        int target_rank_block = (r + i) % p;
        memcpy((char*)result_W + (target_rank_block * blk_bytes), 
               partial_R + (i * blk_size), 
               blk_bytes);
    }

    free(partial_R);
    free(recv_T);
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
    non_pipeline_allreduce(input_V, result_ar, TOTAL_ELEMENTS, comm, local_comm_row_ar);
    
    // --- RESET COMM COUNTER ---
    memset(local_comm_row_ar, 0, p * sizeof(long long));
    
    double start_time, end_time, elapsed_time;
    
    MPI_Barrier(comm); 
    start_time = MPI_Wtime();

    non_pipeline_allreduce(input_V, result_ar, TOTAL_ELEMENTS, comm, local_comm_row_ar);

    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    
    double max_latency;
    MPI_Reduce(&elapsed_time, &max_latency, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (r == 0) {
        long buffer_size_kb = buffer_size_bytes / 1024;
        printf("non_pipeline_allreduce,%d,%ld,%.10f\n", p, buffer_size_kb, max_latency);
    }

    long long* full_comm_matrix_ar = NULL;
    if (r == 0) full_comm_matrix_ar = (long long*)malloc(p * p * sizeof(long long));

    MPI_Gather(local_comm_row_ar, p, MPI_LONG_LONG, full_comm_matrix_ar, p, MPI_LONG_LONG, 0, comm);

    if (r == 0) {
        char filename[100];
        snprintf(filename, sizeof(filename), "non_pipeline_allreduce_%d.csv", p);
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
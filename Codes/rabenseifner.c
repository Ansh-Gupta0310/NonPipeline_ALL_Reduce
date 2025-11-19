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

// Rabenseifner's Allreduce Algorithm
void rabenseifner_allreduce(int* sendbuf, int* result, int count, int rank, int size, long long* local_comm_row) {
    memcpy(result, sendbuf, sizeof(int) * count);
    int* tempbuf = (int*)malloc(sizeof(int) * count);
    
    // Phase 1: Reduce-Scatter (recursive halving)
    for (int d = 1; d < size; d <<= 1) {
        int partner = rank ^ d;
        if (partner < size) {
            int send_offset = 0;
            int recv_offset = 0;
            int segment_size = count / (2*d); // This simplified logic assumes count is divisible by 2*d
            
            if ((rank / d) % 2 == 0) {
                 send_offset = (rank % d) * segment_size + segment_size;
            } else {
                 send_offset = (rank % d) * segment_size;
            }
            recv_offset = send_offset;

            size_t comm_bytes = segment_size * sizeof(int);
            local_comm_row[partner] += comm_bytes;

            MPI_Sendrecv(result + send_offset, segment_size, MPI_INT, partner, 0,
                         tempbuf, segment_size, MPI_INT, partner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            your_reduction_function(result + recv_offset, tempbuf, segment_size);
        }
    }
    
    // Phase 2: Allgather (recursive doubling)
    for (int d = size >> 1; d > 0; d >>= 1) {
        int partner = rank ^ d;
        if (partner < size) {
            // This logic is also simplified and assumes count is divisible
            int send_offset = (rank % (2*d)) * (count / (2*d));
            int recv_offset = ((rank % (2*d)) ^ d) * (count / (2*d));
            int segment_size = count / (2*d);
            
            size_t comm_bytes = segment_size * sizeof(int);
            local_comm_row[partner] += comm_bytes;

            MPI_Sendrecv(result + send_offset, segment_size, MPI_INT, partner, 1,
                         result + recv_offset, segment_size, MPI_INT, partner, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    free(tempbuf);
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
    
    int is_power_of_2 = (p > 0) && ((p & (p - 1)) == 0);
    const int TOTAL_ELEMENTS = buffer_size_bytes / sizeof(int);

    if (is_power_of_2 && (TOTAL_ELEMENTS == 0 || TOTAL_ELEMENTS % p != 0)) {
        if (r == 0) fprintf(stderr, "Error: Total elements (%d) must be non-zero and divisible by process count (%d).\n", TOTAL_ELEMENTS, p);
        MPI_Finalize();
        return 1;
    }

    int* input_V = (int*)malloc(TOTAL_ELEMENTS * sizeof(int));
    int* result_ar = (int*)malloc(TOTAL_ELEMENTS * sizeof(int)); 
    long long* local_comm_row_ar = (long long*)calloc(p, sizeof(long long));

    for (int i = 0; i < TOTAL_ELEMENTS; i++) input_V[i] = r;
    
    // --- WARM-UP CALL (only if algorithm will be run) ---
    if (is_power_of_2) {
        rabenseifner_allreduce(input_V, result_ar, TOTAL_ELEMENTS, r, p, local_comm_row_ar);
        // --- RESET COMM COUNTER ---
        memset(local_comm_row_ar, 0, p * sizeof(long long));
    }

    MPI_Barrier(comm); 
    
    if (is_power_of_2) {
        double start_time = MPI_Wtime();
        rabenseifner_allreduce(input_V, result_ar, TOTAL_ELEMENTS, r, p, local_comm_row_ar);
        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;
        
        double max_latency;
        MPI_Reduce(&elapsed_time, &max_latency, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (r == 0) {
            long buffer_size_kb = buffer_size_bytes / 1024;
            printf("rabenseifner,%d,%ld,%.10f\n", p, buffer_size_kb, max_latency);
        }
    } else {
        if (r == 0) {
            long buffer_size_kb = buffer_size_bytes / 1024;
            printf("rabenseifner,%d,%ld,-1.0\n", p, buffer_size_kb);
        }
    }

    long long* full_comm_matrix_ar = NULL;
    if (r == 0) full_comm_matrix_ar = (long long*)malloc(p * p * sizeof(long long));

    MPI_Gather(local_comm_row_ar, p, MPI_LONG_LONG, full_comm_matrix_ar, p, MPI_LONG_LONG, 0, comm);

    if (r == 0 && is_power_of_2) {
        char filename[100];
        snprintf(filename, sizeof(filename), "rabenseifner_%d.csv", p);
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
    }
    if (r == 0) free(full_comm_matrix_ar);

    free(input_V);
    free(result_ar);
    free(local_comm_row_ar);

    MPI_Finalize();
    return 0;
}
#!/bin/bash

# Clear previous results
rm -f latency_results.csv
touch latency_results.csv

# Header for the CSV file
echo "algorithm,process_count,buffer_size_kb,latency_seconds" > latency_results.csv

# Benchmark Parameters
ALGOS=("linear_exchange" "ring" "rabenseifner" "non_pipeline_allreduce")
PROCS=(4 8 16 32 64)
SIZES_KB=(1 64 256 1024 16384 65536) # 1KB, 64KB, 256KB, 1MB, 16MB, 64MB

echo "Starting benchmarks..."

for algo in "${ALGOS[@]}"; do
    for p in "${PROCS[@]}"; do
        for size_kb in "${SIZES_KB[@]}"; do
            size_bytes=$((size_kb * 1024))
            
            echo "Running: $algo | Processes: $p | Buffer: ${size_kb}KB"
            
            # Use --oversubscribe for local testing if you have fewer cores than 'p'
            # Remove it on a real HPC cluster.
            mpirun -np "$p" --oversubscribe ./"$algo" "$size_bytes" >> latency_results.csv
        done
    done
done

echo "Benchmark data collected in latency_results.csv"
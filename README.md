**Project Overview**
- **What:** A small MPI benchmark suite that implements several Allreduce approaches and instruments the pairwise communication volume between ranks.
- **Location:** All code and helpers are in the `Codes/` folder.

**Algorithms Included**
- **`linear_exchange`**: Each process repeatedly sends/receives the entire buffer to/from partners (linear exchange). This is simple and easy to instrument but causes large point-to-point transfers.
- **`ring`**: A ring-based reduce-scatter followed by an allgather. Each step sends only a chunk, so the communication is balanced and incremental.
- **`rabenseifner`**: Rabenseifner's algorithm (reduce-scatter by recursive halving + allgather by recursive doubling). Implemented only when the process count is a power of two (the program prints `-1.0` for latency otherwise). It splits the buffer into segments and exchanges only needed segments.
- **`non_pipeline_allreduce`**: The repository's unique implementation. It performs a reduce-scatter followed by an allgather but exchanges larger contiguous blocks (not pipelined chunk-by-chunk). It builds an in-memory partial-result buffer (`partial_R`) that holds per-rank blocks, then repeatedly exchanges groups of blocks using `MPI_Sendrecv` (using a stack to replay the allgather phase). The communication pattern tends to move larger contiguous blocks in each step (fewer but larger messages) compared to fully pipelined approaches (e.g., the ring), and it records bytes sent from each rank to each other rank so you can visualize the communication matrix.

**Why `non_pipeline_allreduce` is interesting**
- **Block-oriented (non-pipelined) exchanges:** Instead of sending many small, pipelined chunks, this approach exchanges larger groups of blocks in each communication step. That changes both latency and per-link bandwidth utilization compared to ring or linear exchange.
- **Instrumented comm matrix:** Each implementation accumulates the number of bytes sent from each source rank to every destination rank into a per-rank array and then `MPI_Gather`s the results on rank 0. This produces a CSV file per algorithm containing a p x p matrix of bytes (rows = source rank, columns = destination rank).
- **Comparison points:** Use the CSV matrices and latency outputs to compare (a) number of messages and bytes per link, (b) whether traffic is concentrated along a few links (non-pipelined) or distributed evenly (ring, rabenseifner), and (c) sensitivity to message size.

**Files of interest**
- `Codes/Makefile` : Build targets for each algorithm (`linear_exchange`, `ring`, `rabenseifner`, `non_pipeline_allreduce`).
- `Codes/run_benchmarks.sh` : Simple bash script that runs each algorithm across a set of process counts and buffer sizes and appends latency lines to `latency_results.csv`.
- `Codes/heatmap-generator.py` : A small Python script using `pandas` + `seaborn` to plot a heatmap from an instrumented CSV (example expects `heatmap_reduce_scatter.csv`/similar).
- `Codes/*.c` : The four implementations; each writes two outputs on rank 0:
  - A latency line printed to stdout in the format: `algorithm,process_count,buffer_size_kb,latency_seconds` (example used by `run_benchmarks.sh`).
  - A file named `<algorithm>_<p>.csv` containing the gathered p x p communication volume matrix (bytes).

**Build instructions**
1. Change to the `Codes` directory:

```powershell
cd Codes
```

2. Build all targets (requires `mpicc`):

```powershell
make
```

Or build a single target, e.g.:

```powershell
make non_pipeline_allreduce
```

**Run instructions (single experiment)**
- Basic pattern: run the compiled binary with `mpirun -np <p> ./<algorithm> <buffer_size_bytes>`.
- Example (run `non_pipeline_allreduce` with 8 processes and 1 MB buffer):

```powershell
mpirun -np 8 --oversubscribe ./non_pipeline_allreduce 1048576
```

- Outputs produced by a successful run (on rank 0):
  - A single-line latency print to stdout, e.g. `non_pipeline_allreduce,8,1024,0.0123456789` where `1024` is buffer size in KB.
  - A communication matrix file: `non_pipeline_allreduce_8.csv`. This CSV is p rows by p columns where cell (i,j) is the number of bytes that rank `i` recorded having sent to rank `j`.

**Running the full benchmark suite**
- The provided `run_benchmarks.sh` runs all algorithms, multiple process counts and sizes, and appends latency lines to `latency_results.csv`.
- Usage (from `Codes/`):

```powershell
# ensure the script is executable on Unix. On Windows, run in WSL or adapt the script.
./run_benchmarks.sh
```

- Note: `run_benchmarks.sh` uses `mpirun -np` and `--oversubscribe` (helpful for local testing). On an HPC cluster remove `--oversubscribe` and follow the cluster's job submission system.

**Generating plots / heatmaps**
- Each algorithm produces a `<algorithm>_<p>.csv` matrix on rank 0. To generate a heatmap for an algorithm (for example `non_pipeline_allreduce_8.csv`) move or rename the CSV to the name expected by `heatmap-generator.py` or adapt the script to read the file directly.

Example using the script as-is (edit the script to point to your CSV if needed):

```powershell
# from Codes/
python heatmap-generator.py
```

- What the heatmap shows: rows are source ranks (senders), columns are destination ranks (receivers). The cell values are the number of bytes sent from that source to that destination. Bright or large values indicate concentrated traffic along that rank pair. This helps visualize whether the algorithm concentrates communication on a few links (likely for block-based non-pipelined exchanges) or spreads traffic evenly (typical of ring / balanced algorithms).

**Interpreting results**
- **Latency lines (`latency_results.csv`)**: Use these to compare end-to-end completion time (wall-clock) for an entire Allreduce operation across algorithms, sizes, and process counts.
- **Communication matrices (`<algo>_<p>.csv`)**: Use these to see which rank pairs exchange the most bytes. Patterns to look for:
  - Diagonal/near-diagonal patterns: often indicate neighbor-heavy (ring) communication.
  - Concentrated rows/columns: a rank that sends many bytes to a few ranks (could indicate non-pipelined large transfers).
  - Uniform distribution: well-balanced algorithms like Rabenseifner (when applicable) should spread communication.

**Notes & caveats**
- `rabenseifner` in this repo assumes power-of-two process counts. When `p` is not a power of two the program prints `-1.0` for latency and still produces a CSV with recorded comms (but the algorithm's main flow is skipped).
- The instrumentation records bytes added to a per-rank array via `local_comm_row[...] += comm_bytes;` — this represents the bytes counted for the sender to that destination inside the algorithm. The gathered matrix depends on those increments and how the implementation chooses to account for segments.
- For accurate performance testing use an HPC environment or a multicore machine without `--oversubscribe` and ensure your MPI implementation is tuned appropriately.

**Next steps / Suggestions**
- If you want, I can:
  - Run the `Makefile` and `run_benchmarks.sh` on your machine (if you want me to run them here, confirm access and environment). 
  - Modify `heatmap-generator.py` to accept a filename CLI argument and auto-detect the algorithm name.
  - Add a small `README` inside `Codes/` explaining local vs cluster usage and examples for Windows/PowerShell.

**Contact / Credits**
- This README was generated to document the contents of the `Codes/` folder and to explain how to run and interpret the experiments. If you want more detailed analysis (derived plots, aggregated comparisons), tell me which plots you prefer and I'll add scripts.

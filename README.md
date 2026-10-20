# CUDA K-means Benchmark

K-means clustering implemented three ways, with measured runtimes: a serial CPU version in C, a CUDA GPU version, and Python tooling to benchmark, profile and visualize them. Run on a Google Colab Tesla T4.

> Formerly named `knn_parallel_computation`. The notebook implements **K-means**, not KNN.

## What is in the notebook

`PPproject.ipynb` (a Colab notebook, parallel-programming course project):

1. **CPU serial K-means** (`kmeans_cpu_serial.c`): k-means++ initialization, per-iteration profiling, CSV outputs. Built with `gcc -O3 -march=native`.
2. **CUDA K-means** (`kmeans_cuda_prof.cu`): one thread per point; `assign_accumulate_kernel` assigns each point to its nearest centroid and accumulates cluster sums with `atomicAdd`. Block size 256, profiled per iteration. Built with `nvcc -O3 -arch=sm_75`.
3. **Benchmark harness** in Python: runs both programs across dataset sizes, parses `total_time`, writes CSV summaries.
4. **Monitoring** of runs with `psutil`, and a **visualization** that replays the clustering as a GIF.

Synthetic data from a fixed-seed generator (same logic in the C, CUDA and Python code): D=16 features, K=10 clusters, up to 200 iterations, tolerance 1e-4, k-means++ init.

## Measured results (Tesla T4, from the notebook outputs)

| Points (N) | CPU serial | CUDA | Speedup |
|---:|---:|---:|---:|
| 1,000 | 3.2 ms | 9.0 ms | 0.35x (GPU slower) |
| 5,000 | 106.8 ms | 62.2 ms | 1.7x |
| 20,000 | 379.1 ms | 80.0 ms | 4.7x |
| 100,000 | 3,600 ms | 255 ms | 14.1x |
| 300,000 | 10,413 ms | 577 ms | 18.0x |
| 500,000 | 15,306 ms | 926 ms | 16.5x |

CUDA also ran N = 1,000,000 (1.9 s) and 2,000,000 (3.5 s). The serial run output for those two sizes was not captured in the notebook.

Takeaway: below roughly 5,000 points the GPU launch and transfer overhead dominates and the CPU wins. Above about 100,000 points the GPU is 14-18x faster. These are single runs on one machine and one GPU, not averaged benchmarks.

## Run it

1. Open `PPproject.ipynb` in [Google Colab](https://colab.research.google.com/) and choose a GPU runtime (T4).
2. Run the cells in order. The first writes and compiles the C code, the CUDA cells need `nvcc` (available in Colab GPU runtimes).
3. Change `-arch=sm_75` to match your GPU if you run elsewhere.

Note: one cell recompiles the CUDA file before `kmeans_cpu_serial.c` exists and shows a compile error in the saved output. It is harmless; run the cells in order.

## Ideas to extend

- Add an OpenMP multi-core CPU version (the data generator already mentions one) for a three-way comparison.
- Average several runs and report variance.
- Use shared memory to cut global-memory `atomicAdd` contention in the accumulation step.
- Split the code out of the notebook into `src/` with a Makefile.

## License

MIT

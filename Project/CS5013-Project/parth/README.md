# Vamana GPU Graph Search & Construction

This project implements the Vamana graph construction and search algorithm on the GPU using CUDA. It supports both high-dimensional SIFT datasets and 2D toy datasets for visualization.

## Features

- **CUDA Vamana Construction**: Fast graph construction on GPU (`src/build_graph.cu`).
- **CUDA Search Engine**: Optimized search kernel with robust pruning (`src/engine.cu`).
- **Visualization**: Step-by-step visualization of the search process (`src/engine_vis.cu`, `tests_2d/vis_step_by_step.py`).
- **Automated Benchmarking**: Measures Recall@10 and QPS across different L values.
- **Custom SIFT Support**: Run benchmarks on subsets of the SIFT-10K dataset.

## Prerequisites

- NVIDIA GPU with CUDA support
- `nvcc` compiler
- Python 3.x
- Python packages: `numpy`, `matplotlib`, `imageio`, `scikit-learn`

## Quick Start

Run the entire benchmark suite (Toy, Visualization, SIFT, Custom SIFT):

```bash
make run_all
```

## Individual Commands

### 1. Toy Dataset (2D)
Generates random 2D data, builds the graph, and runs the search engine.
```bash
make run_toy
```

### 2. Visualization
Runs the visualization pipeline and generates `vis_outputs/step_by_step/step_by_step.gif`.
```bash
make run_vis_toy
```

### 3. SIFT-10K Benchmark
Downloads SIFT-10K, builds the graph, and runs the benchmark on the full dataset (10,000 nodes).
```bash
make run_sift
```

### 4. Custom SIFT Benchmark
Runs the benchmark on a subset of the SIFT dataset (default: 5,000 nodes).
```bash
make run_custom_sift
```

## Configuration

Configuration files are located in the root directory:
- `config_sift.json`: Configuration for the full SIFT benchmark.
- `config_custom_sift.json`: Configuration for the custom SIFT benchmark (change `size` here).
- `config_toy.json`: Configuration for the toy dataset.
- `config_vis.json`: Configuration for visualization.

## Documentation

The source code is documented using Doxygen style comments. To generate HTML documentation:

1. Install Doxygen: `sudo apt install doxygen`
2. Run: `doxygen Doxyfile`
3. Open `docs/html/index.html` in your browser.

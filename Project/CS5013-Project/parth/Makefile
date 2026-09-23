NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -I src -Wno-deprecated-gpu-targets
PYTHON = python3

all: sift toy

sift: src/engine.cu src/common.h src/device_funcs.cu
	$(NVCC) $(NVCC_FLAGS) -D DIM=128 -o engine_sift src/engine.cu

toy: src/engine.cu src/common.h src/device_funcs.cu
	$(NVCC) $(NVCC_FLAGS) -D DIM=2 -o engine_toy src/engine.cu

build_graph_sift: src/build_graph.cu src/common.h src/device_funcs.cu
	$(NVCC) $(NVCC_FLAGS) -D DIM=128 -o build_graph_sift src/build_graph.cu

build_graph_toy: src/build_graph.cu src/common.h src/device_funcs.cu
	$(NVCC) $(NVCC_FLAGS) -D DIM=2 -o build_graph_toy src/build_graph.cu

data_sift: build_graph_sift
	$(PYTHON) tools/gen_sift_workload.py --config config_sift.json

data_toy: build_graph_toy
	$(PYTHON) tests_2d/gen_toy_data.py --config config_toy.json

run_sift: sift data_sift
	$(PYTHON) run_engine.py --config config_sift.json --exe ./engine_sift
	$(PYTHON) tools/vis_sift.py --config config_sift.json

data_custom_sift: build_graph_sift
	$(PYTHON) tools/gen_sift_workload.py --config config_custom_sift.json

run_custom_sift: sift data_custom_sift
	$(PYTHON) run_engine.py --config config_custom_sift.json --exe ./engine_sift
	$(PYTHON) tools/vis_sift.py --config config_custom_sift.json

run_toy: toy data_toy
	$(PYTHON) run_engine.py --config config_toy.json --exe ./engine_toy
	$(PYTHON) tests_2d/vis_toy.py --config config_toy.json

clean:
	rm -f engine_sift engine_toy engine_vis build_graph_sift build_graph_toy
	rm -rf data/* vis_outputs/*

vis_toy: src/engine_vis.cu src/common.h src/device_funcs.cu
	$(NVCC) $(NVCC_FLAGS) -D DIM=2 -o engine_vis src/engine_vis.cu

data_vis: build_graph_toy
	$(PYTHON) tests_2d/gen_toy_data.py --config config_vis.json

run_vis_toy: vis_toy data_vis
	$(PYTHON) run_engine.py --config config_vis.json --exe ./engine_vis
	$(PYTHON) tests_2d/vis_step_by_step.py --config config_vis.json

run_all:
	@echo "========================================"
	@echo "Running Toy Benchmark (2D)"
	@echo "========================================"
	$(MAKE) run_toy
	@echo ""
	@echo "========================================"
	@echo "Running Toy Visualization (GIF)"
	@echo "========================================"
	$(MAKE) run_vis_toy
	@echo ""
	@echo "========================================"
	@echo "Running SIFT Benchmark (Full 10k)"
	@echo "========================================"
	$(MAKE) run_sift
	@echo ""
	@echo "========================================"
	@echo "Running Custom SIFT Benchmark (5k)"
	@echo "========================================"
	$(MAKE) run_custom_sift
	@echo ""
	@echo "All benchmarks completed successfully."

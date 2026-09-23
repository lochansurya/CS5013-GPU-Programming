#!/usr/bin/env bash

# Helper function to print and run a command
run_cmd() {
    echo -e "\e[1;31m[ $* ]\e[0m"  # bold red
    eval "$@"                      # execute command
    echo -e ""
}

run_cmd cd vamana_old
mkdir build
run_cmd make
run_cmd compute-sanitizer --tool memcheck ./bin/vamana ../data/sift10k/sift10k_randomgraph.bin _ ./vout/vamana.out
# run_cmd python3 scripts/bang-preprocess.py vout/vamana.out test_data/test_sift10k_index
# run_cmd cd ../bang_exact/build
# run_cmd ./bang_exact _ _ ../../vamana_old/test_data/test_sift10k_index_disk.bin ../../data/sift10k/siftsmall_query.bin _ _ ../../data/sift10k/sift10k_groundtruth.bin 100 1 256 512 256 10 8 1

#!/usr/bin/env bash

# make

cd build/

./bang_exact \
. \
. \
../../data/sift10k/sift10k_index_disk.bin \
../../data/sift10k/siftsmall_query.bin \
. \
. \
../../data/sift10k/sift10k_groundtruth.bin \
100 \
1 \
256 \
512 \
256 \
10 \
8 \
1

#!/bin/sh

rm -f output.txt

for L in 40 #10 20 30 40 60 80 120 160
do
    ./compile.sh SIFT10K "$L" 64

    
    ./build/bang_exact \
        . \
        . \
        ../data/sift10k/sift10k_index_disk.bin \
        ../data/sift10k/siftsmall_query.bin \
        . \
        . \
        ../data/sift10k/sift10k_groundtruth.bin \
        400 1 256 512 256 10 8 1 << EOM >> output.txt
y
y
y
y
y
EOM

done

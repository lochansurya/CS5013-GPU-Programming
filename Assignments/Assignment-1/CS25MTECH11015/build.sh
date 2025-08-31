#!/usr/bin/env bash
# ==========================
# Master Driver Script
# ==========================

# Usage: ./master.sh [1-5]
# 1 -> Problem 1.1, 2 -> Problem 1.2, 3 -> Problem 2, 4 -> Problem 3, 5 -> Problem 4
# No argument -> run all

RUN_ALL=1
if [ $# -eq 1 ]; then
    ARG="$1"
    if ! [[ "$ARG" =~ ^[1-5]$ ]]; then
        echo "Error: Argument must be an integer between 1 and 5"
        exit 1
    fi
    RUN_ALL=0
fi

echo "=========================="
echo "Building all binaries..."
make all
echo "Build completed."
echo "=========================="



# Create results directory
mkdir -p results

# ==========================
# Problem 1.1
# ==========================
if [ "$RUN_ALL" -eq 1 ] || [ "$ARG" -eq 1 ]; then
    echo "Running Problem-1_1 tests..."
    ./problem_1_1.sh
    echo "Problem-1_1 completed."
    echo "=========================="
fi

# ==========================
# Problem 1.2
# ==========================
if [ "$RUN_ALL" -eq 1 ] || [ "$ARG" -eq 2 ]; then
    echo "Running Problem-1_2 tests..."
    ./problem_1_2.sh
    echo "Problem-1_2 completed."
    echo "=========================="
fi

# ==========================
# Problem 2
# ==========================
if [ "$RUN_ALL" -eq 1 ] || [ "$ARG" -eq 3 ]; then
    echo "Running Problem-2 tests..."
    ./problem_2.sh
    echo "Problem-2 completed."
    echo "=========================="
fi

# ==========================
# Problem 3
# ==========================
if [ "$RUN_ALL" -eq 1 ] || [ "$ARG" -eq 4 ]; then
    echo "Running Problem-3 tests..."
    ./problem_3.sh
    echo "Problem-3 completed."
    echo "=========================="
fi

# ==========================
# Problem 4
# ==========================
if [ "$RUN_ALL" -eq 1 ] || [ "$ARG" -eq 5 ]; then
    echo "Running Problem-4 tests..."
    ./problem_4.sh
    echo "Problem-4 completed."
    echo "=========================="
fi

# ==========================
# Clean binaries
# ==========================
echo "Cleaning all binaries..."
make clean
echo "All done."
echo "=========================="

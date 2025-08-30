#!/bin/bash
# ==========================
# Master Driver Script
# ==========================

echo "=========================="
echo "Building all binaries..."
make all
echo "Build completed."
echo "=========================="

echo "=========================="
echo "Generating the matrices for matrix-matrix multiplication..."
python3 matrices_generator.py
echo "=========================="

echo "=========================="
echo "Generating the matrices for matrix transpose..."
python3 transpose_generator.py
echo "=========================="

# Create results directory
mkdir -p results

# ==========================
# Run Problem 1.1
# ==========================
echo "Running Problem-1_1 tests..."
./problem_1_1.sh
echo "Problem-1_1 completed."
echo "=========================="

# ==========================
# Run Problem 1.2
# ==========================
echo "Running Problem-1_2 tests..."
./problem_1_2.sh
echo "Problem-1_2 completed."
echo "=========================="

# ==========================
# Run Problem 2
# ==========================
echo "Running Problem-2 tests..."
./problem_2.sh
echo "Problem-2 completed."
echo "=========================="

# ==========================
# Run Problem 3
# ==========================
echo "Running Problem-3 tests..."
./problem_3.sh
echo "Problem-3 completed."
echo "=========================="

# ==========================
# Run Problem 4
# ==========================
echo "Running Problem-4 tests..."
./problem_4.sh
echo "Problem-4 completed."
echo "=========================="

# ==========================
# Clean binaries
# ==========================
echo "Cleaning all binaries..."
make clean
echo "All done."
echo "=========================="

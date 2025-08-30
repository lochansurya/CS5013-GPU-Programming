#!/usr/bin/env python
import numpy as np
import os

def generate_int32_matrix(rows, cols, filename):
    """
    Generates an int32 matrix and saves it to CSV.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    matrix = np.random.randint(-5, 6, size=(rows, cols), dtype=np.int32)
    np.savetxt(filename, matrix, fmt='%d', delimiter=',')
    print(f"Matrix saved to {filename} ({rows}x{cols})")
    return matrix

def generate_rectangular_matrices(output_dir="matrices", max_entries=10000):
    """
    Generates several rectangular matrices for multiplication.
    Ensures total entries do not exceed max_entries.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Example rectangle sizes (rows x cols)
    sizes = [(64, 64), (128, 128), (10000, 10000), (100, 50), (50, 200), (150, 60), (60, 150)]
    
    for i, (rows, cols) in enumerate(sizes, start=1):
        if rows * cols > max_entries:
            factor = (max_entries / (rows * cols)) ** 0.5
            rows = max(1, int(rows * factor))
            cols = max(1, int(cols * factor))
        
        # Generate matrix A
        A = generate_int32_matrix(rows, cols, f"{output_dir}/matrix_{i}a.csv")
        # Generate compatible matrix B for multiplication
        B = generate_int32_matrix(cols, rows, f"{output_dir}/matrix_{i}b.csv")
        # Compute expected output
        C = np.matmul(A, B).astype(np.int32)
        np.savetxt(f"{output_dir}/matrix_{i}c_expected.csv", C, fmt='%d', delimiter=',')
        print(f"Expected output saved to {output_dir}/matrix_{i}c_expected.csv")

if __name__ == "__main__":
    generate_rectangular_matrices()


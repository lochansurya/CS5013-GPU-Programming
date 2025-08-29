#!/usr/bin/env python
import numpy as np
import os

# ==========================
# Configuration
# ==========================
num_matrices = 3  # how many matrices to generate
rows_cols_list = [
    (1024, 1024),
    (2048, 512),
    (512, 2048)
]  # list of (rows, cols) for each matrix

input_dir = "matrices"
transpose_dir = os.path.join(input_dir, "transpose")

# ==========================
# Create directories
# ==========================
os.makedirs(input_dir, exist_ok=True)
os.makedirs(transpose_dir, exist_ok=True)

# ==========================
# Generate matrices
# ==========================
for idx, (rows, cols) in enumerate(rows_cols_list, start=1):
    matrix_name = f"matrix_{idx}a"
    
    print(f"Generating {matrix_name} of size ({rows}, {cols}) ...")
    
    # Generate random int32 matrix
    A = np.random.randint(-1000, 1000, size=(rows, cols), dtype=np.int32)
    
    # Save original matrix
    input_csv_path = os.path.join(transpose_dir, f"{matrix_name}.csv")
    np.savetxt(input_csv_path, A, fmt="%d", delimiter=',')
    
    # Compute transpose
    A_T = A.T
    transpose_csv_path = os.path.join(transpose_dir, f"{matrix_name}_T.csv")
    np.savetxt(transpose_csv_path, A_T, fmt="%d", delimiter=',')
    
    print(f"Saved original: {input_csv_path}")
    print(f"Saved transpose: {transpose_csv_path}")


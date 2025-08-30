#!/usr/bin/env python3
import numpy as np
import sys

def transpose_csv(input_csv, output_csv="output_matrix_transpose.csv"):
    try:
        # Load the matrix with int32 datatype
        matrix = np.loadtxt(input_csv, delimiter=',', dtype=np.int32)
    except Exception as e:
        print(f"Error reading '{input_csv}': {e}")
        sys.exit(1)

    # Compute the transpose
    matrix_T = matrix.T

    try:
        # Save the transposed matrix
        np.savetxt(output_csv, matrix_T, delimiter=',', fmt='%d')
        print(f"Transposed matrix saved to '{output_csv}'")
    except Exception as e:
        print(f"Error writing '{output_csv}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    transpose_csv(input_csv)

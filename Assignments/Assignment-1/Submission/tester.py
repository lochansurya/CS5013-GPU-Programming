#!/usr/bin/env python
import numpy as np
import sys

def compare_matrices(output_csv, expected_csv):
    """
    Compares two CSV files containing int32 matrices.
    Prints if they match exactly or details differences.
    """
    output_matrix = np.loadtxt(output_csv, delimiter=',', dtype=np.int32)
    expected_matrix = np.loadtxt(expected_csv, delimiter=',', dtype=np.int32)

    if output_matrix.shape != expected_matrix.shape:
        print(f"Dimension mismatch: output {output_matrix.shape} vs expected {expected_matrix.shape}")
        return False

    if np.array_equal(output_matrix, expected_matrix):
        print("Matrices match exactly.")
        return True
    else:
        diff = output_matrix - expected_matrix
        non_zero_indices = np.argwhere(diff != 0)
        print(f"Matrices differ at {non_zero_indices.shape[0]} element(s).")
        for idx in non_zero_indices:
            r, c = idx
            print(f"Difference at ({r}, {c}): output={output_matrix[r, c]}, expected={expected_matrix[r, c]}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_matrices.py <output_csv> <expected_csv>")
        sys.exit(1)

    output_file = sys.argv[1]
    expected_file = sys.argv[2]

    compare_matrices(output_file, expected_file)


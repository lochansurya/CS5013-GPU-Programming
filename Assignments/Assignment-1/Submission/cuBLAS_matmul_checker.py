#!/usr/bin/env python3
import cupy as cp
import numpy as np

def gpu_matmul_uint32(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Perform matrix multiplication on large uint32_t matrices using cuBLAS (via CuPy).
    
    Args:
        A (np.ndarray): Left matrix of shape (m, k), dtype=uint32
        B (np.ndarray): Right matrix of shape (k, n), dtype=uint32

    Returns:
        np.ndarray: Result matrix of shape (m, n), dtype=uint32
    """
    assert A.dtype == np.uint32 and B.dtype == np.uint32, "Inputs must be uint32"

    # Transfer to GPU (cast to float32 for cuBLAS)
    A_gpu = cp.asarray(A, dtype=cp.float32)
    B_gpu = cp.asarray(B, dtype=cp.float32)

    # cuBLAS GEMM (through CuPy)
    C_gpu = cp.matmul(A_gpu, B_gpu)

    # Cast result back to uint32
    return cp.asnumpy(C_gpu.astype(np.uint32))


if __name__ == "__main__":
    # Example: Large matrices (adjust size depending on GPU memory)
    m, k, n = 1024, 1024, 1024  # ces
    A = np.random.randint(0, 100, size=(m, k), dtype=np.uint32)
    B = np.random.randint(0, 100, size=(k, n), dtype=np.uint32)

    # GPU multiplication
    print("Performing GPU matrix multiplication...")
    C = gpu_matmul_uint32(A, B)

    print("Result shape:", C.shape)
    print("Result dtype:", C.dtype)
    print("Sample output [0:5, 0:5]:\n", C[:5, :5])

    # Export result to CSV with Unix-style newlines (\n)
    out_file = "matrix_result.csv"
    with open(out_file, "w", newline="\n") as f:
        np.savetxt(f, C, fmt="%u", delimiter=",")

    print(f"Result exported to {out_file}")
change sizes as needed

    # Generate random uint32 matri

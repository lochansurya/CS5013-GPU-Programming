// naive matrix transpose using 1D grid dimensions
#include "matrix.h"
#include "matrix_csv.h"
#include <stdio.h>
#include <cuda_runtime>


__global__ void matmul_1d_dkernel(Matrix* 

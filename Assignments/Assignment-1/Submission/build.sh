#!/usr/bin/env bash

#./matmul_1d <M> <N> <path to matrix_a.csv> <path to matrix_b.csv>
# Product Matrix of size 123 stored as matrix_c.csv
# Kernel execution time: 456 microseconds


####
#Report:
# | TrialNo | #Thread Blocks(M) | #Threads in a block(N) | KernelExecution time(microsecs) | Your insights on the execution time|
####









# ./matmul_2d <X1> <Y1> <X2> <Y2> <path to matrix_a.csv>
# matrix_b.csv>
# Product Matrix of size 123 stored as matrix_c.csv
# Kernel execution time: 456 microseconds

####
# Report
# # | TrialNo |2D Grid Size(X1, Y1) | 2D Block Size(X2, Y2) | KernelExecution time(microsecs) | Your insights on the execution time|
####

# Optimal Kernel launch Parameters are to be set from experimenting main_1_2.cu
# ./matmul_tiled
# <TILE_WIDTH>
# <path_to_matrix_a.csv>
# <path_to_matrix_b.csv>
# Product Matrix of size 123 stored as matrix_c.csv
# Kernel execution time: 456 microseconds



####
# Report
# # | TrialNo | #Global Memory Accesses| #Shared Memory Accesse | KernelExecution time(microsecs) | Your insights on the execution time|
####




# ./mattrans_basic <M> <N> <path to matrix_a.csv>
# Transpose Matrix of size 123 stored as matrix_c.csv
# Kernel execution time: 456 microseconds


####
#Report:
# | TrialNo | #Thread Blocks(M) | #Threads in a block(N) | KernelExecution time(microsecs) | Your insights on the execution time|
####



# ./mattrans_tiled <TILE_WIDTH> <path to matrix_a.csv>
# Transpose Matrix of size 123 stored as matrix_a_trans.csv
# Kernel execution time: 456 microsecond

####
# Report
# # | TrialNo | KernelExecution time(microsecs) | Your insights on the execution time|
#Try using the entire shared memory (64KB)
####


#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS_PER_BLOCK 1024

// ------------------- Device Helper -------------------
__device__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// ------------------- Device Kernel -------------------
__global__ void sort(int *arr, unsigned int start, unsigned int len,
                     unsigned int distance, bool ascending, size_t n) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x + start;
    if (i + distance < start + len && i + distance < n) {
        if ((ascending && arr[i] > arr[i + distance]) ||
            (!ascending && arr[i] < arr[i + distance])) {
            swap(&arr[i], &arr[i + distance]);
        }
    }
}

// ------------------- Host Bitonic Sort -------------------
size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

void bitonic_sort(int *input, size_t n) {
    size_t m = next_pow2(n);

    int *padded = (int *)malloc(m * sizeof(int));
    memcpy(padded, input, n * sizeof(int));

    // pad with large numbers for ascending sort
    for (size_t i = n; i < m; i++) {
        padded[i] = INT_MAX;
    }

    int *d_input;
    cudaMalloc((void **) &d_input, m * sizeof(int));
    cudaMemcpy(d_input, padded, m * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(NUM_THREADS_PER_BLOCK);
    dim3 blocks((m + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK);

    for (unsigned int len = 2; len <= m; len *= 2) {
        for (unsigned int start = 0; start < m; start += len) {
            bool ascending = !((start / len) & 1);
            for (unsigned int distance = len / 2; distance > 0; distance /= 2) {
                sort<<<blocks, threads>>>(d_input, start, len, distance, ascending, m);
                cudaDeviceSynchronize();
            }
        }
    }

    cudaMemcpy(padded, d_input, m * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    // copy back only the original size
    memcpy(input, padded, n * sizeof(int));
    free(padded);
}

// ------------------- CSV Reader/Writer -------------------
extern "C"
void read_and_sort_and_write_to_csv_int32(int *arr, const char *csv_int32_file_path) {
    FILE *fp_in = fopen(csv_int32_file_path, "r");
    if (!fp_in) {
        printf("Input File Opening Error!\n");
        return;
    }
    FILE *fp_out = fopen("input_output.csv", "w");
    if (!fp_out) {
        printf("Output File Opening Error!\n");
        fclose(fp_in);
        return;
    }

    char *line = NULL;
    size_t len = 0;
    while (getline(&line, &len, fp_in) != -1) {
        char *token = strtok(line, ",");
        if (!token) continue;
        size_t L = atoi(token);

        for (size_t i = 0; i < L; ++i) {
            token = strtok(NULL, ",");
            if (token)
                arr[i] = atoi(token);
        }

        // GPU sort
        bitonic_sort(arr, L);

        // Write output
        fprintf(fp_out, "%zu,", L);
        for (size_t j = 0; j < L; ++j) {
            fprintf(fp_out, "%d", arr[j]);
            if (j != L - 1) fprintf(fp_out, ",");
        }
        fprintf(fp_out, "\n");
    }

    free(line);
    fclose(fp_in);
    fclose(fp_out);
}

// ------------------- Main -------------------
int32_t main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: <executable> <input_csv_filepath>\n");
        return 0;
    }

    const char *input_csv_file_path = argv[1];
    int *arr = (int *) malloc((1 << 20) * sizeof(int)); // buffer for ~1M ints

    read_and_sort_and_write_to_csv_int32(arr, input_csv_file_path);

    free(arr);
    return 0;
}

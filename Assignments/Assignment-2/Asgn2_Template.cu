// #include <chrono>
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

int write_file(string filename, char *data, int size) {
    int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        return -1;
    int written = 0;
    while (written < size) {
        int ret = write(fd, data + written, size - written);
        if (ret < 0) {
            close(fd);
            return -1;
        }
        written += ret;
    }
    close(fd);
    return 0;
}

char *read_file(string &filename, int *out_size) {
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f)
        return NULL;

    struct stat st;
    if (stat(filename.c_str(), &st) != 0) {
        fclose(f);
        return NULL;
    }
    *out_size = st.st_size;

    char *buffer = (char *)malloc(*out_size);
    if (!buffer) {
        fclose(f);
        return NULL;
    }

    size_t read_bytes = fread(buffer, 1, *out_size, f);
    fclose(f);

    if (read_bytes != *out_size) {
        free(buffer);
        return NULL;
    }

    return buffer;
}

__global__ void gpu_sort(char *contents, int filesize, int L, ){
    //complete the function definition
    
}




int main(int argc, char **argv) {

    if (argc < 4 || string(argv[2]) != "-o") {
        string command;
        for (int i = 0; i < argc; i++) {
            command += argv[i];
            command += " ";
        }
        cerr << "Usage: ./main <path_to_the_input_file>.csv -o "
                "<output_file_name>.csv . But command run was: "
             << command << "\n";
        return 1;
    }
    string input_filepath = argv[1];
    int L = 128;
    string output_filename = argv[3];
    int filesize;
    char *contents = read_file(input_filepath, &filesize);


    const int NUM_WARPS; // Define 
    const int WARPS_PER_BLOCK; // Define 

    char *contents_gpu;
    
    cudaMalloc(&contents_gpu, filesize);
    cudaMemcpy(contents_gpu, contents, filesize, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gpu_sort<<< , >>>(contents_gpu, filesize, L, ); //Complete

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;
        exit(1);
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f us\n", ms * 1000);

    cudaMemcpy(contents, contents_gpu, filesize, cudaMemcpyDeviceToHost);
    write_file(output_filename, contents, filesize);

}

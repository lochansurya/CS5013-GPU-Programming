#include <stdio.h>
#include <string.h>
#include <iostream>
#include <stdint.h>

void sort(int *arr, unsigned int start, unsigned int len,
          unsigned int distance, bool ascending, size_t n){
    for(unsigned int i = start; i + distance < start + len && i + distance < n; ++i){
        if((ascending && arr[i] > arr[i + distance]) ||
           (!ascending && arr[i] < arr[i + distance]))
            std::swap(arr[i], arr[i + distance]);
    }
}


void bitonic_sort(int *arr, size_t n){
    for(unsigned int len = 2; len <= n; len *= 2){
        for(unsigned int start = 0; start < n; start += len){
            bool ascending = !((start / len) & 1);
            for(unsigned int distance = len / 2; distance > 0 ; distance /= 2){
                sort(arr, start, len, distance, ascending, n);
            }
        }

    }
}


void read__from_sort_and_write_to_csv_int32(int *arr, const char *csv_int32_file_path){
    FILE *fp_in = fopen(csv_int32_file_path, "r");    
    if(!fp_in){
        printf("Input File Opening Error!\n");
        return;
    }
    FILE* fp_out = fopen("input_output.csv", "w");
    if(!fp_out){
        printf("Output File Opening Error!\n");
        return;
    }

    // First, count the total number of arrays in the input csv file i.e, L
    size_t L = 0;
    char *line = NULL;
    size_t len = 0;
    while(getline(&line, &len, fp_in) != -1){
        char* temp = line;
        char* token = strtok(temp, ",");
        L = atoi(token);
        fprintf(fp_out, "%zu ", L);
        size_t i = 0;
        while(i < L){
            token = strtok(NULL, ",");
            arr[i] = atoi(token);
            ++i;
        }

        bitonic_sort(arr, L);
        for(size_t j = 0; j < L; ++j){
            fprintf(fp_out, "%d", arr[j]);
            if(j != L - 1) fprintf(fp_out, ",");
        }
        fprintf(fp_out, "\n");
    }
    free(line);
    fclose(fp_in);
    fclose(fp_out);}
    // The input csv file contains multiple arrays of integers (int32_t)

int32_t main(int argc, char *argv[]){
    // usage : <executable> <input_csv_file>
    if(argc != 2){
        printf("Usage: <executable> <input_csv_filepath>");
        return 0;
    }

    const char *input_csv_file_path = argv[1];

    int *arr = (int *)malloc( (1 << 20) * sizeof(int));
    read__from_sort_and_write_to_csv_int32(arr, input_csv_file_path);
    free(arr);
    return 0;
}

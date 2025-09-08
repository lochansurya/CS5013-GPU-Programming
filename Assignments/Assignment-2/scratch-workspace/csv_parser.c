#include "csv_parser.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>



Arrays* read_from_csv_uint32(const char *input_csv_file_path){
    Arrays* arrays = (Arrays*)malloc(sizeof(Arrays));
    FILE *fp_in = fopen(input_csv_file_path, "r");
    if(!fp_in){
        printf("File Opening Error\n");
        return NULL;
    }
    
    // upper bounds on the number of arrays N, and the length of an array L
    const size_t N = 1 << 15; // >= 30K
    const size_t L = 1 << 7; // 128


    //preallocate and initialize the arrays struct members
    arrays->arr = malloc(N * L * sizeof( uint32_t));
    arrays->offsets = malloc((N + 1) * sizeof( uint32_t));

    if(!arrays->arr || !arrays->offsets){
        printf("Memory allocation for the struct Arrays failed!\n");
        // delete in the LIFO order
        free(arrays->offsets);
        free(arrays->arr);
        free(arrays);
        fclose(fp_in);
        return NULL;
    }

    arrays->num_arrays = 0;
    arrays->total_len = 0;
    arrays->offsets[0] = 0;
   

    char *lineptr = NULL;
    size_t n = 0;
    while(getline(&lineptr, &n, fp_in) != -1 ){
        char *token = strtok(lineptr, ",");
        if(!token) continue; // skip the blank lines
        size_t len = atoi(token); // read in the array size
        size_t base = arrays->total_len;

        // record the start offset
        arrays->offsets[arrays->num_arrays] = base;
        
        // start reading the elements
        for(size_t i = 0; i < len; ++i){
            token = strtok(NULL, ",");
            arrays->arr[base + i] = token ? (uint32_t)strtoul(token, NULL, 10): 0;
        }
        
        arrays->total_len += len;
        arrays->num_arrays++;
    }

    // offset sentinel
    arrays->offsets[arrays->num_arrays] = arrays->total_len;

    free(lineptr);
    fclose(fp_in);
    
    return arrays;

}


void write_to_csv_uint32(const char *output_csv_file_path, const Arrays *arrays){
    FILE *fp_out = fopen(output_csv_file_path, "w");
    if(!fp_out){
        printf("File Opening Error\n");
        return ;
    }

    for(uint32_t i = 0; i < arrays->num_arrays; ++i){
        // based on the offsets, write each array
        size_t start = arrays->offsets[i];
        size_t end = arrays->offsets[i+1];
        size_t arr_len = end - start; 
         
        // first, write the array length to the output csv file
        fprintf(fp_out, "%zu", arr_len);
        for(size_t j = start; j < end; ++j){
            fprintf(fp_out, ",%u", arrays->arr[j]); 
        }
        fprintf(fp_out, "\n");
    }
    fclose(fp_out);
}

void free_arrays(Arrays *arrays){
    if(!arrays) return;
    free(arrays->arr);
    free(arrays->offsets);
    free(arrays);
}

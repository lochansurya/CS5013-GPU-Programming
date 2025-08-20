#include "matrix_csv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define MAX_BUF_SIZE 1024

void matrix_read_from_csv_int32(int32_t*** matrix, const char *file_path, size_t* num_rows, size_t* num_cols){
    FILE* fp = fopen(file_path, "r"); // read in text mode by default => char by char scanning
    *num_rows = *num_cols = 0;
    if(!fp){
        perror("File Opening Error\n");
        return;
    }

    char line[MAX_BUF_SIZE];

    // infer the number of rows and columns in the matrix
    while(fgets(line, sizeof(line), fp)){
        (*num_rows)++;
        if(*num_rows == 1){
            char *tmp = strdup(line);
            char *token = strtok(tmp, ",\n");
            while(token){
                (*num_cols)++;
                token = strtok(NULL, ",\n");
            }
            free(tmp);
        }
    }

    // Allocate the memory for the matrix
    int32_t **tmp = malloc(*num_rows * sizeof(int*));    

    for(unsigned int row= 0; row < *num_rows; ++row){
        tmp[row] = malloc(*num_cols * sizeof(int));
    }
    *matrix = tmp;


    // reset the file pointer
    rewind(fp);
    

    // read the data now
    unsigned int row = 0;
    while(fgets(line, sizeof(line), fp)){
        unsigned int col = 0;
        char *token = strtok(line, ",\n"); // the second argument is a set of chars concatenated, each character is treated asa delimiter; can consider as a class of delimiter chars
        while(token){
            (*matrix)[row][col] = atoi(token);
            col++;
            token = strtok(NULL, ",\n"); // reset the token to NULL to avoid clobbering
        }                                            
        row++;
    } 


    fclose(fp);
}



void matrix_write_to_csv_int32(int32_t*** matrix, const char *file_path, size_t* num_rows, size_t* num_cols){
    FILE* fp = fopen(file_path, "w");
    if(!fp){
        perror("File Opening Error\n");
        return;
    }

    for(size_t row = 0; row < *num_rows; ++row){
        for(size_t col = 0; col < *num_cols; ++col){
            fprintf(fp, "%d", (*matrix)[row][col]);
            if(col < *num_cols - 1){
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

// print the matrix
void print_matrix(int32_t** matrix, size_t num_rows, size_t num_cols){
    size_t row = 0;
    printf("matrix:\n[");
    for(; row < num_rows; ++row){
        unsigned int col = 0;
        for(; col < num_cols; ++col){
           printf("%d ", matrix[row][col]); 
        }
        printf("\n");
    }
    printf("]\n");
}

// Commenting the main to use this file as a library
//int32_t main(int32_t argc, char* argv[]){
//    // Expecting a CSV file as the command line arg
//    const char *file_path = argv[1];
//    if(!file_path){
//        perror("File opening error!\n");
//        return 0;
//    }
//
//    size_t num_rows = 0, num_cols = 0;
//    int32_t **matrix;
//    matrix_read_from_csv_int32(&matrix, file_path, &num_rows, &num_cols);
//    printf("Printing the matrix read from the input csv file path...\n");
//    print_matrix(matrix, num_rows, num_cols);
//    matrix_write_to_csv_int32(&matrix, file_path, &num_rows, &num_cols);
//
//}

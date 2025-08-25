#include "matrix_csv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define MAX_BUF_SIZE 1024

// Read CSV into a Matrix struct
void matrix_read_from_csv_int32(Matrix* mat, const char *file_path) {
    FILE* fp = fopen(file_path, "r");
    if (!fp) {
        perror("File Opening Error");
        return;
    }

    char line[MAX_BUF_SIZE];
    mat->num_rows = mat->num_cols = 0;

    // First pass: count rows and columns
    while (fgets(line, sizeof(line), fp)) {
        mat->num_rows++;
        if (mat->num_rows == 1) {
            char *tmp = strdup(line);
            char *token = strtok(tmp, ",\n");
            while (token) {
                mat->num_cols++;
                token = strtok(NULL, ",\n");
            }
            free(tmp);
        }
    }

    // Allocate linear row-major array
    mat->elements = malloc(mat->num_rows * mat->num_cols * sizeof(int32_t));
    if (!mat->elements) {
        perror("Memory Allocation Error");
        fclose(fp);
        return;
    }

    // Reset file pointer for reading data
    rewind(fp);

    unsigned int row = 0;
    while (fgets(line, sizeof(line), fp)) {
        unsigned int col = 0;
        char *token = strtok(line, ",\n");
        while (token) {
            mat->elements[row * mat->num_cols + col] = atoi(token);
            col++;
            token = strtok(NULL, ",\n");
        }
        row++;
    }

    fclose(fp);
}

// Write Matrix struct to CSV
void matrix_write_to_csv_int32(Matrix* mat, const char *file_path) {
    FILE* fp = fopen(file_path, "w");
    if (!fp) {
        perror("File Opening Error");
        return;
    }

    for (unsigned int row = 0; row < mat->num_rows; ++row) {
        for (unsigned int col = 0; col < mat->num_cols; ++col) {
            fprintf(fp, "%d", mat->elements[row * mat->num_cols + col]);
            if (col < mat->num_cols - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

// Print matrix
void print_matrix( Matrix* mat) {
    printf("matrix:\n[");
    for (unsigned int row = 0; row < mat->num_rows; ++row) {
        for (unsigned int col = 0; col < mat->num_cols; ++col) {
            printf("%d ", mat->elements[row * mat->num_cols + col]);
        }
        printf("\n");
    }
    printf("]\n");
}


//commenting out the main to use this C source file as a library
// int main(int argc, char* argv[]){
//     // Expecting a CSV file as the command line argument
//     if(argc < 2){
//         fprintf(stderr, "Usage: %s <csv_file_path>\n", argv[0]);
//         return 1;
//     }

//     const char *file_path = argv[1];

//     Matrix mat = {0, 0, NULL};

//     // Read CSV into the Matrix
//     matrix_read_from_csv(&mat, file_path);

//     printf("Printing the matrix read from the input CSV file...\n");
//     print_matrix(&mat);

//     // Optionally write it back to the same CSV file
//     matrix_write_to_csv(&mat, file_path);

//     // Free allocated memory
//     free(mat.elements);

//     return 0;
// }
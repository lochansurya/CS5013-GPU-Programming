#ifndef __LIB_MATRIX_CSV__
#define __LIB__MATRIX_CSV__

#include <stdio.h>
#include <stdint.h>

void matrix_read_from_csv_int32(int32_t*** matrix, const char *file_path, size_t* num_rows, size_t* num_cols);
void matrix_write_to_csv_int32(int32_t*** matrix, const char *file_path, size_t* num_rows, size_t* num_cols);
void print_matrix(int32_t** matrix, size_t num_rows, size_t num_cols);


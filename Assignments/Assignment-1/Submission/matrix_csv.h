#ifndef __LIB_MATRIX_CSV__
#define __LIB_MATRIX_CSV__
#include "matrix.h"
// to avoid name mangling, which is done by the C++ compiler, for this gcc compiled source files
#ifdef __cplusplus
extern "C" {
#endif

void matrix_read_from_csv_int32(Matrix* mat, const char *file_path);
void matrix_write_to_csv_int32(Matrix* mat, const char *file_path);
void print_matrix(Matrix* mat);

#ifdef __cplusplus
}
#endif
#endif

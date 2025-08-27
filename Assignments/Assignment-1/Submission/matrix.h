#ifndef _MATRIX_
#define _MATRIX_
#include <stdint.h>


typedef struct {
    unsigned int num_cols;   // number of columns
    unsigned int num_rows;  // number of rows
    uint32_t *elements;      // linear row-major array: row * width + col
} Matrix;

#endif

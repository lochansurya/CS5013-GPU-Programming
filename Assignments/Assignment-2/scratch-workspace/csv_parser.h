#ifndef __CSV_PARSER__
#define __CSV_PARSER__
#include <stdio.h>
#include <stdint.h>

// return a heap allocated( means manual memory management => have to call free per malloc'd object) 
// flat array of unsigned integers, and another flat array of size of number of input arrays in the csv file
// containing the offsets into the flat array
// format: <a row in the input csv file> = list<<number of elements in each array>,<array>>
// num_arrays: number of arrays in the input csv file
// total_len: the length of the array arr;
struct arrays{
    uint32_t *arr;
    uint32_t *offsets;
    size_t num_arrays;
    size_t total_len;
};

typedef struct arrays Arrays;

// Destination-First APIs
Arrays* read_from_csv_uint32(const char* input_csv_file_path);
void write_to_csv_uint32(const char *output_file_path, const Arrays *arrays) ;
void free_arrays(Arrays *arrays);

#endif

#include <fstream>
#include <cstdint>
#include <vector>
#include <string>

// read a csv file of int32_t type entries into a csv file
// returns the number of entries read
size_t matrix_read(std::istream& is, std::vector<std::vector<int32_t>>& matrix_int32);

// write out to a csv file of int32_t type entries from a vector of vectors
// returns the number of entries written
size_t matrix_write(std::ostream& os, std::vector<std::vector<int32_t>>& matrix_int32_t);


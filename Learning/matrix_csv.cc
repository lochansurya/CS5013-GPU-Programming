#include "matrix_csv.h"
#include <fstream>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>

// read a csv file of int32_t type entries into a csv file
// @return (num_rows, num_cols)
// Important: it should return the number of rows and columns of the matrix, as a part of inference
std::pair<size_t, size_t> matrix_read(std::ifstream& ifs, std::vector<std::vector<int32_t>>& matrix_int32) {
    size_t num_rows_read = 0, num_entries_read = 0;
    std::string line;
    while(std::getline(ifs, line)) {
        std::vector<int32_t> row;
        num_rows_read++;
        std::stringstream ss(line);
        std::string value;
        while(std::getline(ss, value, ',')) {
            num_entries_read++;
            row.push_back(std::stoi(value));
        }
        matrix_int32.push_back(row);
    }
    return std::make_pair(num_rows_read, matrix_int32.empty() ? 0 : matrix_int32[0].size());
}

// write out to a csv file of int32_t type entries from a vector of vectors
// @return (num_row, num_cols)
// Important: it should return the number of rows and columns of the matrix, as a part
std::pair<size_t, size_t> matrix_write(std::ofstream& ofs, const std::vector<std::vector<int32_t>>& matrix_int32) {
    size_t num_rows_written = 0;
    for(const auto& row : matrix_int32) {
        if(row.empty()) continue; // skip empty rows
        num_rows_written++;
        for(size_t i = 0; i < row.size(); i++) {
            ofs << row[i];
            if(i + 1 < row.size()) ofs << ",";
        }
        ofs << "\n";
    }
    return std::make_pair(num_rows_written, matrix_int32.empty() ? 0 : matrix_int32[0].size());
}

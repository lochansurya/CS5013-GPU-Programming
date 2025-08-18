#include "matrix_csv.hpp"
#include <fstream>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

// read a csv file of int32_t type entries into a csv file
// returns the number of entries read
size_t matrix_read(std::istream& is, std::vector<std::vector<int32_t>>& matrix_int32) {
    size_t num_rows_read = 0, num_entries_read = 0;
    std::string line;
    while(std::getline(is, line)) {
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
    return num_entries_read;
}

// write out to a csv file of int32_t type entries from a vector of vectors
// returns the number of entries written
size_t matrix_write(std::ostream& os, const std::vector<std::vector<int32_t>>& matrix_int32) {
    size_t num_entries_written = 0;
    for(const auto& row : matrix_int32) {
        for(size_t i = 0; i < row.size(); i++) {
            os << row[i];
            num_entries_written++;
            if(i + 1 < row.size()) os << ",";
        }
        os << "\n";
    }
    return num_entries_written;
}

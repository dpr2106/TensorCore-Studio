#pragma once

#include "tensorcore/tensor.hpp"
#include <string>
#include <vector>

namespace tensorcore {

struct Dataset {
    Tensor features; // 2D Tensor [N, num_features]
    Tensor targets;  // 2D Tensor [N, num_targets]
    size_t num_samples = 0;
};

class CSVLoader {
public:
    struct Options {
        bool has_header = true;
        char delimiter = ',';
        std::vector<size_t> feature_columns; // Specific feature column indices. If empty, all non-target columns.
        std::vector<size_t> target_columns;  // Specific target column indices. Must not be empty.
    };

    static Dataset load_from_file(const std::string& filepath, const Options& options);
    static Dataset load_from_string(const std::string& csv_content, const Options& options);
};

} // namespace tensorcore
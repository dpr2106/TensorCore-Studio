#include "tensorcore/dataset.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <set>

namespace tensorcore {

namespace {

std::vector<std::string> parse_csv_line(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == delimiter && !in_quotes) {
            tokens.push_back(token);
            token.clear();
        } else {
            token += c;
        }
    }
    tokens.push_back(token);
    return tokens;
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

} // namespace

Dataset CSVLoader::load_from_file(const std::string& filepath, const Options& options) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + filepath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return load_from_string(buffer.str(), options);
}

Dataset CSVLoader::load_from_string(const std::string& csv_content, const Options& options) {
    if (options.target_columns.empty()) {
        throw std::invalid_argument("target_columns must not be empty.");
    }

    // Check for duplicate target columns
    std::set<size_t> target_set(options.target_columns.begin(), options.target_columns.end());
    if (target_set.size() != options.target_columns.size()) {
        throw std::invalid_argument("target_columns contains duplicate column indices.");
    }

    // Check for duplicate feature columns
    std::set<size_t> feature_set(options.feature_columns.begin(), options.feature_columns.end());
    if (feature_set.size() != options.feature_columns.size()) {
        throw std::invalid_argument("feature_columns contains duplicate column indices.");
    }

    // Check for overlap between feature and target columns
    for (size_t fc : options.feature_columns) {
        if (target_set.count(fc)) {
            throw std::invalid_argument("Feature and target columns cannot overlap: index " + std::to_string(fc));
        }
    }

    std::stringstream ss(csv_content);
    std::string line;
    size_t line_number = 0;
    size_t expected_cols = 0;

    std::vector<std::vector<float>> feature_rows;
    std::vector<std::vector<float>> target_rows;

    std::vector<size_t> final_feature_cols = options.feature_columns;

    while (std::getline(ss, line)) {
        line_number++;
        std::string trimmed = trim(line);
        if (trimmed.empty()) continue; // skip blank lines

        auto tokens = parse_csv_line(trimmed, options.delimiter);

        if (line_number == 1 && options.has_header) {
            expected_cols = tokens.size();
            // If feature_columns not specified, use all columns except target_columns
            if (final_feature_cols.empty()) {
                for (size_t c = 0; c < expected_cols; ++c) {
                    if (!target_set.count(c)) {
                        final_feature_cols.push_back(c);
                    }
                }
            }
            continue;
        }

        if (expected_cols == 0) {
            expected_cols = tokens.size();
            if (final_feature_cols.empty()) {
                for (size_t c = 0; c < expected_cols; ++c) {
                    if (!target_set.count(c)) {
                        final_feature_cols.push_back(c);
                    }
                }
            }
        }

        // Validate ragged rows
        if (tokens.size() != expected_cols) {
            throw std::invalid_argument("Ragged row at line " + std::to_string(line_number) +
                                        ": expected " + std::to_string(expected_cols) +
                                        " columns, got " + std::to_string(tokens.size()));
        }

        // Validate column bounds
        for (size_t tc : options.target_columns) {
            if (tc >= expected_cols) {
                throw std::out_of_range("Target column index " + std::to_string(tc) +
                                        " out of range (total cols: " + std::to_string(expected_cols) + ")");
            }
        }
        for (size_t fc : final_feature_cols) {
            if (fc >= expected_cols) {
                throw std::out_of_range("Feature column index " + std::to_string(fc) +
                                        " out of range (total cols: " + std::to_string(expected_cols) + ")");
            }
        }

        // Parse features
        std::vector<float> f_row;
        f_row.reserve(final_feature_cols.size());
        for (size_t fc : final_feature_cols) {
            std::string t = trim(tokens[fc]);
            try {
                f_row.push_back(std::stof(t));
            } catch (const std::exception&) {
                throw std::invalid_argument("Failed to parse numeric float for feature at line " +
                                            std::to_string(line_number) + ", column " + std::to_string(fc) + ": '" + t + "'");
            }
        }
        feature_rows.push_back(f_row);

        // Parse targets
        std::vector<float> t_row;
        t_row.reserve(options.target_columns.size());
        for (size_t tc : options.target_columns) {
            std::string t = trim(tokens[tc]);
            try {
                t_row.push_back(std::stof(t));
            } catch (const std::exception&) {
                throw std::invalid_argument("Failed to parse numeric float for target at line " +
                                            std::to_string(line_number) + ", column " + std::to_string(tc) + ": '" + t + "'");
            }
        }
        target_rows.push_back(t_row);
    }

    size_t num_samples = feature_rows.size();
    if (num_samples == 0) {
        throw std::invalid_argument("CSV contains no valid data rows.");
    }

    size_t num_features = final_feature_cols.size();
    size_t num_targets = options.target_columns.size();

    // Flatten into contiguous buffers for Tensor construction
    std::vector<float> f_flat;
    f_flat.reserve(num_samples * num_features);
    for (const auto& row : feature_rows) {
        f_flat.insert(f_flat.end(), row.begin(), row.end());
    }

    std::vector<float> t_flat;
    t_flat.reserve(num_samples * num_targets);
    for (const auto& row : target_rows) {
        t_flat.insert(t_flat.end(), row.begin(), row.end());
    }

    Dataset dataset;
    dataset.features = Tensor({num_samples, num_features}, f_flat);
    dataset.targets = Tensor({num_samples, num_targets}, t_flat);
    dataset.num_samples = num_samples;

    return dataset;
}

} // namespace tensorcore
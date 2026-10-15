#pragma once

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <functional>
#include <random>
#include <cmath>
#include <sstream>

namespace tensorcore {

using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;

inline size_t compute_size(const Shape& shape) {
    if (shape.empty()) return 0;
    size_t s = 1;
    for (size_t dim : shape) {
        s *= dim;
    }
    return s;
}

inline Strides compute_default_strides(const Shape& shape) {
    if (shape.empty()) return {};
    Strides strides(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// Computes standard NumPy-style broadcasted output shape for two shapes.
// Dimensions are aligned from the right (trailing dimensions).
// Each pair of dimensions must either be equal or one of them must be 1.
inline Shape broadcast_shapes(const Shape& s1, const Shape& s2) {
    size_t r1 = s1.size();
    size_t r2 = s2.size();
    size_t out_rank = std::max(r1, r2);
    Shape out_shape(out_rank);

    for (size_t i = 0; i < out_rank; ++i) {
        size_t d1 = (i < r1) ? s1[r1 - 1 - i] : 1;
        size_t d2 = (i < r2) ? s2[r2 - 1 - i] : 1;

        if (d1 == d2) {
            out_shape[out_rank - 1 - i] = d1;
        } else if (d1 == 1) {
            out_shape[out_rank - 1 - i] = d2;
        } else if (d2 == 1) {
            out_shape[out_rank - 1 - i] = d1;
        } else {
            std::ostringstream oss;
            oss << "Cannot broadcast shapes [";
            for (size_t j = 0; j < s1.size(); ++j) oss << s1[j] << (j + 1 < s1.size() ? ", " : "");
            oss << "] and [";
            for (size_t j = 0; j < s2.size(); ++j) oss << s2[j] << (j + 1 < s2.size() ? ", " : "");
            oss << "]: dimension mismatch at aligned index " << i << " (" << d1 << " vs " << d2 << ")";
            throw std::invalid_argument(oss.str());
        }
    }
    return out_shape;
}

class Tensor {
public:
    Tensor();
    explicit Tensor(const Shape& shape, float initial_val = 0.0f);
    Tensor(const Shape& shape, const std::vector<float>& data);

    const Shape& shape() const { return shape_; }
    const Strides& strides() const { return strides_; }
    size_t size() const { return size_; }
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;

    float& operator[](size_t index) { return data_[index]; }
    const float& operator[](size_t index) const { return data_[index]; }

    void reshape(const Shape& new_shape);
    Tensor clone() const;

    // Factory initializers
    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);
    static Tensor uniform(const Shape& shape, float low, float high, uint32_t seed = 42);
    static Tensor xavier_uniform(size_t fan_in, size_t fan_out, uint32_t seed = 42);

    // Basic arithmetic operations (with NumPy-style broadcasting support)
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const; // Element-wise
    Tensor scale(float scalar) const;
    Tensor matmul(const Tensor& other) const; // 2D Matrix multiplication
    Tensor transpose() const;                 // 2D Transpose

    // Activation functions & derivatives
    Tensor relu() const;
    Tensor relu_backward(const Tensor& grad_output) const;
    Tensor sigmoid() const;
    Tensor sigmoid_backward(const Tensor& grad_output) const;
    Tensor softmax(int axis = -1) const;

    // Reductions
    float sum() const;
    float mean() const;

    std::string to_string() const;

private:
    Shape shape_;
    Strides strides_;
    size_t size_;
    std::vector<float> data_;

    size_t get_flat_index(const std::vector<size_t>& indices) const;
};

} // namespace tensorcore
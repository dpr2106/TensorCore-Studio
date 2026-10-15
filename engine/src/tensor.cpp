#include "tensorcore/tensor.hpp"
#include <iomanip>
#include <algorithm>
#include <cassert>

namespace tensorcore {

Tensor::Tensor() : shape_{}, strides_{}, size_(0) {}

Tensor::Tensor(const Shape& shape, float initial_val)
    : shape_(shape), strides_(compute_default_strides(shape)), size_(compute_size(shape)), data_(size_, initial_val) {}

Tensor::Tensor(const Shape& shape, const std::vector<float>& data)
    : shape_(shape), strides_(compute_default_strides(shape)), size_(compute_size(shape)), data_(data) {
    if (data.size() != size_) {
        throw std::invalid_argument("Provided data size does not match shape capacity.");
    }
}

size_t Tensor::get_flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Indices rank does not match tensor rank.");
    }
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds for dimension.");
        }
        index += indices[i] * strides_[i];
    }
    return index;
}

float& Tensor::at(const std::vector<size_t>& indices) {
    return data_[get_flat_index(indices)];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    return data_[get_flat_index(indices)];
}

void Tensor::reshape(const Shape& new_shape) {
    size_t new_size = compute_size(new_shape);
    if (new_size != size_) {
        throw std::invalid_argument("Cannot reshape: element count must remain invariant.");
    }
    shape_ = new_shape;
    strides_ = compute_default_strides(new_shape);
}

Tensor Tensor::clone() const {
    return Tensor(shape_, data_);
}

Tensor Tensor::zeros(const Shape& shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::ones(const Shape& shape) {
    return Tensor(shape, 1.0f);
}

Tensor Tensor::uniform(const Shape& shape, float low, float high, uint32_t seed) {
    Tensor t(shape);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(low, high);
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = dis(gen);
    }
    return t;
}

Tensor Tensor::xavier_uniform(size_t fan_in, size_t fan_out, uint32_t seed) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    return uniform({fan_in, fan_out}, -limit, limit, seed);
}

namespace {
// Helper for binary element-wise operations with NumPy-style broadcasting support.
template <typename Op>
Tensor binary_op_broadcast(const Tensor& a, const Tensor& b, Op op) {
    // Fast path: identical shapes need zero index-translation overhead.
    if (a.shape() == b.shape()) {
        Tensor result(a.shape());
        size_t n = a.size();
        const float* da = a.data();
        const float* db = b.data();
        float* dr = result.data();
        for (size_t i = 0; i < n; ++i) {
            dr[i] = op(da[i], db[i]);
        }
        return result;
    }

    // Broadcast path: determine broadcast output shape and effective strides.
    Shape out_shape = broadcast_shapes(a.shape(), b.shape());
    size_t out_rank = out_shape.size();
    size_t out_size = compute_size(out_shape);

    if (out_size == 0) {
        return Tensor(out_shape);
    }

    // Compute effective strides for `a` and `b` relative to `out_shape`.
    // When dimension is 1 or missing, stride is 0 so the coordinate does not advance the pointer.
    Strides eff_strides_a(out_rank, 0);
    Strides eff_strides_b(out_rank, 0);

    size_t ra = a.shape().size();
    size_t rb = b.shape().size();

    for (size_t i = 0; i < out_rank; ++i) {
        size_t idx_from_right = out_rank - 1 - i;

        if (i < ra) {
            size_t a_dim = a.shape()[ra - 1 - i];
            if (a_dim != 1) {
                eff_strides_a[idx_from_right] = a.strides()[ra - 1 - i];
            }
        }
        if (i < rb) {
            size_t b_dim = b.shape()[rb - 1 - i];
            if (b_dim != 1) {
                eff_strides_b[idx_from_right] = b.strides()[rb - 1 - i];
            }
        }
    }

    Tensor result(out_shape);
    const float* da = a.data();
    const float* db = b.data();
    float* dr = result.data();

    std::vector<size_t> coord(out_rank, 0);
    size_t offset_a = 0;
    size_t offset_b = 0;

    for (size_t out_idx = 0; out_idx < out_size; ++out_idx) {
        dr[out_idx] = op(da[offset_a], db[offset_b]);

        // Increment N-dimensional coordinate and update linear offsets
        for (int d = static_cast<int>(out_rank) - 1; d >= 0; --d) {
            coord[d]++;
            offset_a += eff_strides_a[d];
            offset_b += eff_strides_b[d];

            if (coord[d] < out_shape[d]) {
                break;
            }
            // Dimension rolled over: reset coordinate and subtract cumulative stride
            offset_a -= coord[d] * eff_strides_a[d];
            offset_b -= coord[d] * eff_strides_b[d];
            coord[d] = 0;
        }
    }
    return result;
}
} // namespace

Tensor Tensor::add(const Tensor& other) const {
    return binary_op_broadcast(*this, other, [](float x, float y) { return x + y; });
}

Tensor Tensor::sub(const Tensor& other) const {
    return binary_op_broadcast(*this, other, [](float x, float y) { return x - y; });
}

Tensor Tensor::mul(const Tensor& other) const {
    return binary_op_broadcast(*this, other, [](float x, float y) { return x * y; });
}

Tensor Tensor::scale(float scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < size_; ++i) {
        result[i] = data_[i] * scalar;
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("matmul currently requires 2D matrices.");
    }
    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t K2 = other.shape_[0];
    size_t N = other.shape_[1];

    if (K != K2) {
        throw std::invalid_argument("Inner matrix dimensions must match for matmul: " +
                                    std::to_string(K) + " vs " + std::to_string(K2));
    }

    Tensor result({M, N}, 0.0f);
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = data_[i * strides_[0] + k];
            for (size_t j = 0; j < N; ++j) {
                result.data()[i * N + j] += a_ik * other.data()[k * other.strides_[0] + j];
            }
        }
    }
    return result;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Transpose currently supports 2D matrices.");
    }
    size_t M = shape_[0];
    size_t N = shape_[1];
    Tensor result({N, M});
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result.at({j, i}) = at({i, j});
        }
    }
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(shape_);
    for (size_t i = 0; i < size_; ++i) {
        result[i] = data_[i] > 0.0f ? data_[i] : 0.0f;
    }
    return result;
}

Tensor Tensor::relu_backward(const Tensor& grad_output) const {
    if (shape_ != grad_output.shape_) {
        throw std::invalid_argument("Gradient shape mismatch in relu_backward.");
    }
    Tensor grad_input(shape_);
    for (size_t i = 0; i < size_; ++i) {
        grad_input[i] = data_[i] > 0.0f ? grad_output[i] : 0.0f;
    }
    return grad_input;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape_);
    for (size_t i = 0; i < size_; ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-data_[i]));
    }
    return result;
}

Tensor Tensor::sigmoid_backward(const Tensor& grad_output) const {
    if (shape_ != grad_output.shape_) {
        throw std::invalid_argument("Gradient shape mismatch in sigmoid_backward.");
    }
    Tensor grad_input(shape_);
    for (size_t i = 0; i < size_; ++i) {
        float s = 1.0f / (1.0f + std::exp(-data_[i]));
        grad_input[i] = grad_output[i] * s * (1.0f - s);
    }
    return grad_input;
}

Tensor Tensor::softmax(int /*axis*/) const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Softmax currently implemented for 2D tensors [batch, classes].");
    }
    size_t batch = shape_[0];
    size_t classes = shape_[1];
    Tensor result(shape_);

    for (size_t b = 0; b < batch; ++b) {
        float max_val = data_[b * classes];
        for (size_t c = 1; c < classes; ++c) {
            max_val = std::max(max_val, data_[b * classes + c]);
        }
        float sum_exp = 0.0f;
        for (size_t c = 0; c < classes; ++c) {
            float exp_val = std::exp(data_[b * classes + c] - max_val);
            result[b * classes + c] = exp_val;
            sum_exp += exp_val;
        }
        for (size_t c = 0; c < classes; ++c) {
            result[b * classes + c] /= sum_exp;
        }
    }
    return result;
}

float Tensor::sum() const {
    return std::accumulate(data_.begin(), data_.end(), 0.0f);
}

float Tensor::mean() const {
    if (size_ == 0) return 0.0f;
    return sum() / static_cast<float>(size_);
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
    }
    oss << "], data=[";
    size_t preview = std::min(size_, size_t(8));
    for (size_t i = 0; i < preview; ++i) {
        oss << std::fixed << std::setprecision(4) << data_[i] << (i + 1 < preview ? ", " : "");
    }
    if (size_ > preview) oss << "... (" << size_ << " total)";
    oss << "])";
    return oss.str();
}

} // namespace tensorcore
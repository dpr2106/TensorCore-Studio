#include "tensorcore/layers.hpp"

namespace tensorcore {

Linear::Linear(size_t in_features, size_t out_features, uint32_t seed)
    : in_features_(in_features),
      out_features_(out_features),
      weights_(Tensor::xavier_uniform(in_features, out_features, seed)),
      bias_(Tensor::zeros({1, out_features})),
      grad_weights_(Tensor::zeros({in_features, out_features})),
      grad_bias_(Tensor::zeros({1, out_features})) {}

Tensor Linear::forward(const Tensor& input) {
    last_input_ = input.clone();
    // Y = X * W + b
    Tensor out = input.matmul(weights_);
    // Add broadcasted bias along rows
    size_t batch_size = out.shape()[0];
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < out_features_; ++c) {
            out.at({b, c}) += bias_[c];
        }
    }
    return out;
}

Tensor Linear::backward(const Tensor& grad_output) {
    // grad_weights = X^T * grad_output
    Tensor xt = last_input_.transpose();
    Tensor gw = xt.matmul(grad_output);
    grad_weights_ = grad_weights_.add(gw);

    // grad_bias = sum over batch dimension of grad_output
    size_t batch_size = grad_output.shape()[0];
    for (size_t c = 0; c < out_features_; ++c) {
        float sum_b = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            sum_b += grad_output.at({b, c});
        }
        grad_bias_[c] += sum_b;
    }

    // grad_input = grad_output * W^T
    Tensor wt = weights_.transpose();
    return grad_output.matmul(wt);
}

void Linear::zero_grad() {
    grad_weights_ = Tensor::zeros(grad_weights_.shape());
    grad_bias_ = Tensor::zeros(grad_bias_.shape());
}

Tensor ReLU::forward(const Tensor& input) {
    last_input_ = input.clone();
    return input.relu();
}

Tensor ReLU::backward(const Tensor& grad_output) {
    return last_input_.relu_backward(grad_output);
}

} // namespace tensorcore

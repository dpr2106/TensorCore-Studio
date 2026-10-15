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
    // Broadcast addition of bias [1, out_features] across batch rows
    return out.add(bias_);
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

// ----------------- Sequential Container ----------------- //

Sequential::Sequential(std::vector<std::shared_ptr<Layer>> layers)
    : layers_(std::move(layers)) {}

void Sequential::add(std::shared_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor current = input;
    for (auto& layer : layers_) {
        current = layer->forward(current);
    }
    return current;
}

Tensor Sequential::backward(const Tensor& grad_output) {
    Tensor current_grad = grad_output;
    // Reverse backward propagation
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        current_grad = layers_[i]->backward(current_grad);
    }
    return current_grad;
}

std::vector<Tensor*> Sequential::parameters() {
    std::vector<Tensor*> all_params;
    for (auto& layer : layers_) {
        auto params = layer->parameters();
        all_params.insert(all_params.end(), params.begin(), params.end());
    }
    return all_params;
}

std::vector<Tensor*> Sequential::gradients() {
    std::vector<Tensor*> all_grads;
    for (auto& layer : layers_) {
        auto grads = layer->gradients();
        all_grads.insert(all_grads.end(), grads.begin(), grads.end());
    }
    return all_grads;
}

void Sequential::zero_grad() {
    for (auto& layer : layers_) {
        layer->zero_grad();
    }
}

} // namespace tensorcore
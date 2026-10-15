#pragma once

#include "tensorcore/tensor.hpp"
#include "tensorcore/layers.hpp"
#include <vector>

namespace tensorcore {

class SGDOptimizer {
public:
    explicit SGDOptimizer(float learning_rate = 0.01f, float momentum = 0.9f)
        : lr_(learning_rate), momentum_(momentum) {}

    void step(std::vector<Linear*>& layers) {
        if (velocity_w_.size() != layers.size()) {
            velocity_w_.clear();
            velocity_b_.clear();
            for (auto* layer : layers) {
                velocity_w_.push_back(Tensor::zeros(layer->weights().shape()));
                velocity_b_.push_back(Tensor::zeros(layer->bias().shape()));
            }
        }

        for (size_t i = 0; i < layers.size(); ++i) {
            auto* layer = layers[i];
            // v = momentum * v + lr * grad
            for (size_t j = 0; j < layer->weights().size(); ++j) {
                velocity_w_[i][j] = momentum_ * velocity_w_[i][j] + lr_ * layer->grad_weights()[j];
                layer->weights()[j] -= velocity_w_[i][j];
            }
            for (size_t j = 0; j < layer->bias().size(); ++j) {
                velocity_b_[i][j] = momentum_ * velocity_b_[i][j] + lr_ * layer->grad_bias()[j];
                layer->bias()[j] -= velocity_b_[i][j];
            }
        }
    }

private:
    float lr_;
    float momentum_;
    std::vector<Tensor> velocity_w_;
    std::vector<Tensor> velocity_b_;
};

} // namespace tensorcore

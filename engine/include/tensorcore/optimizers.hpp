#pragma once

#include "tensorcore/tensor.hpp"
#include "tensorcore/layers.hpp"
#include <vector>

namespace tensorcore {

class SGDOptimizer {
public:
    explicit SGDOptimizer(float learning_rate = 0.01f, float momentum = 0.9f)
        : lr_(learning_rate), momentum_(momentum) {}

    // Generalized parameter step
    void step(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) {
        if (params.size() != grads.size()) {
            throw std::invalid_argument("Parameters and gradients vector count mismatch.");
        }

        if (velocities_.size() != params.size()) {
            velocities_.clear();
            for (auto* p : params) {
                velocities_.push_back(Tensor::zeros(p->shape()));
            }
        }

        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* p = params[i];
            Tensor* g = grads[i];
            Tensor& v = velocities_[i];

            for (size_t j = 0; j < p->size(); ++j) {
                v[j] = momentum_ * v[j] + lr_ * (*g)[j];
                (*p)[j] -= v[j];
            }
        }
    }

    // Convenience overload for Layer / Sequential
    void step(Layer& model) {
        auto params = model.parameters();
        auto grads = model.gradients();
        step(params, grads);
    }

    // Backward-compatible overload for std::vector<Linear*>
    void step(std::vector<Linear*>& layers) {
        std::vector<Tensor*> params;
        std::vector<Tensor*> grads;
        for (auto* layer : layers) {
            auto p = layer->parameters();
            auto g = layer->gradients();
            params.insert(params.end(), p.begin(), p.end());
            grads.insert(grads.end(), g.begin(), g.end());
        }
        step(params, grads);
    }

    void zero_grad(Layer& model) {
        model.zero_grad();
    }

private:
    float lr_;
    float momentum_;
    std::vector<Tensor> velocities_;
};

} // namespace tensorcore
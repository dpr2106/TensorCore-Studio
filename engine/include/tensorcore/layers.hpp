#pragma once

#include "tensorcore/tensor.hpp"
#include <string>

namespace tensorcore {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::string name() const = 0;
};

class Linear : public Layer {
public:
    Linear(size_t in_features, size_t out_features, uint32_t seed = 42);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "Linear"; }

    Tensor& weights() { return weights_; }
    const Tensor& weights() const { return weights_; }
    Tensor& bias() { return bias_; }
    const Tensor& bias() const { return bias_; }

    Tensor& grad_weights() { return grad_weights_; }
    Tensor& grad_bias() { return grad_bias_; }

    void zero_grad();

private:
    size_t in_features_;
    size_t out_features_;
    Tensor weights_;
    Tensor bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;
    Tensor last_input_;
};

class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "ReLU"; }

private:
    Tensor last_input_;
};

} // namespace tensorcore

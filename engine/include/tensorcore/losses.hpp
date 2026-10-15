#pragma once

#include "tensorcore/tensor.hpp"
#include <string>

namespace tensorcore {

enum class Reduction {
    Mean,
    Sum,
    None
};

class Loss {
public:
    virtual ~Loss() = default;
    virtual std::string name() const = 0;
};

class MSELoss : public Loss {
public:
    explicit MSELoss(Reduction reduction = Reduction::Mean);

    std::string name() const override { return "MSELoss"; }
    Reduction reduction() const { return reduction_; }

    // Forward pass returning scalar loss (stored in 1x1 or 1-element Tensor) for Mean/Sum,
    // or element-wise squared error tensor for None.
    Tensor forward(const Tensor& predictions, const Tensor& targets);

    // Scalar loss helper for Mean/Sum reductions
    float forward_scalar(const Tensor& predictions, const Tensor& targets);

    // Backward pass returning gradient with respect to predictions (shape matches predictions.shape()).
    // For Mean and Sum reductions, grad_output is optional (defaults to 1.0f).
    // For None reduction, grad_output supplies incoming upstream gradients (shape matches broadcasted forward output).
    Tensor backward(const Tensor* grad_output = nullptr);

private:
    Reduction reduction_;
    Tensor last_predictions_;
    Tensor last_targets_;
    Tensor last_diff_; // predictions - targets (broadcasted)
    size_t last_numel_;
};

} // namespace tensorcore
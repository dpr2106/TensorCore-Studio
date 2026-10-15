#include "tensorcore/losses.hpp"
#include <cmath>

namespace tensorcore {

MSELoss::MSELoss(Reduction reduction)
    : reduction_(reduction), last_predictions_{}, last_targets_{}, last_diff_{}, last_numel_(0) {}

Tensor MSELoss::forward(const Tensor& predictions, const Tensor& targets) {
    last_predictions_ = predictions.clone();
    last_targets_ = targets.clone();

    // Verify broadcastability via broadcast_shapes.
    // If predictions and targets cannot broadcast, this will throw std::invalid_argument.
    Shape out_shape = broadcast_shapes(predictions.shape(), targets.shape());

    // In a supervised regression context, the output predictions tensor shape must not be
    // expanded by the target. If target has a larger rank or expands prediction dimensions,
    // predictions.shape() != out_shape. We require out_shape == predictions.shape().
    if (out_shape != predictions.shape()) {
        throw std::invalid_argument(
            "Target shape must be broadcast-compatible with and not expand predictions shape.");
    }

    // diff = predictions - targets
    last_diff_ = predictions.sub(targets);
    last_numel_ = last_diff_.size();

    if (reduction_ == Reduction::None) {
        Tensor result(last_diff_.shape());
        for (size_t i = 0; i < last_diff_.size(); ++i) {
            result[i] = last_diff_[i] * last_diff_[i];
        }
        return result;
    }

    if (last_numel_ == 0) {
        return Tensor({1}, 0.0f);
    }

    float sum_sq = 0.0f;
    for (size_t i = 0; i < last_diff_.size(); ++i) {
        sum_sq += last_diff_[i] * last_diff_[i];
    }

    float loss_val = (reduction_ == Reduction::Mean) ? (sum_sq / static_cast<float>(last_numel_)) : sum_sq;
    return Tensor({1}, loss_val);
}

float MSELoss::forward_scalar(const Tensor& predictions, const Tensor& targets) {
    if (reduction_ == Reduction::None) {
        throw std::logic_error("forward_scalar cannot be called on MSELoss with Reduction::None.");
    }
    Tensor loss_tensor = forward(predictions, targets);
    return loss_tensor[0];
}

Tensor MSELoss::backward(const Tensor* grad_output) {
    if (last_numel_ == 0) {
        return Tensor(last_predictions_.shape(), 0.0f);
    }

    Tensor grad_pred(last_predictions_.shape());

    if (reduction_ == Reduction::None) {
        if (!grad_output) {
            throw std::invalid_argument(
                "grad_output must be provided to backward() when Reduction::None is used.");
        }
        if (grad_output->shape() != last_diff_.shape()) {
            throw std::invalid_argument("grad_output shape must match forward output shape.");
        }
        // grad_pred = 2 * (predictions - targets) * grad_output
        for (size_t i = 0; i < last_diff_.size(); ++i) {
            grad_pred[i] = 2.0f * last_diff_[i] * (*grad_output)[i];
        }
        return grad_pred;
    }

    // Scalar loss backward (Mean or Sum)
    float upstream = 1.0f;
    if (grad_output) {
        if (grad_output->size() != 1) {
            throw std::invalid_argument("grad_output for scalar loss reduction must have size 1.");
        }
        upstream = (*grad_output)[0];
    }

    float scale_factor = (reduction_ == Reduction::Mean)
                             ? (2.0f * upstream / static_cast<float>(last_numel_))
                             : (2.0f * upstream);

    for (size_t i = 0; i < last_diff_.size(); ++i) {
        grad_pred[i] = last_diff_[i] * scale_factor;
    }

    return grad_pred;
}

} // namespace tensorcore
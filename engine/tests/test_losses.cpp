#include "tensorcore/losses.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tensorcore;

void test_mse_forward_equal_shapes() {
    Tensor pred({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor target({1, 3}, {1.0f, 5.0f, 3.0f});

    MSELoss loss_mean(Reduction::Mean);
    float l_mean = loss_mean.forward_scalar(pred, target);
    assert(std::fabs(l_mean - 3.0f) < 1e-5);

    MSELoss loss_sum(Reduction::Sum);
    float l_sum = loss_sum.forward_scalar(pred, target);
    assert(std::fabs(l_sum - 9.0f) < 1e-5);

    MSELoss loss_none(Reduction::None);
    Tensor l_none = loss_none.forward(pred, target);
    assert(l_none.shape() == Shape({1, 3}));
    assert(std::fabs(l_none[0] - 0.0f) < 1e-5);
    assert(std::fabs(l_none[1] - 9.0f) < 1e-5);
    assert(std::fabs(l_none[2] - 0.0f) < 1e-5);

    std::cout << "[PASS] test_mse_forward_equal_shapes\n";
}

void test_mse_backward_analytical() {
    Tensor pred({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor target({2, 2}, {0.0f, 2.0f, 1.0f, 6.0f});

    MSELoss loss_mean(Reduction::Mean);
    loss_mean.forward(pred, target);
    Tensor grad_mean = loss_mean.backward();

    assert(grad_mean.shape() == Shape({2, 2}));
    assert(std::fabs(grad_mean[0] - 0.5f) < 1e-5);
    assert(std::fabs(grad_mean[1] - 0.0f) < 1e-5);
    assert(std::fabs(grad_mean[2] - 1.0f) < 1e-5);
    assert(std::fabs(grad_mean[3] - (-1.0f)) < 1e-5);

    MSELoss loss_sum(Reduction::Sum);
    loss_sum.forward(pred, target);
    Tensor grad_sum = loss_sum.backward();

    assert(grad_sum.shape() == Shape({2, 2}));
    assert(std::fabs(grad_sum[0] - 2.0f) < 1e-5);
    assert(std::fabs(grad_sum[1] - 0.0f) < 1e-5);
    assert(std::fabs(grad_sum[2] - 4.0f) < 1e-5);
    assert(std::fabs(grad_sum[3] - (-4.0f)) < 1e-5);

    // Test None reduction backward with custom upstream gradient
    MSELoss loss_none(Reduction::None);
    loss_none.forward(pred, target);
    Tensor upstream({2, 2}, {1.0f, 0.5f, 2.0f, 1.0f});
    Tensor grad_none = loss_none.backward(&upstream);
    assert(std::fabs(grad_none[0] - 2.0f) < 1e-5);
    assert(std::fabs(grad_none[1] - 0.0f) < 1e-5);
    assert(std::fabs(grad_none[2] - 8.0f) < 1e-5);
    assert(std::fabs(grad_none[3] - (-4.0f)) < 1e-5);

    std::cout << "[PASS] test_mse_backward_analytical\n";
}

void test_mse_broadcast_target() {
    Tensor pred({3, 2}, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    });
    Tensor target({1, 2}, {1.0f, 2.0f});

    MSELoss loss_mean(Reduction::Mean);
    float l = loss_mean.forward_scalar(pred, target);
    assert(std::fabs(l - 40.0f / 6.0f) < 1e-5);

    Tensor grad = loss_mean.backward();
    assert(grad.shape() == Shape({3, 2}));
    assert(std::fabs(grad[0] - 0.0f) < 1e-5);
    assert(std::fabs(grad[1] - 0.0f) < 1e-5);
    assert(std::fabs(grad[2] - (2.0f * 2.0f / 6.0f)) < 1e-5);
    assert(std::fabs(grad[3] - (2.0f * 2.0f / 6.0f)) < 1e-5);
    assert(std::fabs(grad[4] - (2.0f * 4.0f / 6.0f)) < 1e-5);
    assert(std::fabs(grad[5] - (2.0f * 4.0f / 6.0f)) < 1e-5);

    std::cout << "[PASS] test_mse_broadcast_target\n";
}

void test_mse_exact_match_and_incompatible_shapes() {
    Tensor pred({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor exact({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    MSELoss loss(Reduction::Mean);
    float l = loss.forward_scalar(pred, exact);
    assert(std::fabs(l - 0.0f) < 1e-6);

    Tensor grad = loss.backward();
    for (size_t i = 0; i < grad.size(); ++i) {
        assert(std::fabs(grad[i] - 0.0f) < 1e-6);
    }

    // Incompatible shape: [2, 3] vs [2, 4]
    Tensor incompatible({2, 4}, 1.0f);
    bool caught = false;
    try {
        loss.forward(pred, incompatible);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);

    std::cout << "[PASS] test_mse_exact_match_and_incompatible_shapes\n";
}

void test_mse_finite_difference_gradient_check() {
    Tensor pred({2, 3}, {1.5f, -0.5f, 2.2f, 3.1f, -1.8f, 0.4f});
    Tensor target({2, 3}, {1.0f, 0.0f, 1.5f, 2.0f, -1.0f, 0.0f});

    MSELoss loss_fn(Reduction::Mean);
    loss_fn.forward(pred, target);
    Tensor analytical_grad = loss_fn.backward();

    const float eps = 1e-3f;
    for (size_t i = 0; i < pred.size(); ++i) {
        Tensor pred_plus = pred.clone();
        Tensor pred_minus = pred.clone();
        pred_plus[i] += eps;
        pred_minus[i] -= eps;

        float l_plus = loss_fn.forward_scalar(pred_plus, target);
        float l_minus = loss_fn.forward_scalar(pred_minus, target);
        float num_grad = (l_plus - l_minus) / (2.0f * eps);

        float diff = std::fabs(analytical_grad[i] - num_grad);
        assert(diff < 5e-4f);
    }

    std::cout << "[PASS] test_mse_finite_difference_gradient_check\n";
}

int main() {
    std::cout << "Running TensorCore MSE Loss Tests (TC-0207)...\n";
    test_mse_forward_equal_shapes();
    test_mse_backward_analytical();
    test_mse_broadcast_target();
    test_mse_exact_match_and_incompatible_shapes();
    test_mse_finite_difference_gradient_check();
    std::cout << "All MSE Loss unit & finite-difference gradient tests passed!\n";
    return 0;
}
#include "tensorcore/tensor.hpp"
#include "tensorcore/layers.hpp"
#include "tensorcore/losses.hpp"
#include "tensorcore/optimizers.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tensorcore;

void test_sequential_forward_backward() {
    Sequential model({
        std::make_shared<Linear>(2, 4, 100),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(4, 1, 101)
    });

    assert(model.size() == 3);
    assert(model.parameters().size() == 4); // 2 weights + 2 biases
    assert(model.gradients().size() == 4);

    Tensor X({3, 2}, {
        1.0f, 2.0f,
        -1.0f, 0.5f,
        0.0f, 1.0f
    });

    Tensor out = model.forward(X);
    assert(out.shape() == Shape({3, 1}));

    Tensor grad_out({3, 1}, 1.0f);
    Tensor grad_in = model.backward(grad_out);
    assert(grad_in.shape() == Shape({3, 2}));

    // Check parameter gradients populated
    auto grads = model.gradients();
    for (auto* g : grads) {
        float sum_abs = 0.0f;
        for (size_t i = 0; i < g->size(); ++i) {
            sum_abs += std::fabs((*g)[i]);
        }
        assert(sum_abs > 0.0f);
    }

    // Check zero_grad
    model.zero_grad();
    for (auto* g : model.gradients()) {
        for (size_t i = 0; i < g->size(); ++i) {
            assert((*g)[i] == 0.0f);
        }
    }

    std::cout << "[PASS] test_sequential_forward_backward\n";
}

void test_sequential_deterministic_training_convergence() {
    // 2-layer MLP learning non-linear continuous regression:
    // f(x1, x2) = x1^2 + x2
    // Training set with fixed deterministic seed and fixed data:
    Tensor X({4, 2}, {
        0.0f, 0.5f,  // target = 0.0 + 0.5 = 0.5
        1.0f, 0.0f,  // target = 1.0 + 0.0 = 1.0
        1.0f, 1.0f,  // target = 1.0 + 1.0 = 2.0
        2.0f, -0.5f  // target = 4.0 - 0.5 = 3.5
    });

    Tensor Y({4, 1}, {
        0.5f,
        1.0f,
        2.0f,
        3.5f
    });

    auto model = std::make_shared<Sequential>(std::vector<std::shared_ptr<Layer>>{
        std::make_shared<Linear>(2, 8, 42),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(8, 1, 43)
    });

    MSELoss loss_fn(Reduction::Mean);
    SGDOptimizer optimizer(0.02f, 0.9f);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < 150; ++epoch) {
        // 1. Reset gradients
        model->zero_grad();

        // 2. Forward pass
        Tensor pred = model->forward(X);

        // 3. Loss computation
        float loss = loss_fn.forward_scalar(pred, Y);
        if (epoch == 0) {
            initial_loss = loss;
        }
        final_loss = loss;

        // 4. Backward loss gradient
        Tensor grad_loss = loss_fn.backward();

        // 5. Model backpropagation
        model->backward(grad_loss);

        // 6. Optimizer update step
        optimizer.step(*model);
    }

    std::cout << "Initial Loss: " << initial_loss << ", Final Loss: " << final_loss << "\n";
    // Assert monotonic substantial decrease
    assert(final_loss < initial_loss * 0.15f); // At least 85% loss reduction
    assert(final_loss < 0.25f);

    std::cout << "[PASS] test_sequential_deterministic_training_convergence\n";
}

int main() {
    std::cout << "Running TensorCore Sequential & Multi-Layer Training Tests (TC-0205)...\n";
    test_sequential_forward_backward();
    test_sequential_deterministic_training_convergence();
    std::cout << "All Sequential & Multi-Layer training tests passed successfully!\n";
    return 0;
}
#include "tensorcore/tensor.hpp"
#include "tensorcore/layers.hpp"
#include "tensorcore/optimizers.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tensorcore;

void test_linear_layer() {
    Linear dense(3, 2, 123);
    Tensor X({4, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        1.0f, 0.0f, -1.0f
    });

    Tensor out = dense.forward(X);
    assert(out.shape()[0] == 4);
    assert(out.shape()[1] == 2);

    Tensor grad_out({4, 2}, 1.0f);
    Tensor grad_in = dense.backward(grad_out);
    assert(grad_in.shape()[0] == 4);
    assert(grad_in.shape()[1] == 3);
    assert(dense.grad_weights().shape()[0] == 3);
    assert(dense.grad_weights().shape()[1] == 2);
    assert(dense.grad_bias().shape()[1] == 2);
    std::cout << "[PASS] test_linear_layer\n";
}

void test_training_step() {
    // Single layer learning simple mapping: Y = X * [2, -1]
    Linear dense(2, 1, 42);
    SGDOptimizer opt(0.01f, 0.9f);
    std::vector<Linear*> layers = {&dense};

    Tensor X({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor Y_target({2, 1}, {0.0f, 2.0f}); // 1*2 - 2*1 = 0; 3*2 - 4*1 = 2

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < 50; ++epoch) {
        dense.zero_grad();
        Tensor pred = dense.forward(X);

        // MSE Loss: (pred - Y)^2 / N
        Tensor diff = pred.sub(Y_target);
        float loss = 0.0f;
        for (size_t i = 0; i < diff.size(); ++i) {
            loss += diff[i] * diff[i];
        }
        loss /= static_cast<float>(diff.size());
        if (epoch == 0) initial_loss = loss;
        final_loss = loss;

        // grad_loss = 2 * (pred - Y) / N
        Tensor grad_loss = diff.scale(2.0f / static_cast<float>(diff.size()));
        dense.backward(grad_loss);
        opt.step(layers);
    }

    assert(final_loss < initial_loss);
    std::cout << "[PASS] test_training_step (initial_loss=" << initial_loss << ", final_loss=" << final_loss << ")\n";
}

int main() {
    std::cout << "Running Autograd & Layers Unit Tests...\n";
    test_linear_layer();
    test_training_step();
    std::cout << "All Autograd and layer tests passed!\n";
    return 0;
}

#include "tensorcore/tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tensorcore;

void test_tensor_creation() {
    Tensor t({2, 3}, 1.5f);
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);
    assert(t.size() == 6);
    assert(t.at({0, 0}) == 1.5f);
    assert(t.at({1, 2}) == 1.5f);
    std::cout << "[PASS] test_tensor_creation\n";
}

void test_matmul_and_transpose() {
    // A: 2x3, B: 3x2
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B({3, 2}, {7, 8, 9, 1, 2, 3});
    // Expected C = A x B:
    // [1*7+2*9+3*2, 1*8+2*1+3*3] = [7+18+6, 8+2+9]  = [31, 19]
    // [4*7+5*9+6*2, 4*8+5*1+6*3] = [28+45+12, 32+5+18] = [85, 55]
    Tensor C = A.matmul(B);
    assert(C.shape()[0] == 2);
    assert(C.shape()[1] == 2);
    assert(std::fabs(C.at({0, 0}) - 31.0f) < 1e-5);
    assert(std::fabs(C.at({0, 1}) - 19.0f) < 1e-5);
    assert(std::fabs(C.at({1, 0}) - 85.0f) < 1e-5);
    assert(std::fabs(C.at({1, 1}) - 55.0f) < 1e-5);

    Tensor BT = B.transpose();
    assert(BT.shape()[0] == 2);
    assert(BT.shape()[1] == 3);
    assert(BT.at({0, 0}) == 7.0f);
    assert(BT.at({1, 0}) == 8.0f);
    assert(BT.at({0, 1}) == 9.0f);
    std::cout << "[PASS] test_matmul_and_transpose\n";
}

void test_relu_backward() {
    Tensor X({1, 4}, {-2.0f, 0.5f, -0.1f, 3.0f});
    Tensor grad_out({1, 4}, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor grad_in = X.relu_backward(grad_out);
    assert(grad_in[0] == 0.0f);
    assert(grad_in[1] == 1.0f);
    assert(grad_in[2] == 0.0f);
    assert(grad_in[3] == 1.0f);
    std::cout << "[PASS] test_relu_backward\n";
}

void test_softmax() {
    Tensor logits({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor probs = logits.softmax();
    float sum_p = probs.sum();
    assert(std::fabs(sum_p - 1.0f) < 1e-5);
    assert(probs[2] > probs[1] && probs[1] > probs[0]);
    std::cout << "[PASS] test_softmax\n";
}

int main() {
    std::cout << "Running TensorCore C++ Engine Tests...\n";
    test_tensor_creation();
    test_matmul_and_transpose();
    test_relu_backward();
    test_softmax();
    std::cout << "All TensorCore engine tests passed successfully!\n";
    return 0;
}

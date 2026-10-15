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
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B({3, 2}, {7, 8, 9, 1, 2, 3});
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

// ---------------- TC-0105 BROADCASTING TESTS ---------------- //

void test_broadcasting_equal_shapes() {
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B({2, 3}, {10, 20, 30, 40, 50, 60});
    Tensor C = A.add(B);
    assert(C.shape() == Shape({2, 3}));
    assert(C.at({0, 0}) == 11.0f);
    assert(C.at({1, 2}) == 66.0f);

    Tensor D = B.sub(A);
    assert(D.at({0, 0}) == 9.0f);
    assert(D.at({1, 2}) == 54.0f);

    Tensor E = A.mul(B);
    assert(E.at({0, 0}) == 10.0f);
    assert(E.at({1, 2}) == 360.0f);
    std::cout << "[PASS] test_broadcasting_equal_shapes\n";
}

void test_broadcasting_1d_to_2d() {
    // A: [2, 3], B: [3] -> result: [2, 3]
    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B({3}, {10, 20, 30});

    Tensor C = A.add(B);
    assert(C.shape() == Shape({2, 3}));
    assert(C.at({0, 0}) == 11.0f);
    assert(C.at({0, 1}) == 22.0f);
    assert(C.at({0, 2}) == 33.0f);
    assert(C.at({1, 0}) == 14.0f);
    assert(C.at({1, 1}) == 25.0f);
    assert(C.at({1, 2}) == 36.0f);

    // Reverse order: B + A -> [2, 3]
    Tensor C2 = B.add(A);
    assert(C2.shape() == Shape({2, 3}));
    assert(C2.at({1, 1}) == 25.0f);

    // Multiplication: [2, 3] * [3]
    Tensor M = A.mul(B);
    assert(M.at({0, 1}) == 40.0f);
    assert(M.at({1, 2}) == 180.0f);
    std::cout << "[PASS] test_broadcasting_1d_to_2d\n";
}

void test_broadcasting_2d_to_3d() {
    // A: [2, 2, 3], B: [3] -> result: [2, 2, 3]
    Tensor A({2, 2, 3}, 1.0f);
    Tensor B({3}, {10, 20, 30});

    Tensor C = A.add(B);
    assert(C.shape() == Shape({2, 2, 3}));
    assert(C.size() == 12);
    assert(C.at({0, 0, 0}) == 11.0f);
    assert(C.at({1, 1, 1}) == 21.0f);
    assert(C.at({1, 1, 2}) == 31.0f);

    // A: [2, 2, 3], B: [2, 3] -> result: [2, 2, 3]
    Tensor B2({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor C2 = A.add(B2);
    assert(C2.shape() == Shape({2, 2, 3}));
    assert(C2.at({0, 0, 0}) == 2.0f);
    assert(C2.at({0, 1, 0}) == 5.0f);
    assert(C2.at({1, 0, 0}) == 2.0f);
    assert(C2.at({1, 1, 0}) == 5.0f);
    std::cout << "[PASS] test_broadcasting_2d_to_3d\n";
}

void test_broadcasting_singleton_dimensions() {
    // A: [2, 1], B: [1, 3] -> result: [2, 3]
    Tensor A({2, 1}, {10, 20});
    Tensor B({1, 3}, {1, 2, 3});

    Tensor C = A.add(B);
    assert(C.shape() == Shape({2, 3}));
    // Row 0: 10 + [1, 2, 3] = [11, 12, 13]
    // Row 1: 20 + [1, 2, 3] = [21, 22, 23]
    assert(C.at({0, 0}) == 11.0f);
    assert(C.at({0, 1}) == 12.0f);
    assert(C.at({0, 2}) == 13.0f);
    assert(C.at({1, 0}) == 21.0f);
    assert(C.at({1, 1}) == 22.0f);
    assert(C.at({1, 2}) == 23.0f);

    // Subtraction: [2, 1] - [1, 3]
    Tensor S = A.sub(B);
    assert(S.at({0, 0}) == 9.0f);
    assert(S.at({1, 2}) == 17.0f);
    std::cout << "[PASS] test_broadcasting_singleton_dimensions\n";
}

void test_broadcasting_multiple_singletons() {
    // A: [2, 1, 4], B: [1, 3, 1] -> result: [2, 3, 4]
    Tensor A({2, 1, 4}, 2.0f);
    Tensor B({1, 3, 1}, 5.0f);

    Tensor C = A.mul(B);
    assert(C.shape() == Shape({2, 3, 4}));
    assert(C.size() == 24);
    for (size_t i = 0; i < C.size(); ++i) {
        assert(C[i] == 10.0f);
    }
    std::cout << "[PASS] test_broadcasting_multiple_singletons\n";
}

void test_broadcasting_incompatible_shapes() {
    Tensor A({2, 3}, 1.0f);
    Tensor B({2, 4}, 2.0f);
    bool caught = false;
    try {
        A.add(B);
    } catch (const std::invalid_argument& e) {
        caught = true;
    }
    assert(caught);

    Tensor C({3, 2}, 1.0f);
    Tensor D({3}, 1.0f);
    caught = false;
    try {
        C.add(D);
    } catch (const std::invalid_argument& e) {
        caught = true;
    }
    assert(caught);
    std::cout << "[PASS] test_broadcasting_incompatible_shapes\n";
}

int main() {
    std::cout << "Running TensorCore C++ Engine Tests (including TC-0105 Broadcasting)...\n";
    test_tensor_creation();
    test_matmul_and_transpose();
    test_relu_backward();
    test_softmax();

    // Broadcasting suite
    test_broadcasting_equal_shapes();
    test_broadcasting_1d_to_2d();
    test_broadcasting_2d_to_3d();
    test_broadcasting_singleton_dimensions();
    test_broadcasting_multiple_singletons();
    test_broadcasting_incompatible_shapes();

    std::cout << "All TensorCore engine & broadcasting tests passed successfully!\n";
    return 0;
}
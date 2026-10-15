#pragma once

#include "tensorcore/tensor.hpp"
#include "tensorcore/layers.hpp"
#include "tensorcore/losses.hpp"
#include "tensorcore/optimizers.hpp"
#include "tensorcore/dataset.hpp"
#include <vector>

namespace tensorcore {

struct TrainingConfig {
    size_t epochs = 100;
};

struct TrainingHistory {
    float initial_loss = 0.0f;
    std::vector<float> epoch_losses;
    float final_loss = 0.0f;
};

class Trainer {
public:
    static TrainingHistory fit(
        Layer& model,
        const Dataset& dataset,
        MSELoss& loss_fn,
        SGDOptimizer& optimizer,
        const TrainingConfig& config
    );
};

} // namespace tensorcore
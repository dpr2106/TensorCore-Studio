#include "tensorcore/trainer.hpp"
#include <stdexcept>

namespace tensorcore {

TrainingHistory Trainer::fit(
    Layer& model,
    const Dataset& dataset,
    MSELoss& loss_fn,
    SGDOptimizer& optimizer,
    const TrainingConfig& config
) {
    if (dataset.num_samples == 0) {
        throw std::invalid_argument("Cannot train on an empty dataset.");
    }
    if (config.epochs == 0) {
        throw std::invalid_argument("Epochs count must be greater than zero.");
    }

    TrainingHistory history;
    history.epoch_losses.reserve(config.epochs);

    // 1. Initial loss evaluation before any parameter updates
    {
        Tensor initial_pred = model.forward(dataset.features);
        history.initial_loss = loss_fn.forward_scalar(initial_pred, dataset.targets);
    }

    // 2. Training loop over epochs
    for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
        // Reset gradients before backward accumulation
        model.zero_grad();

        // Forward propagation
        Tensor pred = model.forward(dataset.features);

        // Compute loss for current epoch
        float epoch_loss = loss_fn.forward_scalar(pred, dataset.targets);
        history.epoch_losses.push_back(epoch_loss);

        // Compute loss gradient with respect to predictions
        Tensor grad_loss = loss_fn.backward();

        // Model backpropagation
        model.backward(grad_loss);

        // Parameter update step
        optimizer.step(model);
    }

    // 3. Fresh evaluation of final loss after last optimizer update
    {
        Tensor final_pred = model.forward(dataset.features);
        history.final_loss = loss_fn.forward_scalar(final_pred, dataset.targets);
    }

    return history;
}

} // namespace tensorcore
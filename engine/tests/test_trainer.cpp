#include "tensorcore/dataset.hpp"
#include "tensorcore/trainer.hpp"
#include "tensorcore/layers.hpp"
#include "tensorcore/losses.hpp"
#include "tensorcore/optimizers.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tensorcore;

void test_csv_loader_basic() {
    std::string csv_data =
        "id,feature1,feature2,target\n"
        "1,1.5,2.5,10.0\n"
        "2,3.0,4.0,20.0\n"
        "3,5.0,6.0,30.0\n";

    CSVLoader::Options opts;
    opts.has_header = true;
    opts.delimiter = ',';
    opts.feature_columns = {1, 2}; // feature1, feature2
    opts.target_columns = {3};     // target

    Dataset ds = CSVLoader::load_from_string(csv_data, opts);

    assert(ds.num_samples == 3);
    assert(ds.features.shape() == Shape({3, 2}));
    assert(ds.targets.shape() == Shape({3, 1}));

    assert(std::fabs(ds.features.at({0, 0}) - 1.5f) < 1e-5);
    assert(std::fabs(ds.features.at({0, 1}) - 2.5f) < 1e-5);
    assert(std::fabs(ds.targets.at({0, 0}) - 10.0f) < 1e-5);

    assert(std::fabs(ds.features.at({2, 0}) - 5.0f) < 1e-5);
    assert(std::fabs(ds.targets.at({2, 0}) - 30.0f) < 1e-5);

    std::cout << "[PASS] test_csv_loader_basic\n";
}

void test_csv_loader_default_feature_selection_and_quotes() {
    std::string csv_data =
        "\"feat_a\",target,\"feat_b\"\n"
        "1.0,10.0,2.0\n"
        "3.0,20.0,4.0\n";

    CSVLoader::Options opts;
    opts.has_header = true;
    opts.target_columns = {1}; // target at col 1; features auto-selected as {0, 2}

    Dataset ds = CSVLoader::load_from_string(csv_data, opts);
    assert(ds.num_samples == 2);
    assert(ds.features.shape() == Shape({2, 2}));
    assert(ds.targets.shape() == Shape({2, 1}));

    assert(std::fabs(ds.features.at({0, 0}) - 1.0f) < 1e-5);
    assert(std::fabs(ds.features.at({0, 1}) - 2.0f) < 1e-5);
    assert(std::fabs(ds.targets.at({0, 0}) - 10.0f) < 1e-5);

    std::cout << "[PASS] test_csv_loader_default_feature_selection_and_quotes\n";
}

void test_csv_loader_validation_errors() {
    // 1. Incompatible / empty target columns
    {
        bool caught = false;
        try {
            CSVLoader::Options opts; // target_columns empty
            CSVLoader::load_from_string("1,2\n", opts);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
    }

    // 2. Overlap between feature and target
    {
        bool caught = false;
        try {
            CSVLoader::Options opts;
            opts.feature_columns = {0, 1};
            opts.target_columns = {1};
            CSVLoader::load_from_string("1,2\n", opts);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
    }

    // 3. Duplicate target columns
    {
        bool caught = false;
        try {
            CSVLoader::Options opts;
            opts.target_columns = {0, 0};
            CSVLoader::load_from_string("1,2\n", opts);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
    }

    // 4. Out of range column index
    {
        bool caught = false;
        try {
            CSVLoader::Options opts;
            opts.target_columns = {5};
            CSVLoader::load_from_string("1,2\n3,4\n", opts);
        } catch (const std::out_of_range&) {
            caught = true;
        }
        assert(caught);
    }

    // 5. Ragged row
    {
        bool caught = false;
        try {
            CSVLoader::Options opts;
            opts.target_columns = {1};
            CSVLoader::load_from_string("1,2\n3,4,5\n", opts);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
    }

    // 6. Non-numeric value
    {
        bool caught = false;
        try {
            CSVLoader::Options opts;
            opts.target_columns = {1};
            CSVLoader::load_from_string("1.0,hello\n", opts);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
    }

    // 7. Empty CSV
    {
        bool caught = false;
        try {
            CSVLoader::Options opts;
            opts.target_columns = {0};
            CSVLoader::load_from_string("header1,header2\n", opts);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
    }

    std::cout << "[PASS] test_csv_loader_validation_errors\n";
}

void test_end_to_end_training_pipeline() {
    // Synthetic dataset representing: y = 2.0 * x1 - 1.0 * x2 + 0.5
    std::string training_csv =
        "x1,x2,y\n"
        "0.0,0.0,0.5\n"
        "1.0,0.0,2.5\n"
        "0.0,1.0,-0.5\n"
        "1.0,1.0,1.5\n"
        "2.0,0.5,4.0\n"
        "0.5,2.0,0.0\n";

    CSVLoader::Options opts;
    opts.has_header = true;
    opts.feature_columns = {0, 1};
    opts.target_columns = {2};

    Dataset dataset = CSVLoader::load_from_string(training_csv, opts);
    assert(dataset.num_samples == 6);

    // Create a 2-layer regression network: Linear(2, 4) -> ReLU -> Linear(4, 1)
    Sequential model({
        std::make_shared<Linear>(2, 4, 123),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(4, 1, 124)
    });

    MSELoss loss_fn(Reduction::Mean);
    SGDOptimizer optimizer(0.015f, 0.9f);

    TrainingConfig config;
    config.epochs = 120;

    TrainingHistory history = Trainer::fit(model, dataset, loss_fn, optimizer, config);

    assert(history.epoch_losses.size() == 120);
    assert(std::isfinite(history.initial_loss));
    assert(std::isfinite(history.final_loss));

    std::cout << "Trainer Fit Results: Initial Loss = " << history.initial_loss
              << ", Final Loss = " << history.final_loss << "\n";

    // Verify substantial convergence (>80% overall reduction)
    assert(history.final_loss < history.initial_loss * 0.20f);
    assert(history.final_loss < 0.25f);

    std::cout << "[PASS] test_end_to_end_training_pipeline\n";
}

int main() {
    std::cout << "Running TensorCore Dataset & Trainer Tests (TC-0131)...\n";
    test_csv_loader_basic();
    test_csv_loader_default_feature_selection_and_quotes();
    test_csv_loader_validation_errors();
    test_end_to_end_training_pipeline();
    std::cout << "All Dataset & Trainer tests passed successfully!\n";
    return 0;
}
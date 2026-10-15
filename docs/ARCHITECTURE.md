# TensorCore Studio — Master Architecture & Roadmap

## 1. System Components
- **`engine/`**: Custom C++ Tensor library (`Tensor<T>`, `Shape`, `Strides`), automatic differentiation / backprop DAG, activation functions (ReLU, Sigmoid, GeLU, Softmax), losses (CrossEntropy, MSE), optimizers (SGD with momentum, AdamW), execution backends (CPU Eigen/Vectorized, and native GPU/Vulkan or WebGPU compute).
- **`backend/`**: Java 17 + Spring Boot 3 REST and WebSocket server for dataset ingestion, preprocessing, project workspaces, job orchestration, and execution dispatching.
- **`frontend/`**: Vite + React + TypeScript web application with rich dark-mode aesthetics, dynamic canvas/WebGL training loss plots, layer visualization, confusion matrices, and dataset inspection.

## 2. Verified Toolchain Matrix (Audit on 2026-10-16)
- **Host OS**: Windows 11
- **C++**: GCC 15.1.0 (`g++`) with C++17 / C++20 support
- **Build System**: CMake 4.3.1
- **Java**: OpenJDK 17.0.20.1 LTS
- **Node**: v24.13.0
- **Containerization**: Docker 29.8.1

## 3. Incremental Milestones
- **TC-0001**: Repository initialisation, branch governance, architecture blueprint.
- **TC-0100**: C++ ML Engine Foundation — Tensor ND-array, memory management, strides, broadcasting, and unit tests.
- **TC-0200**: Computational Graph & Autograd — Computational nodes, backward tape, gradient verification via finite differences.
- **TC-0300**: Neural Network Layers & Optimizers — Dense layer, activations, SGD, Adam, and benchmark tests.
- **TC-0400**: Spring Boot Backend — Workspaces, dataset upload & CSV parsing, train/val/test splitter, training orchestration.
- **TC-0500**: Web Application Frontend — Dataset visualizer, model designer, real-time metrics dashboard.
- **TC-0600**: End-to-End Integration & GPU Acceleration Testing.

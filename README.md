# TensorCore Studio

TensorCore Studio is a modern, high-performance machine learning development and training platform featuring a custom C++ ML engine, GPU/CPU acceleration pathways, dataset processing, interactive training metrics, and full experiment lifecycle management.

## System Architecture

- **Core ML Engine**: C++17 / C++20 custom tensor computation and neural network engine with CMake.
- **Backend Service**: Java 17 / Spring Boot with Maven, providing user workspaces, dataset pipelines, experiment management, and WebSocket metric streaming.
- **Frontend Application**: Modern high-fidelity UI/UX with interactive training visualizers, loss curves, and dataset explorers.
- **Persistence**: PostgreSQL with Flyway migrations.

## Repository Structure

```
├── engine/              # C++ ML Engine & CMake build
│   ├── include/         # Header declarations
│   ├── src/             # Implementation (Tensors, Layers, Optimizers)
│   └── tests/           # Unit and numerical gradient tests
├── backend/             # Spring Boot REST & WebSocket server
│   ├── src/             # API, service layer, worker integration
│   └── pom.xml
├── frontend/            # Web interface & real-time training dashboard
├── docs/                # Architecture specifications & task tracking
└── docker-compose.yml   # Local environment configuration
```

# TensorCore Studio — System Architecture & Component Specification

## 1. High-Level Subsystem Topology

```
+-----------------------------------------------------------------------------------+
|                           TENSORCORE STUDIO WEB FRONTEND                          |
|                     (Vite + React 18 + TypeScript + Canvas 2D)                    |
|  - Tabular Dataset Explorer     - Real-Time Loss Curves    - Interactive Predict  |
|  - Graph Architecture Builder   - Hardware Telemetry Bar   - Workspace Switcher   |
+------------------------------------------+----------------------------------------+
                                           |
                    REST APIs (HTTP/JSON)  |  WebSocket (STOMP / SockJS)
                                           v
+-----------------------------------------------------------------------------------+
|                        JAVA SPRING BOOT 3 BACKEND SERVICE                         |
|  - Security & JWT Auth Filter   - Project & Dataset Svc    - Async Job Dispatcher |
|  - Train/Val/Test Splitter      - Metrics Aggregator       - Process Worker Pool  |
+--------------------+-------------------------------------+------------------------+
                     |                                     |
    JDBC Connection  |                                     |  Bidirectional IPC
    (Flyway SQL)     v                                     v  (Named Pipes / Sockets)
+--------------------+-------------+     +-----------------+-----------------------+
|        POSTGRESQL 16+            |     |        NATIVE C++ ML ENGINE WORKER    |
|  - Users & Workspaces            |     |  - N-Dim Strided Tensor (`Tensor<T>`) |
|  - Datasets & Schemas            |     |  - Dynamic Autograd Tape              |
|  - Experiments & Runs            |     |  - Dense, Activations & Optimizers    |
|  - Checkpoints & Metrics         |     |  - AVX2 CPU SIMD / Vulkan Compute     |
+----------------------------------+     +---------------------------------------+
```

## 2. Component Boundaries & Interfaces

### 2.1 Native C++ Machine Learning Engine
- **Memory Model**: Row-major contiguous heap allocation with dynamic dimension shapes (`std::vector<size_t>`) and stride arithmetic. Slicing and views share underlying reference-counted buffer storage to minimize reallocations.
- **Computation Execution**:
  - CPU Path: Multi-threaded tile-based matrix multiplication (GEMM) using AVX2 FMA vector intrinsics.
  - GPU Path: Native Vulkan Compute pipeline or client WebGPU compute shaders (`wgsl`).
- **Autograd Engine**: Dynamic tape recording node operations, input references, and backward closures for reverse-mode automatic differentiation.
- **Worker Interface**: Standalone native binary communicating with Java via binary framed streams over standard input/output or domain sockets, receiving batch payloads and emitting metric JSON frames.

### 2.2 Java Spring Boot 3 Backend
- **Data Access Layer**: Spring Data JPA with PostgreSQL 16. Database schema versioning strictly controlled via Flyway migrations (`db/migration/V*`).
- **Worker Orchestration**: `ProcessBuilder` handles C++ worker subprocess lifecycle management, health monitoring, timeout management, and graceful signal termination (`SIGTERM`/`SIGKILL`).
- **Streaming Pipeline**: WebSocket STOMP broker broadcasts per-epoch and per-step telemetry payloads directly to subscribers.

### 2.3 Web Workstation Frontend
- **State Management**: Zustand lightweight store for project workspace context, active dataset preview, and training configuration.
- **Canvas Rendering Engine**: HTML5 Canvas rendering engine delivering 60 FPS convergence plots without DOM garbage collection bottlenecks.
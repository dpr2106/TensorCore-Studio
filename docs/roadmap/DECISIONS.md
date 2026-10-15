# TensorCore Studio — Architectural Decisions & Technical Records

## ADR-001: Frontend Framework Selection
- **Status**: Accepted
- **Context**: The platform requires a modern, responsive, high-density workstation interface capable of handling dynamic tabular dataset views, interactive layer graphs, and 60 FPS real-time metric animations.
- **Options Evaluated**:
  1. *Next.js (React)*: Unnecessary server-side rendering (SSR) complexity for an interactive desktop SPA.
  2. *Vite + React 18 + TypeScript*: Extremely fast compilation, zero SSR overhead, excellent ecosystem for canvas visualizers and Web Workers.
  3. *Vanilla JS / HTML5*: Difficult to maintain complex state for neural network graphs and dataset transformations.
- **Decision**: Adopt **Vite + React 18 with TypeScript**.
- **Consequences**: Fast HMR, clean component modularity, effortless Web Worker bundling.

## ADR-002: Java Backend to C++ Engine Interoperability
- **Status**: Accepted
- **Context**: The Spring Boot backend needs to orchestrate model training runs executed by the native C++ machine learning engine.
- **Options Evaluated**:
  1. *Direct JNI (Java Native Interface)*: Fastest possible latency; however, any fatal C++ signal (e.g. segmentation fault from invalid memory access or out-of-memory) will crash the entire JVM host process.
  2. *Out-of-Process Native Subprocess Worker via IPC (Pipes / Local Sockets)*: Slight serialization overhead, but provides complete crash isolation. If a training job crashes, the Spring Boot API, web session, and database remain unaffected.
  3. *Microservice over HTTP*: High HTTP/REST serialization latency for large weight tensors.
- **Decision**: Adopt **Out-of-Process Native Subprocess Worker via IPC**.
- **Consequences**: Superior stability and resiliency; requires defining a structured binary/JSON framing protocol over standard streams or sockets.

## ADR-003: GPU Acceleration Reality & Tiered Execution Model
- **Status**: Accepted
- **Context**: Hardware acceleration must be honest and verified. WebAssembly does not provide GPU compute by default, and a Java backend cannot access a remote user's client GPU.
- **Options Evaluated**:
  1. *CUDA Exclusive*: Limits acceleration to NVIDIA desktop hardware; fails on AMD, Intel, and macOS.
  2. *Vulkan Compute (Native) & WebGPU (Browser)*: Portable, cross-vendor compute shader execution.
  3. *CPU-Only*: Portable but limited throughput for very deep models.
- **Decision**: Implement a **Tiered Runtime Architecture**:
  - Tier 1 (Mandatory MVP): Multi-threaded AVX2 CPU SIMD engine. 100% reliable across all x86_64 machines.
  - Tier 2 (Native GPU): Vulkan Compute worker with explicit queue capability probing.
  - Tier 3 (Client GPU): Browser-local WebGPU compute shaders (`wgsl`) for client-side edge training.
  - If a requested GPU path fails verification, the engine automatically falls back to AVX2 CPU with an explicit diagnostic badge.
- **Consequences**: Absolute technical honesty; prevents fraudulent GPU performance claims.

## ADR-004: Database Schema Migration Strategy
- **Status**: Accepted
- **Context**: Relational schemas for users, projects, datasets, experiments, and checkpoints must remain reproducible across development and production environments.
- **Options Evaluated**:
  1. *Hibernate Auto-DDL (`ddl-auto=update`)*: Unsafe for production; prone to data loss and unversioned schema drift.
  2. *Flyway SQL Migrations*: Version-controlled, immutable plain-SQL scripts executed upon application startup.
  3. *Liquibase*: XML/YAML abstractions that add verbosity.
- **Decision**: Adopt **Flyway SQL Migrations**.
- **Consequences**: Clear, traceable, and repeatable database schema evolution.

## ADR-005: Real-Time Telemetry Streaming Protocol
- **Status**: Accepted
- **Context**: The training engine generates per-epoch and per-step metrics (loss, accuracy, gradients, learning rate) that must be streamed to the web client with minimal latency.
- **Options Evaluated**:
  1. *Polling REST API (`GET /metrics`)*: High HTTP header overhead, sluggish update rates, poor UX.
  2. *Server-Sent Events (SSE)*: Unidirectional only; requires separate HTTP requests to send pause/resume/stop commands.
  3. *WebSocket (STOMP / SockJS)*: Full-duplex communication enabling both bidirectional training controls and high-frequency metric broadcasts.
- **Decision**: Adopt **WebSocket with STOMP over SockJS**.
- **Consequences**: Instantaneous 60Hz metric delivery and responsive interactive training control.
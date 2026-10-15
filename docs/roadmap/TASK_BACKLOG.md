# TensorCore Studio - Master Engineering Task Backlog


## Task Classification Taxonomy
- **DONE - VERIFIED**: Implementation exists and automated verification passes with recorded evidence.
- **IMPLEMENTED - UNVERIFIED**: Implementation exists but requires formal integration verification.
- **PARTIAL**: Substantial implementation exists but acceptance criteria remain.
- **IN PROGRESS**: Active task currently underway.
- **NOT STARTED**: Implementation not yet begun.
- **BLOCKED**: Cannot proceed until named dependency task is complete.

---

## Epic 1: Project Governance, Build Systems & Quality Gates (TC-0001 - TC-0015)

### TC-0001: Project Setup, Architecture & Repository Governance
- **Status**: DONE - VERIFIED (Commit `43532c3`, verified on `main`)
- **Priority**: P0 | **Area**: Governance
- **Scope**: Establish `.gitignore`, initial `README.md`, baseline toolchain definitions, and core project directories.
- **Exclusions**: Application source code.

### TC-0002: Docker Compose Development Stack
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: DevOps
- **Scope**: Create `docker-compose.yml` defining PostgreSQL 16 service with volume persistence, credentials configuration, and healthcheck.
- **Exclusions**: Production Kubernetes configs.

### TC-0003: Multi-Language GitHub Actions CI Matrix
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: CI/CD
- **Scope**: Configure GitHub Actions workflow running CMake C++ tests on GCC/Clang, Maven verify on Java 17, and Vite TypeScript lint/build.
- **Exclusions**: Self-hosted GPU runner deployment.

### TC-0004: EditorConfig & Code Formatting Rules
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Governance
- **Scope**: Add `.editorconfig`, `.clang-format` (LLVM style), and prettier/eslint configs for frontend.
- **Exclusions**: Automated autofix git hooks.

### TC-0005: Git Commit & Branch Policy Documentation
- **Status**: DONE - VERIFIED (Documented in `docs/roadmap/PR_STRATEGY.md`)
- **Priority**: P1 | **Area**: Governance
- **Scope**: Document commit conventions (`type(scope): message`), branch naming rules (`feat/TC-XXXX`), and protected branch policies.
- **Exclusions**: Automated branch protection rule configuration via GitHub API.

### TC-0006: Pre-commit Hook Template
- **Status**: NOT STARTED
- **Priority**: P3 | **Area**: Tooling
- **Scope**: Create optional developer pre-commit hook script checking for secrets, large files, and trailing whitespace.
- **Exclusions**: Enforced mandatory local hooks.

### TC-0007: Architecture Decision Records (ADR) Documentation
- **Status**: DONE - VERIFIED (Documented in `docs/roadmap/DECISIONS.md`)
- **Priority**: P1 | **Area**: Architecture
- **Scope**: Document ADR-001 through ADR-005 covering frontend, backend, engine interop, GPU reality, and migrations.
- **Exclusions**: Future unfinalized subsystem decisions.

### TC-0008: Engineering Task Backlog Publication
- **Status**: IN PROGRESS (This document `docs/roadmap/TASK_BACKLOG.md`)
- **Priority**: P0 | **Area**: Planning
- **Scope**: Comprehensive 120+ task backlog mapping all deliverables to unique reviewable units.
- **Exclusions**: Implementation code.

### TC-0009: Existing Work Audit & Evidence Matrix
- **Status**: DONE - VERIFIED (Documented in `docs/roadmap/EXISTING_WORK_AUDIT.md`)
- **Priority**: P0 | **Area**: Audit
- **Scope**: Physical file-by-file audit of branches `feat/TC-0101`, `feat/TC-0201`, and `feat/TC-0301`.
- **Exclusions**: Speculative claims.

### TC-0010: Master Plan & Roadmap Documentation
- **Status**: DONE - VERIFIED (Documented in `docs/roadmap/MASTER_PLAN.md`)
- **Priority**: P0 | **Area**: Planning
- **Scope**: Document vision, core architectural pillars, and milestones.
- **Exclusions**: Implementation code.

### TC-0011: Dependency Vulnerability Scanning Configuration
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Security
- **Scope**: Integrate Dependabot configuration for npm, maven, and cmake dependencies.
- **Exclusions**: Third-party proprietary security scanners.

### TC-0012: Semantic Versioning & Release Checklist
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Release
- **Scope**: Create `docs/RELEASE_CHECKLIST.md` detailing version bumping, changelog generation, and tagging procedures.
- **Exclusions**: Automated npm/maven package publishing.

### TC-0013: Documentation Site Setup (VitePress / Docusaurus)
- **Status**: NOT STARTED
- **Priority**: P3 | **Area**: Documentation
- **Scope**: Setup lightweight markdown documentation renderer for developer guides.
- **Exclusions**: Interactive playground widgets.

### TC-0014: C++ Static Analysis Configuration (Clang-Tidy)
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Quality Gates
- **Scope**: Add `.clang-tidy` config enforcing modern C++ core guidelines and performance checks.
- **Exclusions**: Commercial coverity scanners.

### TC-0015: Frontend Bundling & Size Limit Budgeting
- **Status**: NOT STARTED
- **Priority**: P3 | **Area**: Frontend
- **Scope**: Configure bundlesize / vite-plugin-visualizer to enforce max asset thresholds.
- **Exclusions**: Microfrontends.
---

## Epic 2: C++ Machine Learning Engine Core (TC-0101 - TC-0130)

### TC-0101: Core N-Dimensional Tensor Data Structure
- **Status**: PARTIAL (Implemented in PR #1 / commit `215c300`, verified unit tests pass)
- **Priority**: P0 | **Area**: C++ Engine
- **Scope**: `Tensor` class with dynamic shape, strides, data buffer, element access (`at`, `operator[]`), and flat index calculations.
- **Remaining**: Arbitrary rank slicing and shared reference-counted storage views.

### TC-0102: Tensor Initializers & Factory Methods
- **Status**: DONE - VERIFIED (Implemented in `engine/src/tensor.cpp`, tests pass in `test_tensor.cpp`)
- **Priority**: P1 | **Area**: C++ Engine
- **Scope**: `zeros`, `ones`, `uniform`, and `xavier_uniform` weight initializers.
- **Exclusions**: He-normal / Kaiming normal initializers.

### TC-0103: He / Kaiming Normal Initializer
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine
- **Scope**: Implement Kaiming He normal weight distribution for deep ReLU networks (`std::normal_distribution`).
- **Exclusions**: Non-Gaussian distributions.

### TC-0104: Element-Wise Arithmetic Operations (Add, Sub, Mul)
- **Status**: DONE - VERIFIED (Implemented in `engine/src/tensor.cpp`, verified in `test_tensor.cpp`)
- **Priority**: P0 | **Area**: C++ Engine
- **Scope**: Same-shape element-wise addition, subtraction, multiplication, and scalar scaling.
- **Exclusions**: Multi-axis broadcasting.

### TC-0105: Multi-Dimensional Numpy-Style Broadcasting
- **Status**: DONE - VERIFIED (Implemented in engine/src/tensor.cpp, 6 test suites pass in 	est_tensor.cpp)
- **Priority**: P1 | **Area**: C++ Engine
- **Scope**: General broadcasting supporting trailing dimension expansion (e.g. `[B, C] + [1, C]`).
- **Exclusions**: In-place broadcasting on non-contiguous memory.

### TC-0106: 2D Matrix Multiplication (GEMM)
- **Status**: DONE - VERIFIED (Implemented in `engine/src/tensor.cpp`, verified in `test_tensor.cpp`)
- **Priority**: P0 | **Area**: C++ Engine
- **Scope**: Standard 2D GEMM matrix multiplication (`A(M,K) * B(K,N) = C(M,N)`).
- **Exclusions**: AVX2 SIMD intrinsics and multi-threading.

### TC-0107: SIMD AVX2 Vectorized Matrix Multiplication
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: C++ Engine / Perf
- **Scope**: AVX2 8-wide float FMA (`_mm256_fmadd_ps`) inner loop acceleration for GEMM.
- **Exclusions**: AVX-512.

### TC-0108: Multi-Threaded Tiled GEMM using std::thread / OpenMP
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: C++ Engine / Perf
- **Scope**: Tile-based matrix partitioning across CPU cores for large batches.
- **Exclusions**: Distributed node parallelization.

### TC-0109: 2D Matrix Transpose
- **Status**: DONE - VERIFIED (Implemented in `engine/src/tensor.cpp`, verified in `test_tensor.cpp`)
- **Priority**: P1 | **Area**: C++ Engine
- **Scope**: Out-of-place 2D matrix transposition.
- **Exclusions**: In-place non-square transposition.

### TC-0110: High-Dimensional Axis Permutation / Transpose
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine
- **Scope**: Arbitrary dimension permutation for N-D tensors (`transpose({0, 2, 1})`).
- **Exclusions**: Continuous restriding without copy.

### TC-0111: Tensor Reshape & Flatten
- **Status**: DONE - VERIFIED (Implemented in `engine/src/tensor.cpp`)
- **Priority**: P1 | **Area**: C++ Engine
- **Scope**: Shape modification preserving total capacity.
- **Exclusions**: Reshaping non-contiguous memory slices.

### TC-0112: Tensor Reduction Operations (Sum, Mean, Argmax)
- **Status**: PARTIAL (Implemented global `sum` and `mean`, argmax missing)
- **Priority**: P1 | **Area**: C++ Engine
- **Scope**: Global and axis-wise reductions (`sum(axis)`, `mean(axis)`, `argmax(axis)`).
- **Exclusions**: Median / variance.

### TC-0113: Numerical Activation: ReLU & Derivative
- **Status**: DONE - VERIFIED (Implemented in `engine/src/tensor.cpp`, verified in `test_tensor.cpp`)
- **Priority**: P0 | **Area**: C++ Engine
- **Scope**: Forward ReLU and gradient backprop step (`relu_backward`).
- **Exclusions**: Leaky ReLU.

### TC-0114: Numerical Activation: LeakyReLU & Derivative
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine
- **Scope**: Parametric/Leaky ReLU supporting configurable alpha negative slope.
- **Exclusions**: ELU.

### TC-0115: Numerical Activation: Sigmoid & Derivative
- **Status**: DONE - VERIFIED (Implemented in `engine/src/tensor.cpp`)
- **Priority**: P1 | **Area**: C++ Engine
- **Scope**: Numerically stable sigmoid function and backward gradient calculation.
- **Exclusions**: Hard sigmoid.

### TC-0116: Numerical Activation: Tanh & Derivative
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine
- **Scope**: Hyperbolic tangent activation and analytical derivative.
- **Exclusions**: Hard tanh.

### TC-0117: Numerical Activation: GeLU (Gaussian Error Linear Unit)
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine
- **Scope**: Fast polynomial approximation of GeLU activation.
- **Exclusions**: Exact error function (`std::erf`) slow path.

### TC-0118: Numerically Stable Softmax (Log-Sum-Exp Trick)
- **Status**: DONE - VERIFIED (Implemented in `engine/src/tensor.cpp`, verified in `test_tensor.cpp`)
- **Priority**: P0 | **Area**: C++ Engine
- **Scope**: 2D Softmax with maximum subtraction to prevent exp overflow.
- **Exclusions**: Multi-axis softmax.

### TC-0119: Tensor Slicing & Sub-tensor Indexing
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine
- **Scope**: Basic range indexing (`tensor.slice(dim, start, end)`).
- **Exclusions**: Python-style slice notation parser.

### TC-0120: Tensor Concatenation along Specified Dimension
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine
- **Scope**: Joining multiple tensors along an existing axis.
- **Exclusions**: Stacking along new axis.

### TC-0121: In-Place Tensor Arithmetic (add_, mul_)
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine / Perf
- **Scope**: In-place operations to minimize temporary memory allocations during training loops.
- **Exclusions**: In-place operations on shared views.

### TC-0122: Reference-Counted Tensor Buffer Storage
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine
- **Scope**: Encapsulate raw data in `std::shared_ptr<Storage>` enabling zero-copy shallow clones and views.
- **Exclusions**: Custom slab allocator.

### TC-0123: CMake Build System Optimization & Target Partitioning
- **Status**: PARTIAL (Single static library target in PR #1)
- **Priority**: P1 | **Area**: Build
- **Scope**: Split CMake into `tensorcore_core`, `tensorcore_ops`, and `tensorcore_nn` modular targets.
- **Exclusions**: CPack installers.

### TC-0124: Benchmark Suite for Matrix Multiplication & Activations
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine / Perf
- **Scope**: High-resolution timer harness measuring GFLOPS across matrix sizes (64x64 to 2048x2048).
- **Exclusions**: Micro-benchmarking cache line misses.

### TC-0125: Engine Serialization to Custom Binary Format
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine
- **Scope**: Binary stream writer saving tensor shapes, datatypes, and raw byte buffers to disk.
- **Exclusions**: Legacy pickle formats.

### TC-0126: Engine Deserialization from Binary Format
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine
- **Scope**: Binary stream reader parsing tensor buffers with checksum validation.
- **Exclusions**: Deserialization of corrupted streams.

### TC-0127: C++ Engine Exception Hierarchy & Diagnostics
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: C++ Engine
- **Scope**: Custom `TensorException`, `ShapeMismatchException`, and `OutOfMemoryException` classes with stack trace info.
- **Exclusions**: POSIX signal handlers.

### TC-0128: Memory Leak & ASAN (AddressSanitizer) Test Suite
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: C++ Engine / QA
- **Scope**: CMake build target with `-fsanitize=address,undefined` validating zero memory leaks during test execution.
- **Exclusions**: Valgrind on Windows.

### TC-0129: Multi-Device Tensor Class (`Device` abstraction)
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: C++ Engine
- **Scope**: Introduce `Device` enum (`CPU`, `VULKAN_GPU`, `WEBGPU`) and device pointer abstraction.
- **Exclusions**: Direct memory mapping across PCIe.

### TC-0130: Engine Version Header & Build Metadata Inspection
- **Status**: NOT STARTED
- **Priority**: P3 | **Area**: C++ Engine
- **Scope**: Generate `version.hpp` with git commit hash, compiler flags, and target architecture strings.
- **Exclusions**: Dynamic runtime patching.
---

## Epic 3: Autograd, Neural Network Layers & Optimizers (TC-0201 - TC-0225)

### TC-0201: Layer Abstract Base Class
- **Status**: DONE - VERIFIED (Implemented in `engine/include/tensorcore/layers.hpp` / PR #2)
- **Priority**: P0 | **Area**: Autograd
- **Scope**: Virtual `forward` and `backward` methods and name introspection.
- **Exclusions**: Dynamic graph tracing.

### TC-0202: Linear (Fully Connected Dense) Layer
- **Status**: DONE - VERIFIED (Implemented in `engine/src/layers.cpp` / PR #2)
- **Priority**: P0 | **Area**: Layers
- **Scope**: Linear layer with weights, bias, forward affine transform, and backward gradient computation.
- **Exclusions**: Weight norm regularization.

### TC-0203: Batch Normalization Layer (1D)
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Layers
- **Scope**: Running mean/variance tracking, batch normalization, scale (`gamma`), shift (`beta`), and backward pass.
- **Exclusions**: 2D Spatial Batch Norm.

### TC-0204: Dropout Layer
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Layers
- **Scope**: Inverted dropout layer with configurable drop rate and train/eval mode toggle.
- **Exclusions**: Alpha dropout.

### TC-0205: Sequential Container Module
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Layers
- **Scope**: Sequential wrapper managing an ordered list of `std::shared_ptr<Layer>` executing cascaded forward/backward passes.
- **Exclusions**: Multi-branch residual connections.

### TC-0206: Dynamic Reverse-Mode Autograd Tape
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Autograd
- **Scope**: General computational graph tape recording intermediate tensors and backward function closures.
- **Exclusions**: Higher-order derivatives (Hessian).

### TC-0207: Mean Squared Error (MSE) Loss Class
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Loss
- **Scope**: `MSELoss` class calculating scalar loss and analytical input gradients (`2*(pred - target)/N`).
- **Exclusions**: Huber loss.

### TC-0208: Binary Cross-Entropy (BCE) Loss Class
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Loss
- **Scope**: `BCELoss` with clipping to avoid `log(0)` undefined behavior and gradient computation.
- **Exclusions**: Label smoothing.

### TC-0209: Categorical Cross-Entropy Loss with Softmax Fusion
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Loss
- **Scope**: Fused Softmax + Cross-Entropy loss computing stable `pred - target` gradient directly.
- **Exclusions**: Sparse categorical cross entropy.

### TC-0210: Huber / Smooth L1 Loss Class
- **Status**: NOT STARTED
- **Priority**: P3 | **Area**: Loss
- **Scope**: Robust regression loss transitioning between L1 and L2 based on delta threshold.
- **Exclusions**: Quantile loss.

### TC-0211: SGD Optimizer with Momentum & Nesterov
- **Status**: PARTIAL (SGD with standard momentum implemented in `engine/include/tensorcore/optimizers.hpp` / PR #2)
- **Priority**: P0 | **Area**: Optimizers
- **Scope**: Enhance `SGDOptimizer` to support configurable Nesterov accelerated momentum and damping.
- **Exclusions**: Second-order optimizers.

### TC-0212: Adam Optimizer (Adaptive Moment Estimation)
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Optimizers
- **Scope**: Adam optimizer with first/second moment tracking (`m`, `v`), bias correction, and epsilon stabilizer.
- **Exclusions**: AMSGrad.

### TC-0213: AdamW Optimizer with Decoupled Weight Decay
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Optimizers
- **Scope**: AdamW implementation decoupling L2 weight decay from gradient updates.
- **Exclusions**: Adafactor.

### TC-0214: RMSProp Optimizer
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Optimizers
- **Scope**: Root Mean Square Propagation optimizer with moving average squared gradients.
- **Exclusions**: Centered RMSProp.

### TC-0215: Learning Rate Scheduler: StepLR
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Optimizers
- **Scope**: Step decay decaying learning rate by gamma every step_size epochs.
- **Exclusions**: Custom warm restarts.

### TC-0216: Learning Rate Scheduler: CosineAnnealingLR
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Optimizers
- **Scope**: Cosine annealing learning rate schedule between initial lr and eta_min.
- **Exclusions**: Cyclic schedules with restart.

### TC-0217: Gradient Clipping by Value
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Optimizers
- **Scope**: Element-wise gradient clamp preventing gradient explosion outside `[-clip_val, clip_val]`.
- **Exclusions**: Norm clipping.

### TC-0218: Gradient Clipping by Global L2 Norm
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Optimizers
- **Scope**: Scale all parameter gradients uniformly if total L2 norm exceeds threshold.
- **Exclusions**: Per-layer adaptive norm clipping.

### TC-0219: Finite Differences Numerical Gradient Verification Harness
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Testing
- **Scope**: Automated test comparing analytical backprop gradients against `(f(x+eps) - f(x-eps))/(2*eps)` across all layers.
- **Exclusions**: Higher-order symbolic differentiation.

### TC-0220: Training Convergence Test on Non-Linear XOR Problem
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Testing
- **Scope**: 2-layer MLP (2 -> 4 -> 1) learning XOR function to 100% accuracy within 200 epochs.
- **Exclusions**: Large datasets.

### TC-0221: Training Convergence Test on Multiclass Iris Dataset
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Testing
- **Scope**: Verification test training 3-class classifier achieving >90% accuracy on Iris benchmark.
- **Exclusions**: Computer vision benchmarks.

### TC-0222: Parameter Zero-Grad Utility
- **Status**: DONE - VERIFIED (Implemented on `Linear` in PR #2)
- **Priority**: P1 | **Area**: Layers
- **Scope**: Recursively reset all parameter gradients to zero across layers.
- **Exclusions**: None.

### TC-0223: Model Evaluation Mode (eval / train toggle)
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Layers
- **Scope**: Model-level flag disabling dropout and freezing batch norm statistics during evaluation.
- **Exclusions**: Mixed-mode subgraphs.

### TC-0224: Metric Calculator: Accuracy, Precision, Recall, F1
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Metrics
- **Scope**: Multi-class confusion matrix, precision, recall, and macro/micro F1 calculation in C++.
- **Exclusions**: ROC-AUC curve rendering.

### TC-0225: Early Stopping Callback
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Training
- **Scope**: Monitor validation loss with patience counter and restore best weights upon stagnation.
- **Exclusions**: Model checkpointing to S3.
---

## Epic 4: Java Spring Boot Backend Architecture (TC-0301 - TC-0330)

### TC-0301: Spring Boot 3 Project Scaffolding
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Backend
- **Scope**: Maven `pom.xml` configured with Spring Boot 3.3.x, Spring Web, Data JPA, Security, and PostgreSQL driver.
- **Exclusions**: Cloud deployment plugins.

### TC-0302: Flyway Database Migration Framework Setup
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Database
- **Scope**: Configure Flyway migration directory `src/main/resources/db/migration/` and baseline script `V1__init_schema.sql`.
- **Exclusions**: Dynamic database creation scripts.

### TC-0303: User Entity, Repository & Security Model
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Security
- **Scope**: User account entity with BCrypt password hashing, role permissions, and JPA repository.
- **Exclusions**: OAuth2 social login.

### TC-0304: JWT Stateless Authentication Filter & Token Provider
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Security
- **Scope**: Generate and validate HMAC-SHA256 JWT tokens with expiration handling and Spring Security filter integration.
- **Exclusions**: Refresh token rotation in Redis.

### TC-0305: User Workspace Entity & Multi-Tenant Data Isolation
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Security
- **Scope**: `Workspace` entity mapping 1-to-N with users, ensuring project and dataset queries are scoped to active workspace.
- **Exclusions**: Cross-workspace public sharing.

### TC-0306: Project Management REST Endpoints
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: API
- **Scope**: CRUD REST controller for machine learning projects (`/api/v1/projects`).
- **Exclusions**: Project version branching.

### TC-0307: Dataset Entity & File Storage Metadata Catalog
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Datasets
- **Scope**: Database entity tracking dataset filename, storage path, row count, byte size, and upload timestamp.
- **Exclusions**: S3 cloud storage adapters.

### TC-0308: Streaming CSV Multipart File Upload Endpoint
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Datasets
- **Scope**: Stream incoming CSV files up to 100MB directly to disk storage with file size validation.
- **Exclusions**: Distributed multi-part S3 upload.

### TC-0309: Fast CSV Header & Schema Inference Engine
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Datasets
- **Scope**: Stream-read initial 1,000 CSV rows to infer column types (Continuous Numeric, Categorical String, Boolean).
- **Exclusions**: Complex nested JSON in CSV cells.

### TC-0310: Dataset Descriptive Statistics Calculator
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Datasets
- **Scope**: Calculate mean, standard deviation, median, min, max, missing values, and unique counts per column.
- **Exclusions**: Heavy histogram binning > 1,000 bins.

### TC-0311: Missing Value Imputation Service
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Datasets
- **Scope**: Support mean, median, mode imputation or row dropping for missing values in CSV data.
- **Exclusions**: KNN-based imputation.

### TC-0312: Feature Scaling Service (Min-Max & Standard Z-Score)
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Datasets
- **Scope**: Normalize continuous columns via min-max `[0, 1]` or standard normal `(x - mu) / sigma`.
- **Exclusions**: Quantile transformation.

### TC-0313: Categorical One-Hot Encoding Service
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Datasets
- **Scope**: Convert categorical string columns to binary indicator columns with unknown handling.
- **Exclusions**: Target encoding.

### TC-0314: Deterministic Train / Validation / Test Splitter
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Datasets
- **Scope**: Partition dataset into train, val, and test partitions based on configured ratios (e.g. 70/15/15) with seed.
- **Exclusions**: K-fold cross validation.

### TC-0315: Stratified Splitter for Imbalanced Categorical Targets
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Datasets
- **Scope**: Partition datasets maintaining proportional class label distributions in each split.
- **Exclusions**: Multi-label stratification.

### TC-0316: Model Configuration Specification & Entity
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Models
- **Scope**: JSON schema and JPA entity storing layer types, hidden units, activation functions, loss, and optimizer settings.
- **Exclusions**: Automated architecture search.

### TC-0317: Training Job Entity & State Machine
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Training
- **Scope**: Job tracking entity with states `QUEUED`, `INITIALIZING`, `TRAINING`, `PAUSED`, `COMPLETED`, `FAILED`, `CANCELLED`.
- **Exclusions**: Distributed scheduler across clusters.

### TC-0318: Native C++ Subprocess Launcher & Lifecycle Manager
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Engine Interop
- **Scope**: Java `ProcessBuilder` launcher executing native C++ engine worker binary with PID tracking and watchdog timer.
- **Exclusions**: Kubernetes pod operator.

### TC-0319: Framed IPC Communication Protocol (Java <-> C++)
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Engine Interop
- **Scope**: Standard IO / Socket framed binary protocol transferring batch float buffers and reading JSON metric frames.
- **Exclusions**: Shared memory circular ring buffer.

### TC-0320: Native Engine Process Heartbeat & Crash Recovery
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Engine Interop
- **Scope**: Detect sudden worker termination (segfault / exit code != 0), record crash logs, and transition job to `FAILED`.
- **Exclusions**: Automatic warm checkpoint restart.

### TC-0321: Training Control API: Pause, Resume, Stop
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Training
- **Scope**: Send IPC control signals to pause/resume or terminate running training subprocess.
- **Exclusions**: Step-by-step debugger stepping.

### TC-0322: WebSocket STOMP Broker Configuration
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Telemetry
- **Scope**: Configure Spring WebSocket message broker at `/ws` supporting SockJS fallback and STOMP channels.
- **Exclusions**: RabbitMQ external broker relay.

### TC-0323: Real-Time Telemetry Broadcasting Service
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Telemetry
- **Scope**: Ingest per-epoch metric frames from C++ engine and publish to topic `/topic/training/{jobId}`.
- **Exclusions**: Client-side video streaming.

### TC-0324: Training Run Checkpoint Persistence Service
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Checkpoints
- **Scope**: Save model weights binary artifacts and training metadata to disk cataloged in PostgreSQL.
- **Exclusions**: Distributed S3 replication.

### TC-0325: Single-Sample Inference Execution Service
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Inference
- **Scope**: Load checkpointed model into native worker, execute forward pass on input vector, and return prediction probabilities.
- **Exclusions**: Batch inference pipeline > 10,000 rows.

### TC-0326: Global Exception Handling & Error Response Standards
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: API
- **Scope**: `@ControllerAdvice` mapping validation errors, resource not found, and worker exceptions to RFC 7807 Problem Details.
- **Exclusions**: Custom SOAP fault handlers.

### TC-0327: Swagger / OpenAPI 3.0 Documentation Generation
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Documentation
- **Scope**: Integrate `springdoc-openapi` exposing interactive Swagger UI at `/swagger-ui.html`.
- **Exclusions**: Client SDK code generator.

### TC-0328: Backend Unit & Mockito Test Suite
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Testing
- **Scope**: Comprehensive unit tests covering DatasetService, PreprocessingService, and SecurityFilters.
- **Exclusions**: End-to-end multi-container tests.

### TC-0329: Backend Integration Test Suite with Testcontainers
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Testing
- **Scope**: Spring Boot integration tests spinning up containerized PostgreSQL via Testcontainers.
- **Exclusions**: Cloud deployment testing.

### TC-0330: CORS Configuration & Security Headers
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Security
- **Scope**: Strict CORS configuration permitting Vite frontend origin (`localhost:5173`), CSP headers, and XSS protection.
- **Exclusions**: Wildcard CORS allowances.
---

## Epic 5: Workstation Frontend Studio (TC-0401 - TC-0430)

### TC-0401: Frontend Theme Tokens & Workstation Layout
- **Status**: DONE - VERIFIED (Implemented in `frontend/src/index.css` & `App.tsx` / PR #3)
- **Priority**: P0 | **Area**: Frontend UI
- **Scope**: Dark workstation theme tokens (`#07090e`), responsive multi-panel layout, Google Fonts integration.
- **Exclusions**: Dynamic light theme switcher.

### TC-0402: Project Workspace Context & Switcher
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Frontend UI
- **Scope**: Workspace selector dropdown and global Zustand state store managing active project context.
- **Exclusions**: User profile picture upload.

### TC-0403: Dataset Upload Drag-and-Drop Component
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Datasets UI
- **Scope**: Drag-and-drop file upload zone supporting CSV files with upload progress bar.
- **Exclusions**: Folder recursive upload.

### TC-0404: Virtualized Tabular Dataset Explorer Grid
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Datasets UI
- **Scope**: High-performance virtualized table rendering up to 10,000 CSV rows smoothly without DOM sluggishness.
- **Exclusions**: Full in-browser spreadsheet formula editor.

### TC-0405: Column Summary Statistics Cards & Histogram Preview
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Datasets UI
- **Scope**: Modal/drawer showing per-column statistics (mean, min, max, null count) and micro histogram.
- **Exclusions**: 3D scatter plots.

### TC-0406: Feature Target & Preprocessing Selector Panel
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Datasets UI
- **Scope**: Interactive checklist to select target label column, include/exclude features, and assign imputation strategies.
- **Exclusions**: Automated feature synthesis.

### TC-0407: Train / Validation / Test Split Interactive Slider
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Datasets UI
- **Scope**: Linked percentage sliders for train/val/test splits summing to 100% with sample count preview.
- **Exclusions**: Non-linear split distribution curves.

### TC-0408: Neural Network Architecture DAG Builder
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Architecture UI
- **Scope**: Layer configuration list/graph allowing users to add Dense layers, set hidden unit counts, and pick activations.
- **Exclusions**: Freeform drag-and-drop node graph wires.

### TC-0409: Architecture Shape Consistency Live Validator
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Architecture UI
- **Scope**: Client-side validation ensuring layer output dimension matches subsequent layer input dimension.
- **Exclusions**: Automatic dimension inferencing.

### TC-0410: Hyperparameter Configuration Panel
- **Status**: DONE - VERIFIED (Implemented in `frontend/src/App.tsx` / PR #3)
- **Priority**: P1 | **Area**: Training UI
- **Scope**: Config inputs for learning rate, batch size, momentum, weight decay, and max epochs.
- **Exclusions**: Genetic algorithm hyperparameter sliders.

### TC-0411: HTML5 Canvas 60 FPS Real-Time Loss Visualizer
- **Status**: DONE - VERIFIED (Implemented in `frontend/src/App.tsx` / PR #3)
- **Priority**: P0 | **Area**: Visualization
- **Scope**: High-performance canvas chart rendering train vs. validation loss trajectory without DOM overhead.
- **Exclusions**: WebGL 3D loss surface terrain.

### TC-0412: WebSocket Client Hook for Live Telemetry
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Frontend Infra
- **Scope**: Custom React hook connecting to STOMP WebSocket topic, updating loss curves and metric counters.
- **Exclusions**: Bluetooth telemetry.

### TC-0413: Interactive Training Control Bar (Start, Pause, Stop, Reset)
- **Status**: PARTIAL (UI buttons and simulation state loop implemented in PR #3; lacks live backend API calls)
- **Priority**: P0 | **Area**: Training UI
- **Scope**: Connect UI start/pause/stop buttons to backend training execution endpoints.
- **Exclusions**: Batch script execution.

### TC-0414: Diagnostics Terminal & Log Stream Pane
- **Status**: DONE - VERIFIED (Implemented in `frontend/src/App.tsx` / PR #3)
- **Priority**: P1 | **Area**: Frontend UI
- **Scope**: Fixed-height auto-scrolling terminal window displaying formatted engine log statements.
- **Exclusions**: Shell bash interactive prompt.

### TC-0415: Confusion Matrix Heatmap Component
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Visualization
- **Scope**: Grid heatmap visualizing true vs. predicted class labels with color gradient density.
- **Exclusions**: Multi-label Venn diagrams.

### TC-0416: Single-Sample Inference Playground Panel
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Inference UI
- **Scope**: Dynamic form generating input inputs matching dataset schema, running single inference, and displaying class bars.
- **Exclusions**: Camera live feed capture.

### TC-0417: Model Checkpoints & Experiment Comparison Table
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Experiments UI
- **Scope**: Tabular view comparing past training runs (loss, accuracy, hyperparameters, duration) with best-model tag.
- **Exclusions**: Cross-experiment parameter diff visualizer.

### TC-0418: Model Weight Distribution Histogram Component
- **Status**: NOT STARTED
- **Priority**: P3 | **Area**: Visualization
- **Scope**: Micro-chart displaying weight and gradient distribution histograms per layer to monitor vanishing gradients.
- **Exclusions**: 3D tensor slice view.

### TC-0419: Model Export Modal (JSON & Binary Download)
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Export UI
- **Scope**: Modal allowing download of trained weights, architecture JSON, and deployment inference snippet.
- **Exclusions**: Cloud model registry deployment.

### TC-0420: Hardware Acceleration Selector & Status Badge
- **Status**: PARTIAL (UI selector badge present in PR #3; lacks real hardware detection logic)
- **Priority**: P1 | **Area**: Frontend UI
- **Scope**: Execution backend picker displaying genuine verified status (AVX2 CPU vs WebGPU).
- **Exclusions**: Quantum backend mock.

### TC-0421: Client-Side WebGPU Direct Compute Adapter Hook
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Edge ML
- **Scope**: React hook querying `navigator.gpu` for WebGPU device adapter and reporting real capabilities.
- **Exclusions**: Fallback to software SwiftShader.

### TC-0422: Toast Notification System for Errors & Alerts
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Frontend UI
- **Scope**: Lightweight non-blocking toast notifications for network failures, validation errors, and completion alerts.
- **Exclusions**: System desktop push notifications.

### TC-0423: Responsive Navigation Drawer & Mobile Adaptation
- **Status**: NOT STARTED
- **Priority**: P3 | **Area**: Frontend UI
- **Scope**: Collapse sidebar to responsive slide-out drawer on smaller viewports.
- **Exclusions**: Native mobile app wrapper.

### TC-0424: Keyboard Shortcuts for Workstation Workflow
- **Status**: NOT STARTED
- **Priority**: P3 | **Area**: Frontend UX
- **Scope**: Support `Ctrl+Enter` to start training, `Space` to pause, `Esc` to close modals.
- **Exclusions**: Custom keybinding mapper.

### TC-0425: Light / Dark Theme Contrast Compliance Review
- **Status**: NOT STARTED
- **Priority**: P3 | **Area**: Frontend UI
- **Scope**: Verify WCAG AA color contrast compliance across text and data visualization elements.
- **Exclusions**: High-contrast black/white mode.

### TC-0426: Frontend Unit Testing Setup (Vitest + React Testing Library)
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Testing
- **Scope**: Configure Vitest for fast component unit testing with mock DOM.
- **Exclusions**: Cypress full browser testing.

### TC-0427: Canvas Chart Performance Benchmark
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Frontend / Perf
- **Scope**: Automated test measuring canvas render frame time under 1,000 incoming telemetry points.
- **Exclusions**: Mobile GPU profiling.

### TC-0428: User Authentication & Login Screen
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Auth UI
- **Scope**: High-fidelity login and workspace selection form storing JWT in secure local storage.
- **Exclusions**: Two-factor authenticator app integration.

### TC-0429: User Settings & Hardware Diagnostic Page
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Frontend UI
- **Scope**: Detailed page displaying host environment specs, CPU SIMD flags, detected GPUs, and backend version.
- **Exclusions**: BIOS flashing tool.

### TC-0430: Frontend Production Bundle Code Splitting & Chunk Optimization
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Build
- **Scope**: Configure Vite manual chunks splitting heavy visualization libraries from main app chunk.
- **Exclusions**: Dynamic remote module federation.
---

## Epic 6: Hardware Acceleration, Verification & Model Export (TC-0501 - TC-0520)

### TC-0501: Hardware Capability Detection & Verification Probe
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Hardware
- **Scope**: Native C++ and browser WebGPU diagnostic probe executing a mathematical test matrix to verify hardware honestly.
- **Exclusions**: Overclocking utilities.

### TC-0502: Native Vulkan Compute Subsystem Initialization
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: GPU Engine
- **Scope**: Initialize Vulkan instance, select physical device with compute queue, and verify SPIR-V shader compilation.
- **Exclusions**: Graphics rasterization pipeline.

### TC-0503: Vulkan Compute Shader for Matrix Multiplication
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: GPU Engine
- **Scope**: Write and compile GLSL/SPIR-V compute shader for 2D GEMM execution on Vulkan compute queues.
- **Exclusions**: Raytracing pipelines.

### TC-0504: In-Browser WebGPU WGSL Shader for Matrix Multiplication
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: WebGPU
- **Scope**: WGSL compute shader executing parallel tiled matrix multiplication in browser via `requestAdapter`.
- **Exclusions**: WebGL 1.0 fallbacks.

### TC-0505: Numerical Parity Verification (CPU vs Vulkan vs WebGPU)
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Testing
- **Scope**: Automated test ensuring matrix results across CPU, Vulkan, and WebGPU match within float tolerance `< 1e-4`.
- **Exclusions**: BFloat16 precision modes.

### TC-0506: Model Checkpoint Format Specification
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Export
- **Scope**: Document schema for model checkpoint files containing architecture JSON, layer weights binary, and metadata.
- **Exclusions**: Legacy caffe formats.

### TC-0507: Standalone C++ Inference Engine Binary
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Inference
- **Scope**: Lightweight C++ CLI binary (`tensorcore-infer`) capable of loading checkpoint and running inference on CSV rows.
- **Exclusions**: Embedded microcontroller runtime.

### TC-0508: Python Inference Wrapper Script
- **Status**: NOT STARTED
- **Priority**: P3 | **Area**: Interop
- **Scope**: Lightweight pure-Python script parsing exported JSON weights to run inference without C++ toolchain.
- **Exclusions**: PyTorch C extensions.

### TC-0509: End-to-End Dataset Validation: MNIST Digits Classification
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Verification
- **Scope**: Full training run on 784-input MNIST dataset converging to >95% accuracy using native C++ engine.
- **Exclusions**: ResNet architectures.

### TC-0510: End-to-End Dataset Validation: California Housing Regression
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Verification
- **Scope**: Regression training run on tabular continuous features measuring MSE loss convergence.
- **Exclusions**: Geospatial visualization.

### TC-0511: System End-to-End Test Suite (Frontend -> Backend -> C++ Engine)
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Integration
- **Scope**: Automated script creating project, uploading CSV, training model, and asserting loss decreases.
- **Exclusions**: Multi-user stress load testing.

### TC-0512: Performance Profiling & GFLOPS Benchmark Report
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Performance
- **Scope**: Measure throughput across matrix operations and document in `docs/BENCHMARKS.md`.
- **Exclusions**: Proprietary hardware benchmarking.

### TC-0513: Out-of-Memory (OOM) Protection & Graceful Degradation
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Reliability
- **Scope**: Enforce batch size limits and catch memory allocation failures before process termination.
- **Exclusions**: Swap file manipulation.

### TC-0514: Production Docker Packaging (Multi-Stage Build)
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: DevOps
- **Scope**: Multi-stage `Dockerfile` compiling C++ engine, building frontend static assets, and packaging Spring Boot JAR.
- **Exclusions**: Helm charts.

### TC-0515: Single-Command Quickstart Runner Script
- **Status**: NOT STARTED
- **Priority**: P2 | **Area**: Developer Experience
- **Scope**: Root `run.ps1` and `run.sh` script to verify toolchains, build engine, and launch application in one step.
- **Exclusions**: Windows MSI installers.

### TC-0516: User Guide & Getting Started Manual
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Documentation
- **Scope**: Comprehensive step-by-step documentation in `docs/USER_GUIDE.md` covering dataset upload, model creation, and training.
- **Exclusions**: Video tutorial production.

### TC-0517: Security & Vulnerability Audit Report
- **Status**: NOT STARTED
- **Priority**: P1 | **Area**: Security
- **Scope**: Review all endpoints, file upload paths, and IPC boundaries for injection vulnerabilities; document in `docs/SECURITY.md`.
- **Exclusions**: Third-party penetration testing certification.

### TC-0518: Final Quality Gate & Release Verification
- **Status**: NOT STARTED
- **Priority**: P0 | **Area**: Release
- **Scope**: Verification checklist ensuring all tests pass, zero compiler warnings, and clean clean-branch merge readiness.
- **Exclusions**: Post-release telemetry.

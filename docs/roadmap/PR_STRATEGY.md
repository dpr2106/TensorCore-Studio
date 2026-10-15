# TensorCore Studio — PR Strategy & Branch Governance

## 1. Protected Branch Rules
- **`main` is strictly protected**:
  - Direct pushes to `main` are strictly forbidden.
  - No feature or bugfix commits are made directly on `main`.
  - Merging into `main` is performed strictly via approved, verified Pull Requests.

## 2. Branch Naming Conventions
Every branch must be short-lived, task-focused, and map directly to a unique Backlog Task ID:
- `chore/TC-XXXX-short-title`: Project setup, CI/CD, dependency updates, tooling.
- `docs/TC-XXXX-short-title`: Documentation, architecture specifications, ADR updates.
- `feat/TC-XXXX-short-title`: New functional features or capabilities.
- `fix/TC-XXXX-short-title`: Bugfixes, numerical corrections, race condition fixes.
- `test/TC-XXXX-short-title`: Unit tests, integration tests, benchmark validation.
- `perf/TC-XXXX-short-title`: Performance optimizations, SIMD vectorization, cache blocking.

## 3. Pull Request Requirements (The "One Task, One PR" Rule)
1. **Branch Freshness**: Feature branches branch off from the latest verified base branch.
2. **Atomic Scope**: Exactly one task (or an explicitly bonded micro-subtask pair) per PR. Unrelated changes across frontend, backend, and C++ engine in a single PR will be rejected.
3. **Automated Verification**:
   - C++: `cmake --build build && ctest --test-dir build --output-on-failure`.
   - Java: `./mvnw clean test` (or `mvn verify`).
   - Frontend: `npm run build` with zero TypeScript compiler errors.
4. **Clean Diffs**: No tracked build artifacts (`build/`, `node_modules/`, `target/`, `.DS_Store`), generated logs, or secret keys.
5. **PR Description**: Must reference the Task ID, provide a clear summary of changes, outline test verification evidence, and include before/after screenshots for UI updates.
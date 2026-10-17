# C++ Native Engine & WebAssembly Runtime

## 1. Overview

The native engine is written in **C++17** and compiled for high-performance CPU inference:
- **Zero Heap Allocations:** Fixed-size structures and stack-allocated aligned buffers (`alignas(32)`).
- **Dual Compilation Targets:**
  1. Standalone native binary: [`cpp_engine/src/inference_cli.cpp`](file:///d:/Tactical%20CBM%20Engine/cpp_engine/src/inference_cli.cpp).
  2. WebAssembly client module: [`dashboard/tactical_geometry.wasm`](file:///d:/Tactical%20CBM%20Engine/dashboard/tactical_geometry.wasm).

---

## 2. Memory Layout & Layout Contracts

All data structures use fixed byte offsets verified by `static_assert`:

```cpp
#include "shard_format.hpp"

static_assert(sizeof(tactical::TacticalFrameRecord) == 2588, "Record size mismatch");
static_assert(offsetof(tactical::TacticalFrameRecord, node_features) == 8, "Node features offset mismatch");
static_assert(offsetof(tactical::TacticalFrameRecord, adj_mask) == 536, "Adjacency mask offset mismatch");
static_assert(offsetof(tactical::TacticalFrameRecord, player_mask) == 2472, "Player mask offset mismatch");
static_assert(offsetof(tactical::TacticalFrameRecord, c_true) == 2560, "Ground-truth concepts offset mismatch");
static_assert(offsetof(tactical::TacticalFrameRecord, y_threat) == 2584, "Threat label offset mismatch");
```

---

## 3. WebAssembly Generation (`scripts/compile_geometry_wasm.py`)

Rather than requiring large Emscripten build toolchains, the engine contains an automated pure Python WebAssembly bytecode generator that emits standard WASM 1.0 binaries directly from arithmetic primitives:

- **Functions Exported:**
  1. `c1_line_height(max_x)`: Computes normalized defensive line height.
  2. `c2_compactness(hull_area)`: Computes Convex Hull block volume ratio.
  3. `c3_width_balance(spread, avg_y, ball_y)`: Computes lateral defensive balance.
  4. `c4_pressure(min_dist, count_within_6m)`: Computes Gegenpress pressure intensity.
  5. `stage5_eval(c1, c2, c3, c4, c5, c6)`: Evaluates linear threat head dot product.

Compile via:
```bash
python scripts/compile_geometry_wasm.py
```

---

## 4. CMake Native Build System (`cpp_engine/CMakeLists.txt`)

Supports building with Microsoft Visual Studio (MSVC), GCC, or Clang:
- Integrates `FetchContent` to download prebuilt ONNX Runtime headers and binaries automatically when `ENABLE_ONNXRUNTIME=ON`.
- Supports air-gapped CI/CD builds via pre-populated `ONNXRUNTIME_ROOTDIR`.

Building the C++ unit test binary:
```bash
cmake -B build -S cpp_engine
cmake --build build --config Release
./bin/test_cpp_engine.exe
```

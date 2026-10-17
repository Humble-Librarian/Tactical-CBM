# Test Suite & Verification Contracts

## 1. Overview

The test suite in [`tests/`](file:///d:/Tactical%20CBM%20Engine/tests/) validates all layers of the Tactical CBM Engine: from low-level memory offsets and zero-copy allocations to cross-runtime mathematical parity and physical invariants.

All tests run via `pytest`:
```bash
pytest tests
```

---

## 2. Test Modules Breakdown

| Test File | Focus & Purpose | Key Invariant Asserted |
| :--- | :--- | :--- |
| [`test_schema_contracts.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_schema_contracts.py) | Memory layout contracts | Total size == 2,588 bytes, exact byte offsets for all fields |
| [`test_geometry_engine.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_geometry_engine.py) | Concept math & edge cases | Breached line collapse, collinear hull handling, ray interception |
| [`test_runtime_parity.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_runtime_parity.py) | Cross-runtime consistency | $\|\text{Python} - \text{JS}\| < 10^{-5}$ on $C_1, C_2, C_3, C_4, C_5, C_6$ |
| [`test_dataset_shards.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_dataset_shards.py) | Shards & memory allocation | Zero heap allocation ($< 500\text{ KB}$ for 5,000 frames), pointer equality |
| [`test_model_bottleneck.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_model_bottleneck.py) | GNN architecture contracts | Concept activations $\in [0, 1]$, threat logits, gradient flow |
| [`test_onnx_and_sidecar.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_onnx_and_sidecar.py) | ONNX runtime parity | PyTorch vs ONNX Runtime inference output difference $< 10^{-5}$ |
| [`test_tactical_invariants.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_tactical_invariants.py) | Coaching domain physics | High line offside trap vs deep block threat monotonicity |
| [`test_kalman_and_wasm_geometry.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_kalman_and_wasm_geometry.py) | Tracking & WebAssembly | Sprint velocity clamp $\le 10\text{ m/s}$, WASM binary header validation |

---

## 3. Continuous Integration Gates

Every commit to `main` must pass the following two deterministic verification commands:

1. **Python Unit & Parity Suite:**
   ```bash
   pytest tests
   ```
   *Gate:* Must pass 39/39 tests with zero warnings or errors.

2. **C++ Native Engine Memory & Numerical Assertions:**
   ```bash
   ./bin/test_cpp_engine.exe
   ```
   *Gate:* Must exit with code 0.

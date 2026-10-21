# Test Suite & Verification Contracts

## 1. Overview

The test suite in [`tests/`](file:///d:/Tactical%20CBM%20Engine/tests/) validates all layers of the Tactical CBM Engine: from low-level memory offsets and zero-copy allocations to cross-runtime mathematical parity, kinematic sports physics, and out-of-distribution generalization.

All tests run via `pytest`:
```bash
pytest tests
```

---

## 2. Test Modules Breakdown (118 Tests Across 17 Suites)

| Test File | Tests | Focus & Purpose | Key Invariant Asserted |
| :--- | :---: | :--- | :--- |
| [`test_schema_contracts.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_schema_contracts.py) | 4 | Memory layout contracts | Total size == 2,588 bytes, exact byte offsets for all fields |
| [`test_geometry_engine.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_geometry_engine.py) | 8 | Concept math & edge cases | Breached line collapse, collinear hull handling, ray interception |
| [`test_runtime_parity.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_runtime_parity.py) | 5 | Cross-runtime consistency | $\|\text{Python} - \text{JS}\| < 10^{-5}$ on $C_1 \dots C_6$ |
| [`test_dataset_shards.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_dataset_shards.py) | 6 | Shards & memory allocation | Zero heap allocation ($< 500\text{ KB}$ for 5,000 frames), pointer equality |
| [`test_model_bottleneck.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_model_bottleneck.py) | 3 | GNN architecture contracts | Concept activations $\in [0, 1]$, threat logits, gradient flow |
| [`test_onnx_and_sidecar.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_onnx_and_sidecar.py) | 4 | ONNX runtime parity | PyTorch vs ONNX Runtime output difference $< 10^{-5}$ |
| [`test_tactical_invariants.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_tactical_invariants.py) | 4 | Coaching domain physics | High line offside trap vs deep block threat monotonicity |
| [`test_kalman_and_wasm_geometry.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_kalman_and_wasm_geometry.py) | 6 | Tracking & WebAssembly | Sprint velocity clamp $\le 10\text{ m/s}$, WASM binary header validation |
| [`test_cbm_architecture.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_cbm_architecture.py) | 16 | Relational attention & BCIT | Sub-millisecond slider latency, 8-role taxonomy kinematics |
| [`test_baselines_reproducibility.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_baselines_reproducibility.py) | 3 | Baseline models suite | Schema integrity and reproducible metrics for Random, LR, GBDT, MLP |
| [`test_evaluation_rigor.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_evaluation_rigor.py) | 9 | Calibration & loss math | Murphy decomposition identity, Beta calibration convex bounds |
| [`test_ablations.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_ablations.py) | 9 | Systematic model ablations | 11-variant forward execution, bottleneck interpretability cost |
| [`test_cross_val.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_cross_val.py) | 4 | Statistical significance | 1,000-iteration bootstrap order bounds ($\text{CI}_{\text{low}} \le \mu \le \text{CI}_{\text{high}}$) |
| [`test_pitch_control.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_pitch_control.py) | 12 | Spearman 2018 kinematics | Ballistic TTI physics, Gumbel softmin completeness ($P_{\text{att}} + P_{\text{def}} \equiv 1.0$) |
| [`test_fan_tactical_rigor.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_fan_tactical_rigor.py) | 15 | Real-world tactical archetypes | Sacchi offside traps, Simeone low block, Klopp counter-press |
| [`test_augmentations.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_augmentations.py) | 6 | Tactical data augmentations | $D_2$ reflection identity ($\text{Mirror}(\text{Mirror}(X)) \equiv X$), ball carrier protection |
| [`test_generalization.py`](file:///d:/Tactical%20CBM%20Engine/tests/test_generalization.py) | 4 | Cross-competition OOD | Stability across 4 tournaments, flank touchline gap $\Delta_{\text{flank}} < 0.03$ |

---

## 3. Continuous Integration Gates

Every commit to `main` must pass the following two deterministic verification commands:

1. **Python Unit & Parity Suite:**
   ```bash
   pytest tests
   ```
   *Gate:* Must pass 118/118 tests with zero warnings or errors.

2. **C++ Native Engine Memory & Numerical Assertions:**
   ```bash
   ./bin/test_cpp_engine.exe
   ```
   *Gate:* Must exit with code 0.

# Tactical-CBM Engine: Master Engineering Roadmap

This master roadmap tracks all milestones across ML experimental rigor, concept bottleneck modeling, kinematic sports physics, and systems deployment.

---

## 🗺️ Dual-Track Architecture Overview

```
TRACK A: Machine Learning Rigor & Concept Modeling
[x] Phase 1: Formula Parity & Zero-Copy Fix
[x] Phase 2: Comprehensive Baselines Suite (Random, Linear, GBDT, Flat-LR, MLP)
[x] Phase 3: Evaluation Metrics, Beta Calibration & Error Forensics (A6, A7)
[ ] Phase 4: Systematic Model Ablation Studies (A2: GAT, Adj, Depth, Bottleneck Cost)
[ ] Phase 5: Training Pipeline Rigor & Deterministic Reproducibility (C1-C4)
[ ] Phase 6: Cross-Validation & Bootstrap Confidence Intervals (A4)
[ ] Phase 7: Dataset Scaling & Generalization (A5, C5)
[ ] Phase 8: Architecture Justification & Hyperparameter Sweep (A3, B1-B4)
[ ] Phase 9: Technical Paper, Portfolio & Systems Cleanup (D1, D4, E1-E3)

TRACK B: Computational Kinematics & Sports Physics (Completed)
[x] Wave 1: Pitch Bilateral Lateral Reflection Symmetry Equivariance (D2 Group Equivariance)
[x] Wave 2: Relational Spatial Radar & Active Passing Corridor Pooling
[x] Wave 3: 4-Head Relational Attention & Live Bilinear Concept Interaction Tensor (BCIT)
[x] Wave 4: Closing Velocity Kinematics, Canonical 8-Role Taxonomy & Drag-Aim HUD
[x] Wave 5: Spearman (2018) 4-Phase TTI, Gumbel Softmin Pitch Control & Lookahead Rollouts
```

---

## Track A: Machine Learning Rigor & Experimental Modeling

### [x] Phase 1: Critical Bug Fixes & Formula Synchronization
- **Scope:** D2 (Python/JS Formula Drift), D3 (Memmap Copy Bug).
- **Deliverables:**
  - Standardize $C_1$ Line Height (divide by 70.0) across `scripts/geometry.py`, `cpp_engine/include/geometry.hpp`, and `dashboard/index.html`.
  - Standardize $C_3$ and $C_4$ Gegenpress decay curve formula between Python, C++, and JavaScript.
  - Remove `.copy()` calls in `ml_trainer/dataset.py` to restore true zero-copy strided tensor views via `mode='c'`.
  - Add end-to-end mathematical parity assertions in `tests/test_runtime_parity.py` and zero-copy tests in `tests/test_dataset_shards.py`.
- **Exit Gate:** `pytest tests` passes with 39/39 tests passing, $|\text{Python} - \text{JS}| < 10^{-5}$, and memory allocation $< 500\text{ KB}$ across 5,000 frames. Verified on 2026-10-17.

---

### [x] Phase 2: Comprehensive Baselines Suite
- **Scope:** A1 (Baselines: Random, Linear, GBDT, MLP).
- **Deliverables:**
  - Create `ml_trainer/baselines.py`.
  - Train & evaluate on 574,560 modern tracking frames:
    1. Stratified Random Prior Baseline (5.23% base rate, ROC-AUC: 0.5000, PR-AUC: 0.0547).
    2. Logistic Regression on Ground-Truth Concepts (ROC-AUC: 0.8101, PR-AUC: 0.2434, 0.09 us).
    3. LightGBM GBDT on Ground-Truth Concepts (ROC-AUC: 0.8293, PR-AUC: 0.3029, 1.73 us).
    4. Logistic Regression on Flattened 132-dim Node Features (ROC-AUC: 0.8367, PR-AUC: 0.2958, 0.69 us).
    5. Mean-Pooled 2-Layer MLP on Node Features (ROC-AUC: 0.8351, PR-AUC: 0.3095, 0.08 us).
  - Export structured benchmark results to `results/baselines_table.md` and `results/baselines_summary.json`.
- **Exit Gate:** Single-command execution `python ml_trainer/baselines.py` and unit test `tests/test_baselines_reproducibility.py` passing 42/42 tests. Verified on 2026-10-17.

---

### [x] Phase 3: Comprehensive Evaluation Metrics & Error Attribution
- **Scope:** A6 (PR-AUC, Calibration, Brier Score, Confusion Matrix), A7 (Error Analysis & Attention Extraction).
- **Deliverables:**
  - Create `ml_trainer/evaluation.py`:
    - Precision-Recall Curve (PR-AUC / Average Precision) for class-imbalanced evaluation.
    - Reliability Diagram & Expected Calibration Error (ECE) via Murphy 3-component decomposition.
    - Beta Calibration (Kull et al., 2017) with strict L-BFGS-B convex bounds ($a \ge 0, b \ge 0$), Platt Sigmoidal Scaling, and Isotonic Regression.
    - Bilinear Concept Interaction Head (BCIT) modeling 15 pairwise tactical synergies.
    - Cost-Sensitive $F_{0.5}$ decision threshold optimizer.
  - Create `ml_trainer/error_analysis.py`:
    - Categorize top 50 false positives and false negatives (transition vs. settled block vs. flank overload).
    - Tracking visibility correlation ($N_{\text{vis}}$) across correct vs. error frames.
    - Output `results/error_analysis.md` and `results/top_errors.json`.
  - Create `scripts/run_phase_3_evaluation.py`:
    - End-to-end evaluation pipeline benchmarking calibration techniques.
    - Generate diagnostic plots (`results/plots/reliability_diagram.png`, `results/plots/pr_curve.png`).
    - Output `results/calibration_benchmark.md`.
  - Create unit test suite `tests/test_evaluation_rigor.py` (9/9 tests passing).
- **Exit Gate:** Brier Score drops from 0.1649 to 0.0454 (beating random baseline 0.0517), ECE drops from 30.73% to 0.31%, ROC-AUC ranking strictly preserved, and 51/51 pytest tests passing in 12.80s. Verified on 2026-10-17.

---

### [ ] Phase 4: Systematic Model Ablation Studies
- **Scope:** A2 (Ablations: GAT, Adjacency, Masking, Depth, Latent Dimension, Pooling, Bottleneck Cost).
- **Deliverables:**
  - Create `ml_trainer/ablations.py`.
  - Evaluate 8 targeted model variants:
    1. **No GAT** (bypass graph attention; direct node projection to concept head).
    2. **Dense Graph Attention** (No Delaunay adjacency mask; dense all-to-all attention).
    3. **Unmasked Tokens** (No player mask; evaluates resilience to padding artifact noise).
    4. **GAT Layer Depth** (1 layer vs. 2 layers vs. 3 layers).
    5. **Latent Dimension Sweep** ($\{16, 32, 64, 128\}$).
    6. **Monolithic Global Pooling** (No team separation; unconditioned global mean pool vs. $h_{\text{def}} \oplus h_{\text{att}}$).
    7. **Unconstrained Direct GNN** ($X \to Y$ directly without concept bottleneck, measuring the exact AUC "tax" of interpretability).
    8. **Linear Threat Head vs. Bilinear BCIT Synergy Head** (measuring the contribution of pairwise tactical interactions).
  - Export formatted markdown results to `results/ablation_table.md` and `results/ablation_summary.json`.
  - Create automated pytest assertions in `tests/test_ablations.py`.
- **Exit Gate:** Full ablation grid executed reproducibly with concept $R^2$ and ROC/PR-AUC deltas confirming necessity of each structural choice.

---

### [ ] Phase 5: Training Pipeline Rigor & Deterministic Reproducibility
- **Scope:** C1 (Loss Weighting Schedule), C2 (pos_weight Class Imbalance Tuning), C3 (Early Stopping & Learning Curves), C4 (Global Random Seeds).
- **Deliverables:**
  - Update `ml_trainer/train.py`:
    - Strict deterministic seeding across `torch`, `numpy`, `random`, and `cudnn.deterministic = True`.
    - Implement patience-based early stopping monitoring validation PR-AUC.
    - Export training/validation learning curves to `results/plots/learning_curves.png`.
    - Compute `pos_weight` dynamically based on exact dataset class ratio ($N_{\text{neg}} / N_{\text{pos}}$) vs. hand-tuned values.
    - Formal mathematical justification for staged concept-to-threat loss weighting.
- **Exit Gate:** `python ml_trainer/train.py` runs with early stopping and logs convergence curves with verified multi-seed reproducibility.

---

### [ ] Phase 6: Cross-Validation & Statistical Significance
- **Scope:** A4 (Leave-One-Match-Out CV, Bootstrap Confidence Intervals).
- **Deliverables:**
  - Create `ml_trainer/cross_val.py`.
  - Implement Leave-One-Match-Out Cross-Validation (LOMOCV) across all dataset matches.
  - Report Mean $\pm$ Standard Deviation for ROC-AUC, PR-AUC, and Brier Score across folds.
  - Implement 1,000-sample non-parametric percentile bootstrap to compute formal 95% Confidence Intervals:
    $$\text{ROC-AUC} = \mu \pm 1.96 \cdot \text{SE} \quad [95\%\text{ CI}]$$
  - Output report to `results/cross_val_results.md`.
- **Exit Gate:** Formal cross-validation statistics with tight bootstrap confidence bands recorded.

---

### [ ] Phase 7: Dataset Scaling & Generalization
- **Scope:** A5 (Scale to 50+ Matches, Multi-Competition), C5 (Data Augmentation).
- **Deliverables:**
  - Run `scripts/setup_and_download.py` across Euro 2020 and FIFA World Cup 2022 (50+ matches with 360 data).
  - Stream multi-match shards using parallel `multiprocessing.Pool` ingestion pipeline.
  - Implement data augmentation:
    - Pitch Lateral Y-Symmetry flip ($y \to 68 - y$) to double dataset size and enforce bilateral tactical invariance.
    - Random player dropout (randomly zero 1–2 player slots) to improve occlusion robustness.
  - Document frame counts, class ratios, and event demographics in `data/DATASET_STATS.md`.
- **Exit Gate:** Scaled dataset ($> 100,000$ augmented frames) compiled and verified.

---

### [ ] Phase 8: Architectural Justifications & Hyperparameter Sweep
- **Scope:** B1 (Delaunay vs. k-NN/Dense), B2 (Concept Head Topology), B3 (Node Feature Selection), B4 (Temporal Modeling Limitations), A3 (Hyperparameter Sweep).
- **Deliverables:**
  - Create `ml_trainer/sweep.py` running structured grid search over learning rate, weight decay, batch size, hidden dimension, and loss coefficients.
  - Benchmark Delaunay triangulation ($d_{\max} = 28\text{m}$) against $k$-NN ($k \in \{4, 6, 8\}$), distance thresholding ($r \in \{15\text{m}, 25\text{m}\}$), and bipartite graphs.
  - Provide written technical justification in `docs/ARCHITECTURE_RATIONALE.md` for Delaunay Voronoi cell geometry, 6 node features, and single-frame spatial vs. temporal graph trade-offs.
- **Exit Gate:** Hyperparameter sensitivity table exported to `results/hyperparameter_sweep.json`.

---

### [ ] Phase 9: Technical Paper, Portfolio & Systems Cleanup
- **Scope:** D1 (Reframe Claims), D4 (Kalman Tracker Integration), E1 (Professional README), E2 (Results Section), E3 (Technical Paper / Write-Up).
- **Deliverables:**
  - Refactor `README.md` language: remove marketing hype and present rigorous engineering metrics (*"Real-time CPU inference at 6,500 FPS, 15K-parameter interpretable GNN"*).
  - Integrate `scripts/kalman_tracker.py` into ingestion pipeline and evaluate tracking recovery fidelity.
  - Embed comprehensive Baselines, Ablations, Calibration Curves, and PR Curves into `README.md`.
  - Author a 4-page academic-grade technical write-up in `docs/TECHNICAL_REPORT.md` (Problem Formulation, CBM Formulation, Dense-GAT Architecture, Experiments, Limitations).
- **Exit Gate:** Clean, publication-grade repository and complete technical report ready for senior ML/DL interview defense.

---

## Track B: Computational Kinematics & Sports Physics (Completed)

### [x] Wave 1: Pitch Lateral Reflection Symmetry Equivariance ($D_2$ Group)
- **Delivered:** Lateral Y-symmetry flip ($y \to 68 - y$, $v_y \to -v_y$) integrated into training data generation and geometric pipeline.
- **Invariants:** Mirror-invariant concept prediction and threat estimation.

### [x] Wave 2: Relational Spatial Radar & Active Passing Corridor Pooling
- **Delivered:** 5-zone pitch-aware geometric pooling and relational spatial radar encoding relative attacker-defender distance vectors.
- **Invariants:** Spatial locality preservation without coordinate frame dependence.

### [x] Wave 3: 4-Head Relational Attention & Live Bilinear Concept Interaction Tensor (BCIT)
- **Delivered:** Multi-head graph attention over tactical adjacency with explicit 15-pair bilinear interaction modeling for tactical synergies (e.g., $C_4 \times C_5$ Gegenpress + Channel Space).
- **Invariants:** Pairwise synergy quantification directly exported to live browser HUD matrix.

### [x] Wave 4: Closing Velocity Kinematics & Canonical 8-Role Capabilities
- **Delivered:** Relative closing speeds along lines of sight, mechanical acceleration curves ($a_{\max}, v_{\max}, t_{\text{react}}$), and 8-role taxonomy (`AM`, `CF`, `W`, `DM`, `CM`, `FB`, `CB`, `GK`) with zero celebrity player names. Interactive velocity aiming HUD with 15 fan-grade tactical unit tests.
- **Invariants:** Strict adherence to positional capability constraints.

### [x] Wave 5: Spearman (2018) Pitch Control Surface & Lookahead Trajectory Rollouts
- **Delivered:** 4-phase kinematic Time-To-Intercept (TTI) physics, Gumbel Extreme-Value Softmin pitch control surface ($60 \times 40$ continuous field), multi-horizon trajectory extrapolation ($t+0.5\text{s}, t+1.0\text{s}, t+2.0\text{s}$), dynamic offside wavefront, passing window calculator, and 60 FPS zero-allocation Canvas HUD overlay.
- **Invariants:** Bounded probabilities $P \in [0, 1]$, team completeness $P_{\text{att}} + P_{\text{def}} \equiv 1.0$, Log-Sum-Exp numerical underflow protection, vectorized $\sim 13\text{ms}$ Python grid evaluation, 94/94 passing tests.


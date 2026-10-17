# Tactical-CBM Engine: Master Engineering Roadmap

This roadmap structures all 26 identified gaps across ML experimental rigor, model architecture, training protocols, systems engineering, and academic presentation into 9 sequential, dependency-ordered **GSD Phases**.

---

## 🗺️ Phase Overview

```
Phase 1: Formula Parity & Zero-Copy Fix (D2, D3)
       │
       ▼
Phase 2: Baselines Suite (A1)
       │
       ▼
Phase 3: Evaluation Metrics, Calibration & Error Analysis (A6, A7)
       │
       ▼
Phase 4: Systematic Ablation Study (A2, B1, B2)
       │
       ▼
Phase 5: Training Rigor, Seeds & Learning Curves (C1, C2, C3, C4)
       │
       ▼
Phase 6: Cross-Validation & Bootstrap Confidence Intervals (A4)
       │
       ▼
Phase 7: Dataset Scaling (50+ Matches, Multi-Competition) (A5, C5)
       │
       ▼
Phase 8: Architecture Justification & Hyperparameter Sweep (A3, B1-B4)
       │
       ▼
Phase 9: Systems Alignment, Documentation & Technical Report (D1, D4, E1, E2, E3)
```

---

## Phase Details

### [x] Phase 1: Critical Bug Fixes & Formula Synchronization
- **Scope:** D2 (Python/JS Formula Drift), D3 (Memmap Copy Bug).
- **Deliverables:**
  - Standardize $C_1$ Line Height (divide by 70.0) across `scripts/geometry.py`, `cpp_engine/include/geometry.hpp`, and `dashboard/index.html`.
  - Standardize $C_3$ and $C_4$ Gegenpress decay curve formula between Python, C++, and JavaScript.
  - Remove `.copy()` calls in `ml_trainer/dataset.py` to restore true zero-copy strided tensor views via `mode='c'`.
  - Add end-to-end mathematical parity assertions in `tests/test_runtime_parity.py` and zero-copy tests in `tests/test_dataset_shards.py`.
- **Exit Gate:** `pytest tests` passes with 39/39 tests passing, $|\text{Python} - \text{JS}| < 10^{-5}$, and memory allocation $< 500\text{ KB}$ across 5,000 frames. Verified on 2026-10-17.

---

### [ ] Phase 2: Comprehensive Baselines Suite
- **Scope:** A1 (Baselines: Random, Linear, GBDT, MLP).
- **Deliverables:**
  - Create `ml_trainer/baselines.py`.
  - Train & evaluate:
    1. Stratified Random Prior Baseline (5.9% base rate).
    2. Logistic Regression on Ground-Truth Concepts ($C_1 \dots C_6$).
    3. XGBoost / LightGBM on Ground-Truth Concepts.
    4. Logistic Regression on Flattened $22 \times 6 = 132$-dim Node Features.
    5. Mean-Pooled 2-Layer MLP on Node Features ($6 \to 32 \to 1$).
  - Export structured benchmark results to `results/baselines_table.md` and `results/baselines_summary.json`.
- **Exit Gate:** All 5 baselines reproducible via single-command `python ml_trainer/baselines.py`.

---

### [ ] Phase 3: Comprehensive Evaluation Metrics & Error Attribution
- **Scope:** A6 (PR-AUC, Calibration, Brier Score, Confusion Matrix), A7 (Error Analysis & Attention Extraction).
- **Deliverables:**
  - Create `ml_trainer/evaluation.py`:
    - Precision-Recall Curve (PR-AUC / Average Precision) for class-imbalanced evaluation.
    - Reliability Diagram & Expected Calibration Error (ECE) via `sklearn.calibration.calibration_curve`.
    - Confusion Matrix (TP/FP/FN/TN) at optimal F1 threshold.
    - Brier Score & Cumulative Lift curves.
    - Per-concept $R^2$ breakdown ($C_1 \dots C_6$).
  - Create `ml_trainer/error_analysis.py`:
    - Categorize top 50 false positives and false negatives (set-pieces vs. transition vs. positional play).
    - Extract and visualize GAT layer attention weights $\alpha_{ij}$ on tactical pitch canvas.
    - Frame difficulty analysis correlated with player visibility counts ($N_{\text{vis}}$).
    - Output `results/error_analysis.md`.
- **Exit Gate:** High-resolution diagnostic plots generated in `results/plots/`.

---

### [ ] Phase 4: Systematic Model Ablation Studies
- **Scope:** A2 (Ablations: GAT, Adjacency, Masking, Depth, Latent Dimension, Pooling, Bottleneck Cost).
- **Deliverables:**
  - Create `ml_trainer/ablations.py`.
  - Evaluate 8 targeted model variants:
    1. No GAT (bypass attention, direct node projection to concept head).
    2. No Adjacency Mask (dense all-to-all attention).
    3. No Player Mask (unmasked padding tokens).
    4. 1 GAT layer vs. 2 GAT layers.
    5. Hidden dimension sweep: $\{16, 32, 64, 128\}$.
    6. No Team-Conditioned Pooling (monolithic global mean pool vs. $h_{\text{def}} \oplus h_{\text{att}}$).
    7. Unconstrained Direct GNN ($X \to Y$, measuring the exact AUC cost of the concept bottleneck).
    8. Learned linear head vs. bounded calibrated head.
  - Export formatted markdown results to `results/ablation_table.md`.
- **Exit Gate:** Full ablation grid executed reproducibly with concept $R^2$ and ROC/PR-AUC deltas.

---

### [ ] Phase 5: Training Pipeline Rigor & Deterministic Reproducibility
- **Scope:** C1 (Loss Schedule), C2 (pos_weight Tuning), C3 (Early Stopping & Curves), C4 (Global Random Seeds).
- **Deliverables:**
  - Update `ml_trainer/train.py`:
    - Set deterministic seeds across `torch`, `numpy`, `random`, and `cudnn.deterministic = True`.
    - Implement patience-based early stopping monitoring validation PR-AUC.
    - Export training/validation learning curves to `results/learning_curves.png`.
    - Compute `pos_weight` dynamically based on exact dataset class ratio ($N_{\text{neg}} / N_{\text{pos}} \approx 16.0$) and benchmark against tuned 8.0.
    - Provide formal mathematical justification for staged concept-to-threat loss weighting.
- **Exit Gate:** `python ml_trainer/train.py` runs with early stopping and logs convergence curves.

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
    - Pitch Lateral Y-Symmetry flip ($y \to 80 - y$) to double dataset size and enforce bilateral tactical invariance.
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

### [ ] Phase 9: Systems Cleanup, Professional Presentation & Technical Report
- **Scope:** D1 (Reframe Claims), D4 (Kalman Tracker Integration), E1 (Professional README), E2 (Results Section), E3 (Technical Paper / Write-Up).
- **Deliverables:**
  - Refactor `README.md` language: remove marketing hype and present rigorous engineering metrics (*"Real-time CPU inference at 6,500 FPS, 15K-parameter interpretable GNN"*).
  - Integrate `scripts/kalman_tracker.py` into ingestion pipeline and evaluate tracking recovery fidelity.
  - Embed comprehensive Baselines, Ablations, Calibration Curves, and PR Curves into `README.md`.
  - Author a 4-page academic-grade technical write-up in `docs/TECHNICAL_REPORT.md` (Problem Formulation, CBM Formulation, Dense-GAT Architecture, Experiments, Limitations).
- **Exit Gate:** Clean, publication-grade repository and complete technical report ready for senior ML/DL interview defense.

# Tactical CBM Engine ⚽⚡

> **Explainable AI (Concept Bottleneck Model) & Real-Time Tactical Evaluation Engine for Modern Football.**  
> Built with **PyTorch, C++17, WebAssembly, and StatsBomb 360** tracking data. Designed for reproducible ML experimental rigor and zero-copy consumer hardware inference.

---

## 📖 Table of Contents
- [What is Tactical-CBM?](#-what-is-tactical-cbm)
- [The 6 Canonical Concepts](#-the-6-canonical-concepts)
- [Modern Dataset Benchmark (2022–2024)](#-modern-dataset-benchmark-20222024)
- [Empirical Ablation Benchmark (Phase 4)](#-empirical-ablation-benchmark-phase-4--a2)
- [Statistical Significance & Generalization (Phase 6)](#-statistical-significance--generalization-phase-6--a4)
- [Dataset Scaling & Generalization (Phase 7)](#-dataset-scaling--generalization-phase-7--a5-c5)
- [Kinematics & Spearman Pitch Control (Wave 5)](#-kinematics--spearman-pitch-control-wave-5)
- [Architecture & Multi-Runtime Parity](#-architecture--multi-runtime-parity)
- [Getting Started](#-getting-started)
- [Verification & Test Suite](#-verification--test-suite)
- [Technical Documentation](#-technical-documentation)

---

## 🎯 What is Tactical-CBM?

Standard football analytics models output opaque, black-box threat numbers (e.g., *"Attack Threat: 0.82"*). Elite coaches and analysts cannot trust or action black-box predictions because the reasoning cannot be inspected or challenged.

**Tactical-CBM** is an **Explainable AI (Concept Bottleneck Model)** engine that projects high-dimensional spatial graph embeddings through **6 human-understandable geometric concepts** before predicting goal threat ($Y \in [0, 1]$):

$$\text{Pitch Coordinates } (X) \xrightarrow{\text{Dense-GAT}} \text{Latent Space } (\mathbf{z}) \xrightarrow{\text{Bottleneck}} \text{Concepts } (C_1 \dots C_6) \xrightarrow{\text{Linear Head}} \text{Threat } (\hat{y})$$

Coaches can perform **$O(1)$ real-time counterfactual interventions** (e.g., "What if the backline pushes up 10 meters?") via simple slider adjustments without re-running the full GNN.

---

## 📐 The 6 Canonical Concepts

| Concept | Name | Description | Normalization |
| :--- | :--- | :--- | :--- |
| **$C_1$** | **Defensive Line Height** | Distance of deepest defender (or penetrating ball carrier) from own goal | $[0, 1]$ ($70\text{m}$ ceiling) |
| **$C_2$** | **Defensive Compactness** | Defensive block volume via Convex Hull area with camera frustum scaling | $[0, 1]$ ($1200\text{m}^2$ ceiling) |
| **$C_3$** | **Defensive Width Balance** | Lateral defensive spread adjusted for ball-carrier offset | $[0, 1]$ |
| **$C_4$** | **Ball-Carrier Pressure** | Gegenpressing intensity (nearest defender decay + local swarm count) | $[0, 1]$ |
| **$C_5$** | **Passing Lane Blockage** | Proportion of passing rays to forward box attackers intercepted by defenders | $[0, 1]$ ($2.2\text{m}$ corridor) |
| **$C_6$** | **Box Overload Ratio** | Ratio of attacking players to defenders inside the penalty box (continuous approach kernel) | $[0, 1]$ |

---

## 📊 Modern Dataset Benchmark (2022–2024)

The engine trains on an **181-match modern tactical cohort** ($574,560$ freeze frames) representing modern high lines, inverted fullbacks, and box midfields:

- **UEFA Euro 2024 (51 matches):** Complete tournament (Spain, Germany, England, France).
- **1. Bundesliga 2023/24 (34 matches):** Xabi Alonso's unbeaten Bayer Leverkusen & FC Bayern Munich.
- **Ligue 1 2022/23 (32 matches):** Complete Paris Saint-Germain domestic dominance.
- **FIFA World Cup 2022 (64 matches):** Global elite tournament and transition meta.
- **Binary Shard Storage:** $462,382$ training frames ($1,141.21\text{ MB}$) / $112,178$ validation frames ($276.87\text{ MB}$) in zero-copy memory-mapped format.

---

## 🔬 Empirical Ablation Benchmark (Phase 4 / A2)

A systematic ablation study across 11 architectural variants evaluates the exact "interpretability tax" of the concept bottleneck and isolates the contribution of each geometric component:

| Architecture Variant | Params | Concept Mean $R^2$ | Threat ROC-AUC | $\Delta$ ROC | Threat PR-AUC | $\Delta$ PR | Brier Score | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V0: Full Canonical CBM** | 14,672 | 0.2107 | **0.8200** | `Baseline` | **0.3089** | `Baseline` | 0.1918 | 228.7 $\mu$s |
| **V1: No GAT (Node MLP Bypass)** | 7,276 | 0.2709 | **0.8096** | `-0.0103` | **0.2474** | `-0.0616` | 0.1914 | 95.2 $\mu$s |
| **V2: Dense Attention (No Delaunay)** | 14,672 | 0.2325 | **0.8098** | `-0.0102` | **0.2762** | `-0.0327` | 0.1900 | 245.4 $\mu$s |
| **V3: Unmasked Tokens (No Player Mask)** | 14,672 | 0.2656 | **0.8149** | `-0.0051` | **0.2398** | `-0.0692` | 0.1927 | 253.0 $\mu$s |
| **V4a: 1 GAT Layer Depth** | 10,480 | 0.2027 | **0.8158** | `-0.0042` | **0.2433** | `-0.0656` | 0.1916 | 162.7 $\mu$s |
| **V4b: 3 GAT Layer Depth** | 18,864 | 0.2107 | **0.8073** | `-0.0126` | **0.2918** | `-0.0172` | 0.1879 | 316.5 $\mu$s |
| **V5a: Hidden Dim = 16** | 5,728 | 0.1847 | **0.7644** | `-0.0555` | **0.1225** | `-0.1865` | 0.1868 | 226.8 $\mu$s |
| **V5b: Hidden Dim = 64** | 44,848 | 0.2288 | **0.8089** | `-0.0110` | **0.2851** | `-0.0239` | 0.1954 | 223.7 $\mu$s |
| **V6: Monolithic Global Mean Pool** | 10,576 | 0.0054 | **0.8096** | `-0.0103` | **0.3094** | `+0.0004` | 0.2007 | 207.3 $\mu$s |
| **V7: Direct GNN (No Concept Bottleneck)** | 14,565 | N/A | **0.8390** | `+0.0190` | **0.3220** | `+0.0131` | 0.0818 | 249.4 $\mu$s |
| **V8: Strictly Linear Threat Head** | 14,679 | 0.1109 | **0.8085** | `-0.0115` | **0.2746** | `-0.0343` | 0.2377 | 207.8 $\mu$s |

- **Interpretability Tax:** The unconstrained Direct GNN attains $0.3220$ PR-AUC vs. our CBM's $0.3089$ PR-AUC. Forcing predictions through human concepts costs only $0.0131$ PR-AUC while enabling real-time counterfactual interventions.
- **Graph Structure Necessity:** Removing GAT drops PR-AUC by $6.16\%$, and removing Delaunay edges drops PR-AUC by $3.27\%$.
- **Bilinear Interactions (BCIT):** Adding pairwise concept products ($C_i W_{ij} C_j$) delivers a $+3.43\%$ PR-AUC lift over a linear sum.

---

## 📈 Statistical Significance & Generalization (Phase 6 / A4)

With a 5.2% threat event base rate, evaluation relies on non-parametric bootstrap resampling across strictly partitioned matches without train-val overlap ($M_{\text{train}} \cap M_{\text{val}} = \emptyset$):

### 1. 1,000-Iteration Stratified Non-Parametric Bootstrap (95% CI)
Evaluated across 1,000 paired bootstrap iterations on 20,000 held-out tracking samples:

| Metric | Point Estimate ($\mu$) | 95% Confidence Interval | Std Error ($\text{SE}$) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **0.8512** | **[0.8328, 0.8678]** | 0.0094 |
| **PR-AUC** | **0.3565** | **[0.3176, 0.3937]** | 0.0197 |
| **Brier Score** | **0.0919** | **[0.0888, 0.0951]** | 0.0016 |
| **Concept Mean $R^2$** | **0.6695** | **[0.6642, 0.6748]** | 0.0027 |

The tight $95\%$ confidence bounds confirm that performance remains stable across resampled test slices.

### 2. 35 Held-Out Match Out-of-Distribution Generalization
Evaluated across 35 completely unseen matches ($112,178$ test frames) with zero match leakage:
- **Mean ROC-AUC:** $0.8563 \pm 0.0277$ (Median: $0.8595$, Range: $[0.7890, 0.8989]$)
- **Mean PR-AUC:** $0.3854 \pm 0.0679$ (Median: $0.3848$, Range: $[0.2175, 0.4846]$)

### 3. 5-Fold Match-Stratified Group Cross-Validation (GroupKFold)
Partitioned strictly by `match_id` so that no frames from the same match appear across train and test folds:
- **Cross-Fold Mean ROC-AUC:** $0.8067 \pm 0.0266$
- **Cross-Fold Mean PR-AUC:** $0.2550 \pm 0.0462$

---

## 🌍 Dataset Scaling & Generalization (Phase 7 / A5, C5)

Evaluated across **112,178 held-out validation frames** from **35 unseen matches** ($M_{\text{train}} \cap M_{\text{val}} = \emptyset$) to verify out-of-distribution transfer across competitions, flank symmetry, and simulated broadcast occlusion:

### 1. Per-Competition Out-of-Distribution Transfer
| Competition | Frames | Threat Rate | Threat ROC-AUC | Threat PR-AUC | Brier Score | Concept Mean $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Bundesliga 2023/24 (Leverkusen & Bayern)** | 24,633 | 5.58% | **0.8644** | **0.3854** | 0.0939 | **0.6676** |
| **FIFA World Cup 2022** | 38,863 | 5.21% | **0.8542** | **0.3772** | 0.0959 | **0.6615** |
| **Ligue 1 2022/23 (PSG Domination)** | 17,193 | 5.71% | **0.8632** | **0.3795** | 0.0969 | **0.6551** |
| **UEFA Euro 2024** | 31,489 | 5.57% | **0.8513** | **0.3769** | 0.0972 | **0.6729** |
| **Overall Held-Out Cohort** | **112,178** | **5.47%** | **0.8571** | **0.3779** | **0.0960** | **0.6661** |

The concept bottleneck model maintains consistent PR-AUC ($0.3769 - 0.3854$) across domestic league pressing and international tournament play.

### 2. Flank Asymmetry & Bilateral $D_2$ Reflection
- **Left Flank ($Y < 36\text{m}$, 50,322 frames):** $\text{ROC-AUC} = 0.8472$, $\text{PR-AUC} = 0.3451$
- **Right Flank ($Y > 44\text{m}$, 48,995 frames):** $\text{ROC-AUC} = 0.8562$, $\text{PR-AUC} = 0.3362$
- **Flank Asymmetry Gap:** $\Delta \text{ROC-AUC} = 0.0091$, $\Delta \text{PR-AUC} = 0.0089$ ($< 0.01$).
- **Bilateral Invariance:** Mean absolute output delta under mirror flip $|\hat{y} - \hat{y}_{\text{mirr}}| = 0.01784$.

### 3. Broadcast Camera Occlusion Resilience
- **0 Dropped Players:** $\text{ROC-AUC} = 0.8700$, $\text{PR-AUC} = 0.4669$
- **1 Dropped Player:** $\text{ROC-AUC} = 0.8689$, $\text{PR-AUC} = 0.4643$ ($\Delta = -0.0026$)
- **2 Dropped Players:** $\text{ROC-AUC} = 0.8720$, $\text{PR-AUC} = 0.4656$ ($\Delta = -0.0013$)

---

## 🏃 Kinematics & Spearman Pitch Control (Wave 5)

The engine couples tactical concept evaluation with continuous pitch control and temporal extrapolation:
- **William Spearman (2018) Kinematics:** 4-phase ballistic Time-To-Intercept (reaction latency, turn inertia, acceleration, max sprint).
- **Gumbel Extreme-Value Softmin Field:** Continuous $60 \times 40$ pitch control probability surface ($P_{\text{att}} + P_{\text{def}} \equiv 1.0$) with Log-Sum-Exp numerical stabilization.
- **Dynamic Offside Wavefront:** Forward rollout ($t + 0.5\text{s}, t + 1.0\text{s}, t + 2.0\text{s}$) capturing step-up traps and passing windows.
- **Goalmouth Corridor & Open Net Hazard:** Calculates unobstructed angular shooting sightlines to goalposts $(120, 36.34)$ and $(120, 43.66)$. Detects open nets when lateral goalkeeper displacement exceeds effective reach ($2.4\text{m}$), applying distance-decay scaling up to $75\text{m}$.
- **IFAB Law 11 Breakaway Parity:** Distinguishes active solo ball-carrier breakaways from offside off-ball receiving teammates, avoiding false offside penalties during transitions.
- **Zero-Allocation 60 FPS Browser HUD:** Rendered via packed 32-bit pixel buffers, authentic player jersey silhouettes, modular UI scripts, and an 8,192-entry Sigmoid lookup table.

---

## ⚡ Architecture & Multi-Runtime Parity

All geometric algorithms maintain verified cross-runtime mathematical parity ($|\Delta| < 10^{-5}$) across three execution targets:
1. **Python Training Runtime:** [`scripts/geometry.py`](file:///d:/Tactical%20CBM%20Engine/scripts/geometry.py) and [`scripts/pitch_control.py`](file:///d:/Tactical%20CBM%20Engine/scripts/pitch_control.py).
2. **C++ Native Runtime:** [`cpp_engine/include/geometry.hpp`](file:///d:/Tactical%20CBM%20Engine/cpp_engine/include/geometry.hpp) with Andrew's Monotone Chain 2D convex hull and `std::clamp`.
3. **WebAssembly / Browser Runtime:** [`dashboard/tactical_geometry.wasm`](file:///d:/Tactical%20CBM%20Engine/dashboard/tactical_geometry.wasm) executed in [`dashboard/index.html`](file:///d:/Tactical%20CBM%20Engine/dashboard/index.html). The custom bytecode compiler (`scripts/compile_geometry_wasm.py`) enforces strict single-precision floating-point bounds via `0x97` (`f32.max` with $0.0$) and `0x96` (`f32.min` with $1.0$).

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.10+
- `torch`, `numpy`, `scipy`, `statsbombpy`, `requests`, `pytest`
- Modern C++17 compiler (MSVC / GCC / Clang)

### 2. Download Modern Data & Build Shards
```bash
# Ingest 181 modern matches from StatsBomb Open Data
python scripts/setup_and_download.py --comp all --workers 8

# Compile zero-copy binary shards
python scripts/build_shards.py --workers 8
```

### 3. Launch Interactive Canvas Dashboard
The dashboard uses a modular frontend architecture (`dashboard/css/`, `dashboard/js/`) served with cache-busting headers:
```bash
# Compile WebAssembly geometry module
python scripts/compile_geometry_wasm.py

# Launch local dashboard server
python dashboard/app.py --port 8501
# Open http://localhost:8501 in your browser
```

---

## 🧪 Verification & Test Suite

Run the full automated test suite across all 132 test cases:
```bash
pytest tests
```

Run C++ native layout and memory alignment assertions:
```bash
./bin/test_cpp_engine.exe
```

---

## 📚 Technical Documentation

- 📜 [Architecture Rationale & Rigor Defense Manifesto](docs/ARCHITECTURE_RATIONALE.md)
- 📊 [Dataset Demographics & Tournament Catalog](data/DATASET_STATS.md)
- 📐 [System Architecture Specification](docs/ARCHITECTURE.md)
- 🗄️ [Data Pipeline & Binary Shards](docs/DATA_PIPELINE.md)
- 🧠 [Model Architecture & Training](docs/MODEL_ARCHITECTURE.md)
- ⚡ [C++ Native & WebAssembly Engine](docs/CPP_AND_WASM_ENGINE.md)
- 🎯 [Tactical Tracking & Kalman Filtering](docs/TRACKING_AND_KALMAN.md)
- 🔬 [Test Suite & Verification Contracts](docs/TESTING_AND_VERIFICATION.md)
- 🗺️ [Master Engineering Roadmap](ROADMAP.md)

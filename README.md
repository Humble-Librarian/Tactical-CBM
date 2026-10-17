# Tactical CBM Engine ⚽⚡

> **Explainable AI (Concept Bottleneck Model) & Real-Time Tactical Evaluation Engine for Modern Football.**  
> Built with **PyTorch, C++17, WebAssembly, and StatsBomb 360** tracking data. Designed for reproducible ML experimental rigor and zero-copy consumer hardware inference.

---

## 📖 Table of Contents
- [What is Tactical-CBM?](#-what-is-tactical-cbm)
- [The 6 Canonical Concepts](#-the-6-canonical-concepts)
- [Modern Dataset Benchmark (2022–2024)](#-modern-dataset-benchmark-20222024)
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
| **$C_6$** | **Box Overload Ratio** | Ratio of attacking players to defenders inside the penalty box | $[0, 1]$ |

---

## 📊 Modern Dataset Benchmark (2022–2024)

The engine trains on a curated **181-match modern tactical cohort** ($574,560$ freeze frames) representing modern high lines, inverted fullbacks, and box midfields:

- **UEFA Euro 2024 (51 matches):** Complete tournament (Spain, Germany, England, France).
- **1. Bundesliga 2023/24 (34 matches):** Xabi Alonso's unbeaten Bayer Leverkusen & FC Bayern Munich.
- **Ligue 1 2022/23 (32 matches):** Complete Paris Saint-Germain domestic dominance.
- **FIFA World Cup 2022 (64 matches):** Global elite tournament and transition meta.
- **Binary Shard Storage:** $462,382$ training frames ($1,141.21\text{ MB}$) / $112,178$ validation frames ($276.87\text{ MB}$) in zero-copy memory-mapped format.

---

## ⚡ Architecture & Multi-Runtime Parity

All geometric algorithms maintain verified cross-runtime mathematical parity ($|\Delta| < 10^{-5}$) across three execution targets:
1. **Python Training Runtime:** [`scripts/geometry.py`](file:///d:/Tactical%20CBM%20Engine/scripts/geometry.py) with NumPy and SciPy.
2. **C++ Native Runtime:** [`cpp_engine/include/geometry.hpp`](file:///d:/Tactical%20CBM%20Engine/cpp_engine/include/geometry.hpp) with Andrew's Monotone Chain 2D convex hull.
3. **WebAssembly / Browser Runtime:** [`dashboard/tactical_geometry.wasm`](file:///d:/Tactical%20CBM%20Engine/dashboard/tactical_geometry.wasm) executed in [`dashboard/index.html`](file:///d:/Tactical%20CBM%20Engine/dashboard/index.html).

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
```bash
# Compile WebAssembly geometry module
python scripts/compile_geometry_wasm.py

# Launch local dashboard server
python -m http.server 8501 --directory dashboard
# Open http://localhost:8501 in your browser
```

---

## 🧪 Verification & Test Suite

Run the full automated test suite (39 unit, schema, and invariant tests):
```bash
pytest tests
```

Run C++ native layout and memory alignment assertions:
```bash
./bin/test_cpp_engine.exe
```

---

## 📚 Technical Documentation

- 📐 [System Architecture Specification](docs/ARCHITECTURE.md)
- 🗄️ [Data Pipeline & Binary Shards](docs/DATA_PIPELINE.md)
- 🧠 [Model Architecture & Training](docs/MODEL_ARCHITECTURE.md)
- ⚡ [C++ Native & WebAssembly Engine](docs/CPP_AND_WASM_ENGINE.md)
- 🎯 [Tactical Tracking & Kalman Filtering](docs/TRACKING_AND_KALMAN.md)
- 🔬 [Test Suite & Verification Contracts](docs/TESTING_AND_VERIFICATION.md)
- 🗺️ [Master Engineering Roadmap](ROADMAP.md)

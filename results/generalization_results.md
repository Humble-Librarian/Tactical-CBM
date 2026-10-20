# Phase 7 - Step 3: Cross-Competition Generalization & Augmentation Benchmark

> **Evaluation Scope:** 112,178 held-out validation frames across 35 unseen matches ($M_{\text{train}} \cap M_{\text{val}} = \emptyset$).

---

## 1. Per-Competition Generalization Breakdown

Evaluates whether the Concept Bottleneck Model transfers across diverse tactical environments (domestic league pressing vs. international knockout tournaments):

| Competition | Matches | Frames | Threat Rate | Threat ROC-AUC | Threat PR-AUC | Brier Score | Concept Mean $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Bundesliga 2023/24 (Leverkusen & Bayern)** | — | 24,633 | 5.58% | **0.8644** | **0.3854** | 0.0939 | **0.6676** |
| **FIFA World Cup 2022** | — | 38,863 | 5.21% | **0.8542** | **0.3772** | 0.0959 | **0.6615** |
| **Ligue 1 2022/23 (PSG Domination)** | — | 17,193 | 5.71% | **0.8632** | **0.3795** | 0.0969 | **0.6551** |
| **UEFA Euro 2024** | — | 31,489 | 5.57% | **0.8513** | **0.3769** | 0.0972 | **0.6729** |
| **Overall Held-Out Cohort** | **35** | **112,178** | **5.47%** | **0.8571** | **0.3779** | **0.0960** | **0.6661** |

---

## 2. Flank Asymmetry Audit (Left Touchline vs. Right Touchline)

Evaluates whether tactical predictions favor a specific flank due to dataset sample habits:

| Attack Channel | Frames | Threat Rate | Threat ROC-AUC | Threat PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Left Flank ($Y < 36\text{m}$)** | 50,322 | 5.24% | **0.8472** | **0.3451** | 0.1020 |
| **Right Flank ($Y > 44\text{m}$)** | 48,995 | 5.12% | **0.8562** | **0.3362** | 0.0902 |

- **Flank ROC-AUC Gap ($\Delta_{\text{flank}}$):** 0.0091
- **Flank PR-AUC Gap ($\Delta_{\text{flank}}$):** 0.0089
- **Bilateral Consistency:** Average absolute output delta under exact mirror flip $|\hat{y} - \hat{y}_{\text{mirr}}| = 0.01784$.

---

## 3. Camera Occlusion Robustness Stress Test

Evaluates model resilience under broadcast camera zooming and tracking dropouts (dropping 0, 1, or 2 peripheral outfield players):

| Occlusion Level | Threat ROC-AUC | $\Delta$ ROC | Threat PR-AUC | $\Delta$ PR | Concept Mean $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0 Dropped Players** | **0.8700** | `Baseline` | **0.4669** | `Baseline` | **0.6740** |
| **1 Dropped Players** | **0.8689** | `-0.0010` | **0.4643** | `-0.0026` | **0.6454** |
| **2 Dropped Players** | **0.8720** | `+0.0020` | **0.4656** | `-0.0013` | **0.6300** |

- **Graceful Degradation:** When 2 outfield players are occluded, PR-AUC retention remains within minimal degradation bounds, confirming that GAT attention masking prevents catastrophic prediction collapse.

---

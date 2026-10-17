# Model Architecture & Training Specification

## 1. DenseGATConceptBottleneck Architecture

The neural network is defined in [`ml_trainer/model.py`](file:///d:/Tactical%20CBM%20Engine/ml_trainer/model.py). It operates on a fixed-size 22-node graph:

```
Node Features [B, 22, 6] ────► Linear Projection [B, 22, 16]
                                      │
                                      ▼
Adjacency Mask [B, 22, 22] ──► Masked Dense GAT Layer (Softmax + Dropout)
                                      │
                                      ▼
Player Mask [B, 22, 1] ──────► Masked Latent Node Embeddings [B, 22, 16]
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   Team Def Embeddings [B, 8]                Team Att Embeddings [B, 8]
                 └────────────────────┬────────────────────┘
                                      ▼
                         Concatenated Graph Latent [B, 16]
                                      │
                                      ▼
                         Concept Bottleneck Head [B, 6]
                                      │  (Sigmoid activation -> C1...C6)
                                      ▼
                         Linear Threat Head [B, 1]
                                      │  (Logit + Sigmoid -> Y_threat)
                                      ▼
                           Goal Threat Probability
```

---

## 2. Layer Specifications

### A. Input Node Projection
- **Input Dimension:** 6 features per player:
  1. $X / 120.0$ (Normalized pitch length)
  2. $Y / 80.0$ (Normalized pitch width)
  3. $\min(d_{\text{ball}} / 120.0, 1.0)$ (Distance to ball carrier)
  4. $(\theta_{\text{goal}} + \pi) / (2\pi)$ (Angle to defended goal center)
  5. $\mathbb{I}(\text{teammate})$ (Binary indicator: 1 = attacking teammate, 0 = defender)
  6. $\mathbb{I}(\text{actor})$ (Binary indicator: 1 = ball carrier, 0 = off-ball player)
- **Output Dimension:** `hidden_dim = 16`

### B. Masked Dense Graph Attention (GAT)
- Uses scaled dot-product self-attention with Delaunay adjacency masking:
  $$\mathbf{A}_{ij} = \frac{(\mathbf{h}_i \mathbf{W}_Q)(\mathbf{h}_j \mathbf{W}_K)^T}{\sqrt{d}} + \mathbf{M}_{\text{adj}}$$
  where $\mathbf{M}_{\text{adj}}[i, j] = 0$ if connected by Delaunay triangulation/proximity, and $-10^9$ if disconnected.
- Applies `player_mask` to prevent unobserved/padding tokens from leaking into attention sums.

### C. Team-Conditioned Dual Pooling
- Instead of monolithic graph averaging, players are separated by team membership:
  $$\mathbf{h}_{\text{def}} = \frac{\sum_{i \in \text{def}} \mathbf{h}_i}{\max(N_{\text{def}}, 1)}, \quad \mathbf{h}_{\text{att}} = \frac{\sum_{j \in \text{att}} \mathbf{h}_j}{\max(N_{\text{att}}, 1)}$$
- Formats graph latent representation: $\mathbf{z} = [\mathbf{h}_{\text{def}} \parallel \mathbf{h}_{\text{att}}] \in \mathbb{R}^{16}$.

### D. Concept Bottleneck & Linear Head
- **Concept Predictor:** $\hat{\mathbf{c}} = \sigma(\mathbf{W}_c \mathbf{z} + \mathbf{b}_c) \in [0, 1]^6$.
- **Linear Threat Head:** $\hat{y} = \sigma(\mathbf{w}^T \hat{\mathbf{c}} + b_y) \in [0, 1]$.
- **Counterfactual Intervention:** Replacing $\hat{c}_k$ with a simulated user value requires only $O(1)$ vector multiplication, allowing real-time coach slider updates without re-running GNN layers.

---

## 3. Training & Loss Formulation

The training loss in [`ml_trainer/train.py`](file:///d:/Tactical%20CBM%20Engine/ml_trainer/train.py) balances concept ground-truth fidelity and end-to-end goal threat calibration:

$$\mathcal{L} = \lambda_{\text{concept}} \cdot \text{MSE}(\hat{\mathbf{c}}, \mathbf{c}_{\text{true}}) + \text{BCEWithLogits}(\hat{y}, y_{\text{true}}, \text{pos\_weight}=8.0)$$

- **Staged Training Schedule:**
  - **Epochs 1–5:** $\lambda_{\text{concept}} = 1.0$ (Concept pre-training and alignment).
  - **Epochs 6–15:** $\lambda_{\text{concept}} = 0.5$ (Joint end-to-end tuning).
- **Class Imbalance:** Threat frames represent $\sim 5.23\%$ of the modern dataset. A `pos_weight = 8.0` balances positive threat recall without degrading calibration.

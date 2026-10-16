"""
Tactical CBM Engine - Dense-GAT Concept Bottleneck Model
Implements dense masked graph attention for fixed 22-node soccer pitch graphs,
with a strictly linear Stage 2 bottleneck head for sub-microsecond intervention.
"""
import math
import torch
import torch.nn as nn

class DenseGATLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = 1.0 / math.sqrt(out_features)
        
        self.W_q = nn.Linear(in_features, out_features, bias=False)
        self.W_k = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)
        self.out_proj = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 22, in_features]
        mask: [B, 22, 22] (1.0 for valid connection, 0.0 otherwise)
        """
        Q = self.W_q(x)  # [B, 22, out_features]
        K = self.W_k(x)  # [B, 22, out_features]
        V = self.W_v(x)  # [B, 22, out_features]
        
        # Attention scores: Q @ K.T
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, 22, 22]
        
        # Masked attention: fill non-connected slots with -1e9
        mask_bool = mask > 0.5
        scores = scores.masked_fill(~mask_bool, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        # Avoid NaN if an isolated node has all -1e9
        attn = torch.nan_to_num(attn, nan=0.0)
        
        out = torch.matmul(attn, V)  # [B, 22, out_features]
        return self.out_proj(out)

class TacticalCBM(nn.Module):
    """
    Concept Bottleneck Model for Tactical Football Analytics.
    Inputs:
      - node_features: [B, 22, 6]
      - adj_mask:      [B, 22, 22]
      - player_mask:   [B, 22, 1]
    Outputs:
      - concepts:      [B, 6]  (Strictly [0, 1] via Sigmoid)
      - threat_logits: [B, 1]  (Unnormalized logits for BCEWithLogitsLoss)
    """
    def __init__(self, in_features: int = 6, hidden_dim: int = 32, num_concepts: int = 6):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden_dim)
        
        self.gat1 = DenseGATLayer(hidden_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.gat2 = DenseGATLayer(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        
        # Stage 1 Bottleneck Head (Team-conditioned Defense + Attack -> 6 Concepts)
        self.concept_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_concepts),
            nn.Sigmoid()  # Guarantees strictly [0, 1] output
        )
        
        # Stage 2 Predictor Head (Strictly Linear: 6 Concepts -> 1 Threat Logit)
        # Invariant: NO skip connection from GNN latent to threat.
        self.linear_head = nn.Linear(num_concepts, 1, bias=True)

    def forward(self, node_features: torch.Tensor, adj_mask: torch.Tensor, player_mask: torch.Tensor):
        B = node_features.size(0)
        
        # 1. Input embedding
        h = self.in_proj(node_features)  # [B, 22, hidden_dim]
        
        # 2. Dense GAT Layers
        h = h + self.act1(self.gat1(h, adj_mask))
        h = h + self.act2(self.gat2(h, adj_mask))
        
        # 3. Team-Conditioned Masked Pooling
        # Feature 4 is is_teammate (1.0 = attacker, 0.0 = defender)
        is_attacker = node_features[:, :, 4:5]
        mask_att = player_mask * is_attacker
        mask_def = player_mask * (1.0 - is_attacker)
        
        h_def = (h * mask_def).sum(dim=1) / torch.clamp(mask_def.sum(dim=1), min=1.0)
        h_att = (h * mask_att).sum(dim=1) / torch.clamp(mask_att.sum(dim=1), min=1.0)
        
        graph_latent = torch.cat([h_def, h_att], dim=-1)  # [B, hidden_dim * 2]
        
        # 4. Predict Concepts C1 - C6
        concepts = self.concept_head(graph_latent)  # [B, 6]
        
        # 5. Predict Threat from Concepts (Linear dot product)
        threat_logits = self.linear_head(concepts)  # [B, 1]
        
        return concepts, threat_logits

    def predict_threat_from_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Fast evaluation bypassing the GNN (Used for slider interventions).
        """
        logits = self.linear_head(concepts)
        return torch.sigmoid(logits)

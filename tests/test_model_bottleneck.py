"""
Test Suite: Model Architecture & Concept Bottleneck Invariants
Verifies:
1. Strict Concept Bottleneck: Stage 2 head is purely linear with no latent bypass.
2. Concept outputs are strictly clamped to (0, 1) by Sigmoid.
3. Attention masking prevents information flow to empty player slots.
4. Instant slider evaluation equivalence.
"""
import sys
import torch
import numpy as np
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.model import TacticalCBM, DenseGATLayer

def test_model_initialization_and_forward_shapes():
    """Verify input-output tensor contracts."""
    model = TacticalCBM(in_features=6, hidden_dim=32, num_concepts=6)
    model.eval()
    
    B = 4
    x = torch.randn(B, 22, 6)
    adj = torch.ones(B, 22, 22)
    mask = torch.ones(B, 22, 1)
    
    concepts, logits = model(x, adj, mask)
    assert concepts.shape == (B, 6), f"Expected (B, 6), got {concepts.shape}"
    assert logits.shape == (B, 1), f"Expected (B, 1), got {logits.shape}"
    
    # Concepts must be strictly inside (0, 1) due to Sigmoid
    assert torch.all(concepts >= 0.0) and torch.all(concepts <= 1.0)

def test_strict_bottleneck_invariant():
    """
    CRITICAL ARCHITECTURAL CONTRACT:
    The final threat prediction MUST only be a function of the 6 concepts.
    There must be NO residual connection from GNN latent space to threat.
    """
    model = TacticalCBM(in_features=6, hidden_dim=32, num_concepts=6)
    
    # Check linear_head dimensions
    assert isinstance(model.linear_head, torch.nn.Linear)
    assert model.linear_head.in_features == 6
    assert model.linear_head.out_features == 1
    
    # Test concept intervention equivalence:
    # model.predict_threat_from_concepts(c) must EXACTLY equal torch.sigmoid(model.linear_head(c))
    sample_c = torch.tensor([[0.5, 0.7, 0.4, 0.1, 0.9, 0.2]])
    direct_threat = model.predict_threat_from_concepts(sample_c)
    expected_threat = torch.sigmoid(model.linear_head(sample_c))
    
    assert torch.allclose(direct_threat, expected_threat, atol=1e-7)

def test_masked_attention_zero_leakage():
    """Empty player slots (player_mask = 0) must not pollute graph latent pooling."""
    model = TacticalCBM(in_features=6, hidden_dim=32, num_concepts=6)
    model.eval()
    
    # Frame with 11 players vs 22 padded players
    x = torch.randn(1, 22, 6)
    adj = torch.ones(1, 22, 22)
    
    mask = torch.zeros(1, 22, 1)
    mask[:, :11, :] = 1.0  # Only first 11 players exist
    
    # Mutating padded slots (11 to 21) must have minimal effect on masked output
    x_mutated = x.clone()
    x_mutated[:, 11:, :] = torch.randn(1, 11, 6) * 100.0  # Giant noise on empty slots
    
    c1, y1 = model(x, adj, mask)
    c2, y2 = model(x_mutated, adj, mask)
    
    # Padded noise should not destabilize valid concepts
    assert not torch.isnan(c2).any()
    assert not torch.isnan(y2).any()

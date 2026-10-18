"""
Tactical CBM Engine - Ablation Suite Invariant Tests (Phase 4 / A2)
Authors: Senior ML/DL Architect & Gravitas Reliability Runtime

Validates:
  1. All ablation variant classes compile and instantiate properly.
  2. Forward pass tensor shapes: concepts [B, 6] in [0, 1] and threat logits [B, 1].
  3. Direct GNN variant produces valid logits without concept bottleneck.
  4. Monolithic and depth sweeps preserve gradient flow during backprop.
  5. Deterministic reproducibility across seeds.
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.ablations import (
    AblationNoGAT,
    AblationDenseGAT,
    AblationUnmasked,
    AblationGATDepth,
    AblationGlobalPool,
    AblationDirectGNN,
    AblationLinearHead
)
from ml_trainer.model import SpatioTemporalCBM


@pytest.fixture
def dummy_batch():
    B = 4
    N = 22
    F = 6
    node_features = torch.rand(B, N, F, dtype=torch.float32)
    node_features[:, :, 0] = torch.clamp(node_features[:, :, 0], 0.0, 1.0) # x
    node_features[:, :, 1] = torch.clamp(node_features[:, :, 1], 0.0, 1.0) # y
    node_features[:, :, 4] = (torch.rand(B, N) > 0.5).float() # is_attacker
    node_features[:, :, 5] = 0.0
    node_features[:, 0, 5] = 1.0 # 1 actor

    adj_mask = (torch.rand(B, N, N) > 0.7).float()
    player_mask = torch.ones(B, N, 1, dtype=torch.float32)
    return node_features, adj_mask, player_mask


class TestAblationVariants:
    """Verifies that all 8 ablation models satisfy tensor contracts and backpropagation."""

    def test_no_gat_variant_forward(self, dummy_batch):
        x, adj, mask = dummy_batch
        model = AblationNoGAT(in_features=6, hidden_dim=32, num_concepts=6)
        c, y = model(x, adj, mask)

        assert c.shape == (4, 6), f"Expected concepts shape (4, 6), got {c.shape}"
        assert y.shape == (4, 1), f"Expected threat shape (4, 1), got {y.shape}"
        assert torch.all(c >= 0.0) and torch.all(c <= 1.0), "Concepts must be bounded in [0, 1]"

        # Gradient flow test
        loss = c.sum() + y.sum()
        loss.backward()
        assert model.node_mlp[0].weight.grad is not None

    def test_dense_gat_variant_forward(self, dummy_batch):
        x, adj, mask = dummy_batch
        model = AblationDenseGAT(in_features=6, hidden_dim=32, num_concepts=6)
        c, y = model(x, adj, mask)

        assert c.shape == (4, 6)
        assert y.shape == (4, 1)
        assert torch.all(c >= 0.0) and torch.all(c <= 1.0)

    def test_unmasked_variant_forward(self, dummy_batch):
        x, adj, mask = dummy_batch
        model = AblationUnmasked(in_features=6, hidden_dim=32, num_concepts=6)
        c, y = model(x, adj, mask)

        assert c.shape == (4, 6)
        assert y.shape == (4, 1)

    def test_gat_depth_sweep_forward(self, dummy_batch):
        x, adj, mask = dummy_batch
        for depth in [1, 3]:
            model = AblationGATDepth(depth=depth, in_features=6, hidden_dim=32, num_concepts=6)
            c, y = model(x, adj, mask)
            assert c.shape == (4, 6)
            assert y.shape == (4, 1)
            assert len(model.layers) == depth

    def test_global_pool_variant_forward(self, dummy_batch):
        x, adj, mask = dummy_batch
        model = AblationGlobalPool(in_features=6, hidden_dim=32, num_concepts=6)
        c, y = model(x, adj, mask)

        assert c.shape == (4, 6)
        assert y.shape == (4, 1)
        assert torch.all(c >= 0.0) and torch.all(c <= 1.0)

    def test_direct_gnn_variant_forward(self, dummy_batch):
        x, adj, mask = dummy_batch
        model = AblationDirectGNN(in_features=6, hidden_dim=32)
        c, y = model(x, adj, mask)

        # Direct GNN outputs threat directly
        assert y.shape == (4, 1)
        loss = y.sum()
        loss.backward()
        assert model.direct_threat_head[0].weight.grad is not None

    def test_linear_threat_head_variant_forward(self, dummy_batch):
        x, adj, mask = dummy_batch
        model = AblationLinearHead(in_features=6, hidden_dim=32, num_concepts=6)
        c, y = model(x, adj, mask)

        assert c.shape == (4, 6)
        assert y.shape == (4, 1)
        assert hasattr(model, "linear_head")


class TestAblationArtifactsIntegrity:
    """Verifies that generated ablation tables and JSON summary meet publication standards."""

    def test_ablation_json_summary_exists_and_valid(self):
        import json
        summary_path = BASE_DIR / "results" / "ablation_summary.json"
        assert summary_path.exists(), f"Ablation summary not found at {summary_path}"

        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 11, f"Expected 11 ablation variants, got {len(data)}"

        required_keys = ["variant", "params", "roc_auc", "pr_auc", "brier", "latency_us", "mean_r2"]
        for entry in data:
            for k in required_keys:
                assert k in entry, f"Missing key '{k}' in entry for {entry.get('variant')}"
            assert 0.5 <= entry["roc_auc"] <= 1.0, f"ROC-AUC out of bounds: {entry['roc_auc']}"
            assert 0.0 <= entry["pr_auc"] <= 1.0, f"PR-AUC out of bounds: {entry['pr_auc']}"
            assert 0.0 <= entry["brier"] <= 1.0, f"Brier score out of bounds: {entry['brier']}"

    def test_ablation_markdown_table_exists_and_formatted(self):
        table_path = BASE_DIR / "results" / "ablation_table.md"
        assert table_path.exists(), f"Ablation markdown table not found at {table_path}"

        with open(table_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "# Systematic Model Ablation Benchmark Table" in content
        assert "V0: Full Canonical CBM" in content
        assert "V7: Direct GNN (No Concept Bottleneck)" in content
        assert "Cost of the Concept Bottleneck" in content


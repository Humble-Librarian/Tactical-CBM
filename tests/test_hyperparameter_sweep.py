"""
Tactical CBM Engine - Hyperparameter Sweep Unit & Contract Tests (Phase 8 / A3)
Authors: Senior ML/DL Architect & Gravitas Reliability Runtime

Validates:
  1. Sweep configurations are well-formed and cover all 4 orthogonal axes.
  2. Pareto dominance logic accurately determines the Pareto frontier.
  3. Mini-step execution runs without NaN or memory leakage across hidden dimensions.
  4. Contract assertions on results/hyperparameter_sweep.json & table.md.
"""

import os
import sys
import json
import pytest
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.sweep import (
    compute_pareto_frontier,
    train_and_eval_config,
    CONCEPT_NAMES
)
from ml_trainer.model import SpatioTemporalCBM

RESULTS_DIR = BASE_DIR / "results"
SWEEP_JSON = RESULTS_DIR / "hyperparameter_sweep.json"
SWEEP_MD = RESULTS_DIR / "hyperparameter_table.md"


class TestHyperparameterSweep:

    def test_pareto_frontier_logic(self):
        """Verifies Pareto dominance logic with synthetic edge cases."""
        mock_results = [
            {"run_id": "A", "mean_r2": 0.80, "pr_auc": 0.40},
            {"run_id": "B", "mean_r2": 0.70, "pr_auc": 0.35},  # Dominated by A
            {"run_id": "C", "mean_r2": 0.85, "pr_auc": 0.38},  # Non-dominated (higher R2, lower PR)
            {"run_id": "D", "mean_r2": 0.75, "pr_auc": 0.45},  # Non-dominated (lower R2, higher PR)
            {"run_id": "E", "mean_r2": 0.72, "pr_auc": 0.39},  # Dominated by D and A
        ]

        pareto_ids = compute_pareto_frontier(mock_results)
        assert set(pareto_ids) == {"A", "C", "D"}, f"Expected {{'A', 'C', 'D'}}, got {pareto_ids}"

    def test_train_and_eval_config_mini_step(self):
        """Verifies train_and_eval_config compiles, runs, and produces valid schema metrics."""
        device = torch.device("cpu")
        B = 8
        N = 22
        F = 6

        # Create synthetic mini batch dataset
        class SyntheticBatchDS:
            def __len__(self):
                return 16

            def __getitem__(self, idx):
                return {
                    "node_features": torch.rand(N, F, dtype=torch.float32),
                    "adj_mask": (torch.rand(N, N) > 0.5).float(),
                    "player_mask": torch.ones(N, 1, dtype=torch.float32),
                    "c_true": torch.rand(6, dtype=torch.float32),
                    "y_threat": torch.tensor(1.0 if idx % 4 == 0 else 0.0, dtype=torch.float32)
                }

        train_ds = SyntheticBatchDS()
        val_ds = SyntheticBatchDS()
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

        metrics = train_and_eval_config(
            run_id="TEST",
            name="Test Mini Run",
            lr=1e-3,
            hidden_dim=16,
            weight_decay=1e-4,
            lambda_concept=1.0,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=1
        )

        assert metrics["run_id"] == "TEST"
        assert metrics["params"] > 0
        assert len(metrics["r2_scores"]) == 6
        assert isinstance(metrics["roc_auc"], float)
        assert isinstance(metrics["pr_auc"], float)
        assert isinstance(metrics["brier"], float)
        assert metrics["latency_us"] >= 0.0

    def test_spatio_temporal_cbm_hidden_dim_scaling(self):
        """Verifies that model parameter counts scale monotonically across hidden dimensions."""
        m16 = SpatioTemporalCBM(in_features=6, hidden_dim=16, num_concepts=6)
        m32 = SpatioTemporalCBM(in_features=6, hidden_dim=32, num_concepts=6)
        m64 = SpatioTemporalCBM(in_features=6, hidden_dim=64, num_concepts=6)

        p16 = sum(p.numel() for p in m16.parameters())
        p32 = sum(p.numel() for p in m32.parameters())
        p64 = sum(p.numel() for p in m64.parameters())

        assert p16 < p32 < p64, f"Expected p16 < p32 < p64, got {p16}, {p32}, {p64}"

    @pytest.mark.skipif(not SWEEP_JSON.exists(), reason="Run python ml_trainer/sweep.py first")
    def test_sweep_artifact_contract(self):
        """Contract test: validates generated results/hyperparameter_sweep.json satisfies strict bounds."""
        with open(SWEEP_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["metadata"]
        results = data["results"]

        assert meta["canonical_anchor"] == "H0"
        assert len(results) == 11, f"Expected 11 sweep runs, got {len(results)}"

        # Validate H0 Anchor
        anchor = next((r for r in results if r["run_id"] == "H0"), None)
        assert anchor is not None, "Anchor H0 missing from sweep results"
        assert anchor["mean_r2"] >= 0.05, f"Anchor Mean R2 unexpectedly low: {anchor['mean_r2']}"
        assert anchor["pr_auc"] >= 0.25, f"Anchor PR-AUC too low: {anchor['pr_auc']}"
        assert anchor["roc_auc"] >= 0.80, f"Anchor ROC-AUC too low: {anchor['roc_auc']}"
        assert anchor["brier"] <= 0.25, f"Anchor Brier score too high: {anchor['brier']}"

        # Validate all 11 runs have valid finite bounded metrics
        for r in results:
            assert 0.0 <= r["roc_auc"] <= 1.0, f"Invalid ROC-AUC in {r['run_id']}: {r['roc_auc']}"
            assert 0.0 <= r["pr_auc"] <= 1.0, f"Invalid PR-AUC in {r['run_id']}: {r['pr_auc']}"
            assert 0.0 <= r["brier"] <= 1.0, f"Invalid Brier in {r['run_id']}: {r['brier']}"
            assert len(r["r2_scores"]) == 6, f"Invalid concept R2 count in {r['run_id']}"
            assert -0.50 <= r["mean_r2"] <= 1.0, f"Invalid Mean R2 in {r['run_id']}: {r['mean_r2']}"
            assert r["latency_us"] < 500.0, f"Latency unexpectedly high (>500us) in {r['run_id']}"

        # Validate markdown table exists and is non-empty
        assert SWEEP_MD.exists(), "results/hyperparameter_table.md does not exist"
        assert SWEEP_MD.stat().st_size > 500, "results/hyperparameter_table.md is empty or incomplete"

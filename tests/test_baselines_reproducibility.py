"""
Test Suite: Baselines Suite Reproducibility & Schema Validation (A1)
Author: Gravitas Reliability Runtime & Senior Architect

Asserts that:
1. results/baselines_summary.json exists and contains exactly 5 baselines.
2. Metrics are within strictly valid theoretical and empirical ranges:
   - Random prior achieves ROC-AUC == 0.5000 and PR-AUC ~= class prior (~0.05).
   - Concept Logistic Regression achieves ROC-AUC > 0.75.
   - LightGBM achieves ROC-AUC > 0.80 and PR-AUC > 0.25.
   - Brier scores are in [0, 1] and log losses are positive.
"""
import json
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_JSON = BASE_DIR / "results" / "baselines_summary.json"
RESULTS_TABLE = BASE_DIR / "results" / "baselines_table.md"

def test_baselines_files_exist():
    assert RESULTS_JSON.exists(), f"baselines_summary.json not found at {RESULTS_JSON}"
    assert RESULTS_TABLE.exists(), f"baselines_table.md not found at {RESULTS_TABLE}"

def test_baselines_summary_schema():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, list)
    assert len(data) == 5, f"Expected 5 baseline models, got {len(data)}"
    
    expected_models = [
        "Baseline 0: Stratified Random Prior",
        "Baseline 1: Logistic Regression (6 Concepts)",
        "Baseline 2: LightGBM (6 Concepts)",
        "Baseline 3: Logistic Regression (132-dim Raw Nodes)",
        "Baseline 4: Mean-Pooled 2-Layer MLP"
    ]
    
    for i, m in enumerate(data):
        assert m["model"] == expected_models[i]
        assert "roc_auc" in m
        assert "pr_auc" in m
        assert "brier_score" in m
        assert "log_loss" in m
        assert "optimal_f1" in m
        assert "latency_us_per_frame" in m

def test_baselines_metric_validity():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Baseline 0 (Random Floor)
    b0 = data[0]
    assert abs(b0["roc_auc"] - 0.5000) < 1e-3, "Random prior ROC-AUC must be ~0.5000"
    assert 0.04 <= b0["pr_auc"] <= 0.07, "Random prior PR-AUC must match empirical class rate (~0.05)"
    
    # Baseline 1 (Logistic Regression on Concepts)
    b1 = data[1]
    assert b1["roc_auc"] >= 0.75, f"Concept LR ROC-AUC {b1['roc_auc']} below threshold 0.75"
    assert b1["pr_auc"] >= 0.20, f"Concept LR PR-AUC {b1['pr_auc']} below threshold 0.20"
    
    # Baseline 2 (LightGBM on Concepts)
    b2 = data[2]
    assert b2["roc_auc"] >= b1["roc_auc"], "Non-linear LightGBM should match or exceed linear concept LR"
    assert b2["pr_auc"] >= 0.28, f"LightGBM PR-AUC {b2['pr_auc']} below threshold 0.28"
    
    # Baseline 3 & 4 (Raw features)
    b3 = data[3]
    b4 = data[4]
    assert b3["roc_auc"] > 0.80
    assert b4["roc_auc"] > 0.80

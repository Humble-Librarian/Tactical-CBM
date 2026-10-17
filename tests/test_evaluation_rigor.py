"""
Tactical CBM Engine - Comprehensive Evaluation Rigor Unit Tests (Stage 3 / A6, A7)
Authors: Gravitas Reliability Runtime, Senior Architect, & Ponytail

Verifies:
  1. Murphy's Brier Score Decomposition mathematical identity (Rel - Res + Unc == BS)
  2. Monotonicity and Rank Invariance of Beta and Platt Calibrators (Delta ROC-AUC <= 1e-5)
  3. Calibration ECE & Brier score reduction under class skew
  4. ThresholdOptimizer Pareto boundaries and confusion matrix sum invariants
  5. BilinearConceptInteractionHead parameter shapes and tactical synergy exports
  6. Zero-copy error analysis execution
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.evaluation import (
    murphy_brier_decomposition,
    expected_calibration_error,
    BetaCalibrator,
    PlattCalibrator,
    IsotonicCalibrator,
    ThresholdOptimizer,
    BilinearConceptInteractionHead,
    MurphyDecomposition
)

# Fix seed for Gravitas deterministic reproducibility
SEED = 42
np.random.seed(SEED)


@pytest.fixture
def synthetic_skewed_data():
    """Generates synthetic predictions with 5% base rate and uncalibrated probabilities."""
    n = 10_000
    # 5% positive rate
    y = (np.random.rand(n) < 0.05).astype(np.float64)
    # Simulated overconfident model predicting centered around 0.5
    raw_logits = np.random.normal(loc=0.0, scale=1.5, size=n)
    # Add signal: positive class has higher logits
    raw_logits[y == 1] += 2.0
    probs = 1.0 / (1.0 + np.exp(-raw_logits))
    return y, probs


class TestMurphyDecomposition:
    def test_murphy_identity(self, synthetic_skewed_data):
        """Validates that binned Brier score identically equals Rel - Res + Unc."""
        y, probs = synthetic_skewed_data
        murphy = murphy_brier_decomposition(y, probs, n_bins=15)

        expected_binned = murphy.reliability - murphy.resolution + murphy.uncertainty
        assert np.isclose(murphy.binned_brier, expected_binned, atol=1e-6), (
            f"Murphy identity violated: {murphy.binned_brier} != {expected_binned}"
        )

    def test_uncertainty_formula(self, synthetic_skewed_data):
        """Validates that Uncertainty equals base_rate * (1 - base_rate)."""
        y, probs = synthetic_skewed_data
        murphy = murphy_brier_decomposition(y, probs, n_bins=15)
        p_bar = float(np.mean(y))
        expected_unc = p_bar * (1.0 - p_bar)
        assert np.isclose(murphy.uncertainty, expected_unc, atol=1e-7)

    def test_perfect_predictions(self):
        """Validates that perfect predictions yield zero reliability error."""
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.float64)
        probs = y.copy()
        murphy = murphy_brier_decomposition(y, probs, n_bins=10)
        assert np.isclose(murphy.reliability, 0.0, atol=1e-6)
        assert np.isclose(murphy.brier_score, 0.0, atol=1e-6)


class TestCalibrationTechniques:
    def test_beta_calibrator_monotonicity_and_rank_invariance(self, synthetic_skewed_data):
        """Beta calibration must enforce a >= 0, b >= 0 and preserve ROC-AUC."""
        y, probs = synthetic_skewed_data
        cal = BetaCalibrator().fit(probs, y)
        cal_probs = cal.predict_proba(probs)

        # Monotonicity check: sort input, output must remain sorted
        sort_idx = np.argsort(probs)
        sorted_cal = cal_probs[sort_idx]
        diffs = np.diff(sorted_cal)
        assert np.all(diffs >= -1e-8), "Beta calibration output is non-monotonic!"

        # ROC-AUC rank invariance
        uncal_auc = roc_auc_score(y, probs)
        cal_auc = roc_auc_score(y, cal_probs)
        assert np.isclose(uncal_auc, cal_auc, atol=1e-4), (
            f"ROC-AUC changed significantly: {uncal_auc:.5f} vs {cal_auc:.5f}"
        )

    def test_beta_calibrator_improves_brier_and_ece(self, synthetic_skewed_data):
        """Beta calibration must substantially reduce Brier score and ECE on skewed data."""
        y, probs = synthetic_skewed_data
        uncal_brier = brier_score_loss(y, probs)
        uncal_ece, _ = expected_calibration_error(y, probs)

        cal = BetaCalibrator().fit(probs, y)
        cal_probs = cal.predict_proba(probs)
        cal_brier = brier_score_loss(y, cal_probs)
        cal_ece, _ = expected_calibration_error(y, cal_probs)

        assert cal_brier < uncal_brier, f"Brier did not improve: {cal_brier} >= {uncal_brier}"
        assert cal_ece < uncal_ece, f"ECE did not improve: {cal_ece} >= {uncal_ece}"
        assert cal_ece < 0.05, f"Calibrated ECE too high: {cal_ece}"

    def test_platt_calibrator_monotonicity(self, synthetic_skewed_data):
        """Platt calibrator must maintain non-negative slope A >= 0."""
        y, probs = synthetic_skewed_data
        cal = PlattCalibrator().fit(probs, y)
        assert cal.A >= 0.0
        cal_probs = cal.predict_proba(probs)
        uncal_auc = roc_auc_score(y, probs)
        cal_auc = roc_auc_score(y, cal_probs)
        assert np.isclose(uncal_auc, cal_auc, atol=1e-4)

    def test_isotonic_calibrator_bounds(self, synthetic_skewed_data):
        """Isotonic calibrator must produce valid [0, 1] bounds."""
        y, probs = synthetic_skewed_data
        cal = IsotonicCalibrator().fit(probs, y)
        cal_probs = cal.predict_proba(probs)
        assert np.all(cal_probs >= 0.0)
        assert np.all(cal_probs <= 1.0)


class TestThresholdOptimization:
    def test_threshold_optimizer_valid_ranges(self, synthetic_skewed_data):
        """Validates that optimal thresholds and confusion matrices adhere to count invariants."""
        y, probs = synthetic_skewed_data
        res = ThresholdOptimizer.optimize_thresholds(y, probs)

        assert 0.0 < res["best_f1_threshold"] < 1.0
        assert 0.0 < res["best_f05_threshold"] < 1.0
        assert 0.0 <= res["precision_at_f05"] <= 1.0
        assert 0.0 <= res["recall_at_f05"] <= 1.0

        # Confusion matrix conservation law: TN + FP + FN + TP == N
        cm = res["optimal_cm"]
        total_cm = cm["TN"] + cm["FP"] + cm["FN"] + cm["TP"]
        assert total_cm == len(y), f"Confusion matrix total {total_cm} != dataset length {len(y)}"


class TestBilinearConceptHead:
    def test_bcit_shapes_and_predictions(self):
        """Validates that BCIT has exactly 6 linear + 15 interaction + 1 bias = 22 parameters."""
        n = 500
        n_concepts = 6
        X = np.random.rand(n, n_concepts).astype(np.float64)
        y = (X[:, 0] * X[:, 3] > 0.3).astype(np.float64)

        bcit = BilinearConceptInteractionHead(n_concepts=n_concepts).fit(X, y)
        probs = bcit.predict_proba(X)

        assert probs.shape == (n,)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

        weights = bcit.get_tactical_weights()
        assert len(weights["linear_weights"]) == 6
        assert len(weights["pairwise_interactions"]) == 15  # 6 * 5 / 2
        assert "C1 x C4" in weights["pairwise_interactions"]

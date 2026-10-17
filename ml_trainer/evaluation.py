"""
Tactical CBM Engine - Comprehensive Evaluation & Calibration Suite (Stage 3 / A6, A7)
Authors: Gravitas Reliability Runtime, Senior Architect, & Ponytail

Implements mathematically rigorous evaluation tools for class-skewed spatio-temporal data:
  1. Murphy's Brier Score 3-Component Decomposition (Reliability - Resolution + Uncertainty)
  2. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
  3. Beta Calibration (Kull et al., 2017) via convex L-BFGS-B optimization
  4. Platt Sigmoidal Scaling and Non-Parametric Isotonic Calibration
  5. Cost-Sensitive & F-beta (F0.5, F1) Pareto Optimal Decision Thresholding
  6. Bilinear Concept Interaction Head (BCIT) modeling pairwise tactical synergies
  7. Publication-grade Diagnostic Plot Generation (PR Curves, Reliability Diagrams, Confusion Matrices)
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPSILON = 1e-7


@dataclass
class MurphyDecomposition:
    brier_score: float
    reliability: float
    resolution: float
    uncertainty: float
    binned_brier: float
    n_bins: int
    bin_counts: List[int]
    bin_accuracies: List[float]
    bin_confidences: List[float]


@dataclass
class CalibrationReport:
    model_name: str
    calibrator_type: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss: float
    ece: float
    mce: float
    murphy: MurphyDecomposition
    optimal_f1: float
    optimal_f1_threshold: float
    optimal_f05: float
    optimal_f05_threshold: float
    precision_at_f05: float
    recall_at_f05: float
    default_cm: Dict[str, int]
    optimal_cm: Dict[str, int]


class BetaCalibrator:
    """
    Monotonic Beta Calibration for binary classification (Kull, Silva Filho, & Flach, 2017).
    Calibrates probabilities under class skew via logistic representation:
      ln(p / (1 - p)) = a * ln(s) - b * ln(1 - s) + c
    Constrained to a >= 0, b >= 0 to ensure strict monotonicity (ROC-AUC preserving).
    """
    def __init__(self):
        self.a: float = 1.0
        self.b: float = 1.0
        self.c: float = 0.0
        self.fitted: bool = False

    def _loss(self, params: np.ndarray, s_clipped: np.ndarray, y: np.ndarray) -> float:
        a, b, c = params
        log_odds = a * np.log(s_clipped) - b * np.log(1.0 - s_clipped) + c
        # Numerically stable binary cross-entropy:
        # loss = max(x, 0) - x * y + log(1 + exp(-abs(x)))
        loss = np.maximum(log_odds, 0.0) - log_odds * y + np.log1p(np.exp(-np.abs(log_odds)))
        return float(np.mean(loss))

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "BetaCalibrator":
        s = np.clip(probs, EPSILON, 1.0 - EPSILON)
        y = y.astype(np.float64)

        # Initial parameters: identity-like scaling
        x0 = np.array([1.0, 1.0, 0.0])
        bounds = [(0.0, None), (0.0, None), (None, None)]

        res = minimize(
            self._loss,
            x0,
            args=(s, y),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9}
        )

        self.a, self.b, self.c = res.x
        self.fitted = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("BetaCalibrator must be fitted before predict_proba.")
        s = np.clip(probs, EPSILON, 1.0 - EPSILON)
        log_odds = self.a * np.log(s) - self.b * np.log(1.0 - s) + self.c
        return expit(log_odds)


class PlattCalibrator:
    """
    Platt Sigmoidal Calibrator.
    Fits: ln(p / (1 - p)) = A * logit(s) + B
    Constrained to A >= 0 for strict ranking preservation.
    """
    def __init__(self):
        self.A: float = 1.0
        self.B: float = 0.0
        self.fitted: bool = False

    def _loss(self, params: np.ndarray, l_s: np.ndarray, y: np.ndarray) -> float:
        A, B = params
        log_odds = A * l_s + B
        loss = np.maximum(log_odds, 0.0) - log_odds * y + np.log1p(np.exp(-np.abs(log_odds)))
        return float(np.mean(loss))

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        s = np.clip(probs, EPSILON, 1.0 - EPSILON)
        l_s = logit(s)
        y = y.astype(np.float64)

        x0 = np.array([1.0, 0.0])
        bounds = [(0.0, None), (None, None)]

        res = minimize(
            self._loss,
            x0,
            args=(l_s, y),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9}
        )

        self.A, self.B = res.x
        self.fitted = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("PlattCalibrator must be fitted before predict_proba.")
        s = np.clip(probs, EPSILON, 1.0 - EPSILON)
        l_s = logit(s)
        return expit(self.A * l_s + self.B)


class IsotonicCalibrator:
    """
    Non-parametric Isotonic Regression Calibrator.
    Fits a non-decreasing piecewise constant function.
    """
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        self.fitted: bool = False

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(probs, y)
        self.fitted = True
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("IsotonicCalibrator must be fitted before predict_proba.")
        return np.clip(self.iso.predict(probs), 0.0, 1.0)


def murphy_brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> MurphyDecomposition:
    """
    Calculates Murphy's exact 3-component Brier Score decomposition:
      Brier = Reliability - Resolution + Uncertainty
    Where:
      - Reliability = (1/N) * sum_k n_k * (y_bar_k - p_bar_k)^2 (Calibration error)
      - Resolution  = (1/N) * sum_k n_k * (y_bar_k - y_bar)^2   (Discriminative capability)
      - Uncertainty = y_bar * (1 - y_bar)                       (Base rate entropy)
    """
    N = len(y_true)
    y_bar = float(np.mean(y_true))
    unc = float(y_bar * (1.0 - y_bar))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    rel = 0.0
    res = 0.0
    bin_counts = []
    bin_accuracies = []
    bin_confidences = []

    for k in range(n_bins):
        mask = (bin_indices == k)
        nk = int(np.sum(mask))
        bin_counts.append(nk)
        if nk > 0:
            yk = float(np.mean(y_true[mask]))
            pk = float(np.mean(y_prob[mask]))
            rel += (nk / N) * ((yk - pk) ** 2)
            res += (nk / N) * ((yk - y_bar) ** 2)
            bin_accuracies.append(yk)
            bin_confidences.append(pk)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((bins[k] + bins[k + 1]) / 2.0))

    binned_brier = rel - res + unc
    actual_brier = float(brier_score_loss(y_true, y_prob))

    return MurphyDecomposition(
        brier_score=actual_brier,
        reliability=float(rel),
        resolution=float(res),
        uncertainty=float(unc),
        binned_brier=float(binned_brier),
        n_bins=n_bins,
        bin_counts=bin_counts,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences
    )


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[float, float]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    with M uniform confidence bins.
    """
    N = len(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0

    for k in range(n_bins):
        mask = (bin_indices == k)
        nk = np.sum(mask)
        if nk > 0:
            yk = np.mean(y_true[mask])
            pk = np.mean(y_prob[mask])
            abs_diff = np.abs(yk - pk)
            ece += (nk / N) * abs_diff
            if abs_diff > mce:
                mce = float(abs_diff)

    return float(ece), float(mce)


class ThresholdOptimizer:
    """
    Evaluates classification thresholds on hold-out probabilities to maximize F-beta
    or minimize asymmetric cost matrix.
    """
    @staticmethod
    def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        # Avoid division by zero
        denom_f1 = np.maximum(precision + recall, 1e-8)
        f1_scores = 2.0 * (precision * recall) / denom_f1
        best_f1_idx = int(np.argmax(f1_scores))
        best_f1 = float(f1_scores[best_f1_idx])
        best_f1_t = float(thresholds[min(best_f1_idx, len(thresholds) - 1)])

        # F0.5: beta = 0.5 (Precision has 2x weight of Recall)
        beta = 0.5
        beta_sq = beta ** 2
        denom_f05 = np.maximum(beta_sq * precision + recall, 1e-8)
        f05_scores = (1.0 + beta_sq) * (precision * recall) / denom_f05
        best_f05_idx = int(np.argmax(f05_scores))
        best_f05 = float(f05_scores[best_f05_idx])
        best_f05_t = float(thresholds[min(best_f05_idx, len(thresholds) - 1)])
        p_at_f05 = float(precision[best_f05_idx])
        r_at_f05 = float(recall[best_f05_idx])

        # Confusion matrix at default (0.50)
        y_pred_default = (y_prob >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_default).ravel()
        default_cm = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

        # Confusion matrix at optimal F0.5
        y_pred_opt = (y_prob >= best_f05_t).astype(int)
        tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_true, y_pred_opt).ravel()
        optimal_cm = {"TN": int(tn_opt), "FP": int(fp_opt), "FN": int(fn_opt), "TP": int(tp_opt)}

        return {
            "best_f1": best_f1,
            "best_f1_threshold": best_f1_t,
            "best_f05": best_f05,
            "best_f05_threshold": best_f05_t,
            "precision_at_f05": p_at_f05,
            "recall_at_f05": r_at_f05,
            "default_cm": default_cm,
            "optimal_cm": optimal_cm
        }


class BilinearConceptInteractionHead:
    """
    Bilinear Concept Interaction Tensor (BCIT):
      y_hat = sigma( w^T c + c^T W_interact c + b )
    Where W_interact is an upper-triangular matrix with zero diagonal (15 pairwise interaction weights),
    w is a 6-dim linear weight vector, and b is a scalar bias.
    100% transparent and interpretable with explicit tactical conjunction parameters.
    """
    def __init__(self, n_concepts: int = 6):
        self.n = n_concepts
        # Total parameters: 6 linear + 15 interaction + 1 bias = 22 parameters
        self.w_linear = np.zeros(n_concepts, dtype=np.float64)
        self.w_interact = np.zeros((n_concepts, n_concepts), dtype=np.float64)
        self.bias: float = 0.0
        self.fitted: bool = False

    def _pack_params(self) -> np.ndarray:
        triu_idx = np.triu_indices(self.n, k=1)
        return np.concatenate([self.w_linear, self.w_interact[triu_idx], [self.bias]])

    def _unpack_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        n_lin = self.n
        w_lin = params[:n_lin]
        triu_idx = np.triu_indices(self.n, k=1)
        n_inter = len(triu_idx[0])
        w_int = np.zeros((self.n, self.n), dtype=np.float64)
        w_int[triu_idx] = params[n_lin:n_lin + n_inter]
        bias = float(params[-1])
        return w_lin, w_int, bias

    def _compute_logits(self, X: np.ndarray, w_lin: np.ndarray, w_int: np.ndarray, bias: float) -> np.ndarray:
        # X: [N, 6]
        # Linear term: X @ w_lin -> [N]
        lin_term = X @ w_lin
        # Interaction term: sum over j > i of X[:, i] * X[:, j] * w_int[i, j]
        triu_i, triu_j = np.triu_indices(self.n, k=1)
        inter_terms = (X[:, triu_i] * X[:, triu_j]) @ w_int[triu_i, triu_j]
        return lin_term + inter_terms + bias

    def _loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray, l2_reg: float = 1e-4) -> float:
        w_lin, w_int, bias = self._unpack_params(params)
        logits = self._compute_logits(X, w_lin, w_int, bias)
        bce = np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
        reg = l2_reg * (np.sum(w_lin ** 2) + np.sum(w_int ** 2))
        return float(np.mean(bce) + reg)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BilinearConceptInteractionHead":
        # Initial parameters: warm start with zeros, bias = logit(mean(y))
        y_mean = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        init_bias = float(logit(y_mean))
        triu_idx = np.triu_indices(self.n, k=1)
        x0 = np.concatenate([np.zeros(self.n), np.zeros(len(triu_idx[0])), [init_bias]])

        res = minimize(
            self._loss,
            x0,
            args=(X, y),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-9}
        )

        self.w_linear, self.w_interact, self.bias = self._unpack_params(res.x)
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("BilinearConceptInteractionHead must be fitted before predict_proba.")
        logits = self._compute_logits(X, self.w_linear, self.w_interact, self.bias)
        return expit(logits)

    def get_tactical_weights(self, concept_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Returns readable table of linear weights and pairwise interaction synergies."""
        if concept_names is None:
            concept_names = [f"C{i+1}" for i in range(self.n)]
        
        linear_dict = {concept_names[i]: float(self.w_linear[i]) for i in range(self.n)}
        interactions = {}
        triu_i, triu_j = np.triu_indices(self.n, k=1)
        for i, j in zip(triu_i, triu_j):
            name = f"{concept_names[i]} x {concept_names[j]}"
            interactions[name] = float(self.w_interact[i, j])
            
        return {
            "linear_weights": linear_dict,
            "pairwise_interactions": interactions,
            "bias": float(self.bias)
        }


def run_full_calibration_benchmark(
    y_true: np.ndarray,
    uncal_probs: np.ndarray,
    model_name: str
) -> Dict[str, CalibrationReport]:
    """
    Evaluates uncalibrated model probabilities against Beta, Platt, and Isotonic Calibrators.
    Returns full calibration reports with Murphy decompositions.
    """
    # 1. Uncalibrated Baseline
    uncal_ece, uncal_mce = expected_calibration_error(y_true, uncal_probs)
    uncal_murphy = murphy_brier_decomposition(y_true, uncal_probs)
    uncal_thresh = ThresholdOptimizer.optimize_thresholds(y_true, uncal_probs)
    uncal_report = CalibrationReport(
        model_name=model_name,
        calibrator_type="None (Raw Output)",
        roc_auc=float(roc_auc_score(y_true, uncal_probs)),
        pr_auc=float(average_precision_score(y_true, uncal_probs)),
        brier_score=float(brier_score_loss(y_true, uncal_probs)),
        log_loss=float(log_loss(y_true, np.clip(uncal_probs, EPSILON, 1.0 - EPSILON))),
        ece=uncal_ece,
        mce=uncal_mce,
        murphy=uncal_murphy,
        optimal_f1=uncal_thresh["best_f1"],
        optimal_f1_threshold=uncal_thresh["best_f1_threshold"],
        optimal_f05=uncal_thresh["best_f05"],
        optimal_f05_threshold=uncal_thresh["best_f05_threshold"],
        precision_at_f05=uncal_thresh["precision_at_f05"],
        recall_at_f05=uncal_thresh["recall_at_f05"],
        default_cm=uncal_thresh["default_cm"],
        optimal_cm=uncal_thresh["optimal_cm"]
    )

    # 2. Beta Calibration
    beta_cal = BetaCalibrator().fit(uncal_probs, y_true)
    beta_probs = beta_cal.predict_proba(uncal_probs)
    beta_ece, beta_mce = expected_calibration_error(y_true, beta_probs)
    beta_murphy = murphy_brier_decomposition(y_true, beta_probs)
    beta_thresh = ThresholdOptimizer.optimize_thresholds(y_true, beta_probs)
    beta_report = CalibrationReport(
        model_name=model_name,
        calibrator_type="Beta Calibration (Kull et al., 2017)",
        roc_auc=float(roc_auc_score(y_true, beta_probs)),
        pr_auc=float(average_precision_score(y_true, beta_probs)),
        brier_score=float(brier_score_loss(y_true, beta_probs)),
        log_loss=float(log_loss(y_true, np.clip(beta_probs, EPSILON, 1.0 - EPSILON))),
        ece=beta_ece,
        mce=beta_mce,
        murphy=beta_murphy,
        optimal_f1=beta_thresh["best_f1"],
        optimal_f1_threshold=beta_thresh["best_f1_threshold"],
        optimal_f05=beta_thresh["best_f05"],
        optimal_f05_threshold=beta_thresh["best_f05_threshold"],
        precision_at_f05=beta_thresh["precision_at_f05"],
        recall_at_f05=beta_thresh["recall_at_f05"],
        default_cm=beta_thresh["default_cm"],
        optimal_cm=beta_thresh["optimal_cm"]
    )

    # 3. Platt Calibration
    platt_cal = PlattCalibrator().fit(uncal_probs, y_true)
    platt_probs = platt_cal.predict_proba(uncal_probs)
    platt_ece, platt_mce = expected_calibration_error(y_true, platt_probs)
    platt_murphy = murphy_brier_decomposition(y_true, platt_probs)
    platt_thresh = ThresholdOptimizer.optimize_thresholds(y_true, platt_probs)
    platt_report = CalibrationReport(
        model_name=model_name,
        calibrator_type="Platt Sigmoidal Scaling",
        roc_auc=float(roc_auc_score(y_true, platt_probs)),
        pr_auc=float(average_precision_score(y_true, platt_probs)),
        brier_score=float(brier_score_loss(y_true, platt_probs)),
        log_loss=float(log_loss(y_true, np.clip(platt_probs, EPSILON, 1.0 - EPSILON))),
        ece=platt_ece,
        mce=platt_mce,
        murphy=platt_murphy,
        optimal_f1=platt_thresh["best_f1"],
        optimal_f1_threshold=platt_thresh["best_f1_threshold"],
        optimal_f05=platt_thresh["best_f05"],
        optimal_f05_threshold=platt_thresh["best_f05_threshold"],
        precision_at_f05=platt_thresh["precision_at_f05"],
        recall_at_f05=platt_thresh["recall_at_f05"],
        default_cm=platt_thresh["default_cm"],
        optimal_cm=platt_thresh["optimal_cm"]
    )

    # 4. Isotonic Calibration
    iso_cal = IsotonicCalibrator().fit(uncal_probs, y_true)
    iso_probs = iso_cal.predict_proba(uncal_probs)
    iso_ece, iso_mce = expected_calibration_error(y_true, iso_probs)
    iso_murphy = murphy_brier_decomposition(y_true, iso_probs)
    iso_thresh = ThresholdOptimizer.optimize_thresholds(y_true, iso_probs)
    iso_report = CalibrationReport(
        model_name=model_name,
        calibrator_type="Isotonic Non-Parametric Regression",
        roc_auc=float(roc_auc_score(y_true, iso_probs)),
        pr_auc=float(average_precision_score(y_true, iso_probs)),
        brier_score=float(brier_score_loss(y_true, iso_probs)),
        log_loss=float(log_loss(y_true, np.clip(iso_probs, EPSILON, 1.0 - EPSILON))),
        ece=iso_ece,
        mce=iso_mce,
        murphy=iso_murphy,
        optimal_f1=iso_thresh["best_f1"],
        optimal_f1_threshold=iso_thresh["best_f1_threshold"],
        optimal_f05=iso_thresh["best_f05"],
        optimal_f05_threshold=iso_thresh["best_f05_threshold"],
        precision_at_f05=iso_thresh["precision_at_f05"],
        recall_at_f05=iso_thresh["recall_at_f05"],
        default_cm=iso_thresh["default_cm"],
        optimal_cm=iso_thresh["optimal_cm"]
    )

    return {
        "raw": uncal_report,
        "beta": beta_report,
        "platt": platt_report,
        "isotonic": iso_report
    }


def generate_diagnostic_plots(
    y_true: np.ndarray,
    uncal_probs: np.ndarray,
    cal_probs: np.ndarray,
    model_name: str,
    output_dir: Path
):
    """
    Generates high-resolution diagnostic plots for scientific publications:
      1. Reliability Diagram with Bin Histograms (ECE comparison)
      2. Precision-Recall Curves with F-beta isocontours
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Figure 1: Reliability Diagram ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Perfect calibration reference
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    
    # Uncalibrated curve
    uncal_murphy = murphy_brier_decomposition(y_true, uncal_probs, n_bins=15)
    uncal_ece, _ = expected_calibration_error(y_true, uncal_probs)
    ax1.plot(
        uncal_murphy.bin_confidences,
        uncal_murphy.bin_accuracies,
        "s-",
        color="#e74c3c",
        label=f"Uncalibrated (ECE={uncal_ece*100:.1f}%, Brier={uncal_murphy.brier_score:.3f})",
        linewidth=2
    )
    
    # Calibrated curve (Beta)
    cal_murphy = murphy_brier_decomposition(y_true, cal_probs, n_bins=15)
    cal_ece, _ = expected_calibration_error(y_true, cal_probs)
    ax1.plot(
        cal_murphy.bin_confidences,
        cal_murphy.bin_accuracies,
        "o-",
        color="#2ecc71",
        label=f"Beta Calibrated (ECE={cal_ece*100:.1f}%, Brier={cal_murphy.brier_score:.3f})",
        linewidth=2
    )
    
    ax1.set_ylabel("Empirical True Threat Rate", fontsize=12)
    ax1.set_title(f"Reliability Diagram: {model_name} (Murphy Decomposition)", fontsize=14, fontweight='bold')
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Lower histogram of sample counts
    bins = np.linspace(0.0, 1.0, 16)
    ax2.hist(uncal_probs, bins=bins, color="#e74c3c", alpha=0.5, label="Uncalibrated", log=True)
    ax2.hist(cal_probs, bins=bins, color="#2ecc71", alpha=0.5, label="Calibrated", log=True)
    ax2.set_xlabel("Predicted Probability Confidence", fontsize=12)
    ax2.set_ylabel("Log Count", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(output_dir / "reliability_diagram.png", dpi=300)
    plt.close(fig)

    # --- Figure 2: Precision-Recall Curve ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot iso-F1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2, linestyle='--')
        plt.annotate(f'F1={f_score:.1f}', xy=(0.88, y[x >= 0.90][0] if np.any(x >= 0.90) else 0.5), fontsize=8, alpha=0.5)

    p_uncal, r_uncal, _ = precision_recall_curve(y_true, uncal_probs)
    pr_auc_uncal = average_precision_score(y_true, uncal_probs)
    ax.plot(r_uncal, p_uncal, color="#3498db", linewidth=2, label=f"{model_name} (PR-AUC = {pr_auc_uncal:.4f})")
    
    # Baseline chance floor
    base_rate = float(np.mean(y_true))
    ax.axhline(base_rate, color="k", linestyle=":", label=f"Random Prior Baseline ({base_rate*100:.2f}%)")

    ax.set_xlabel("Recall (Coverage of True Threats)", fontsize=12)
    ax.set_ylabel("Precision (Actionable Coaching Alarms)", fontsize=12)
    ax.set_title(f"Precision-Recall Curve Under Class Skew: {model_name}", fontsize=14, fontweight='bold')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=300)
    plt.close(fig)
    print(f"Generated diagnostic plots in {output_dir}", flush=True)

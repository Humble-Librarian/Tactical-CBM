"""
Tactical CBM Engine - Cross-Competition & Generalization Test Suite (Phase 7 - Step 3)
Tests:
  1. Generalization summary JSON and markdown artifacts exist and match schema contracts
  2. Cross-competition PR-AUC and ROC-AUC stability (no catastrophic tournament collapse)
  3. Flank gap bounds (left vs right touchline Delta PR-AUC < 0.03)
  4. Camera occlusion degradation tolerance (< 5% PR-AUC drop under 2 occluded players)
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

RESULTS_DIR = BASE_DIR / "results"
SUMMARY_JSON = RESULTS_DIR / "generalization_summary.json"
REPORT_MD = RESULTS_DIR / "generalization_results.md"

def test_generalization_artifacts_exist_and_valid():
    """Verify Phase 7 generalization artifacts are generated and follow schema."""
    assert SUMMARY_JSON.exists(), "generalization_summary.json must exist"
    assert REPORT_MD.exists(), "generalization_results.md must exist"
    assert REPORT_MD.stat().st_size > 1000, "Markdown report must contain full evaluation tables"

    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("total_frames_evaluated", 0) > 50000
    assert "per_competition" in data
    assert "flank_asymmetry" in data
    assert "mirroring_consistency" in data
    assert "occlusion_stress_test" in data

def test_cross_competition_stability():
    """Verify model maintains strong performance across all 4 distinct competitions."""
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    comps = data["per_competition"]
    expected_comps = [
        "FIFA World Cup 2022",
        "UEFA Euro 2024",
        "1. Bundesliga 2023/24 (Leverkusen & Bayern)",
        "Ligue 1 2022/23 (PSG Domination)"
    ]

    for c in expected_comps:
        assert c in comps, f"Missing evaluation for competition: {c}"
        m = comps[c]
        # ROC-AUC must beat 0.80 across every tournament
        assert m["roc_auc"] > 0.80, f"{c} ROC-AUC {m['roc_auc']} fell below 0.80"
        # PR-AUC must beat 0.30 across every tournament
        assert m["pr_auc"] > 0.30, f"{c} PR-AUC {m['pr_auc']} fell below 0.30"
        # Concept R2 must beat 0.50 across every tournament
        assert m["concept_mean_r2"] > 0.50, f"{c} Concept R2 {m['concept_mean_r2']} fell below 0.50"

def test_flank_asymmetry_tolerance():
    """Verify left-vs-right flank threat prediction gap is minimal (< 0.03)."""
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    flank = data["flank_asymmetry"]
    delta_roc = flank.get("delta_roc_auc", 1.0)
    delta_pr = flank.get("delta_pr_auc", 1.0)

    assert delta_roc < 0.03, f"Flank ROC-AUC gap {delta_roc:.4f} exceeds 0.03 tolerance"
    assert delta_pr < 0.03, f"Flank PR-AUC gap {delta_pr:.4f} exceeds 0.03 tolerance"

def test_camera_occlusion_graceful_degradation():
    """Verify camera occlusion does not trigger catastrophic PR-AUC collapse (< 5% relative drop)."""
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    occ = data["occlusion_stress_test"]
    base_pr = occ["0_dropped_players"]["pr_auc"]
    drop2_pr = occ["2_dropped_players"]["pr_auc"]

    relative_drop = (base_pr - drop2_pr) / base_pr
    assert relative_drop < 0.05, f"Relative PR-AUC drop {relative_drop:.4f} exceeds 5% under 2 dropped players"

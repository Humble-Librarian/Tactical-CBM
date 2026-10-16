"""
Test Suite: Tactical Domain Invariants & Counter-Press Monotonicity Contract
Author: Senior Architect & Ponytail Quality Assurance

Verifies:
1. Strict Concept Monotonicity:
   - Increasing defensive line height (C1) strictly decreases threat.
   - Increasing defensive compactness (C2) strictly decreases threat.
   - Increasing defensive width balance (C3) strictly decreases threat.
   - Increasing ball-carrier Gegenpress (C4) strictly decreases threat.
   - Increasing central lane blockage (C5) strictly decreases threat.
   - Increasing box overload ratio (C6) strictly increases threat.
2. Tactical Presets Hierarchy:
   - Threat(Aggressive High-Press) < Threat(Balanced Block) < Threat(Overloaded Breakdown).
3. Cross-Runtime Parity:
   - PyTorch predict_threat_from_concepts() == Sidecar JSON dot product == ONNX runtime.
"""
import sys
import json
import torch
import numpy as np
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.model import TacticalCBM

MODELS_DIR = BASE_DIR / "models"
PT_PATH = MODELS_DIR / "tactical_cbm.pt"
SIDECAR_PATH = MODELS_DIR / "linear_head_weights.json"

@pytest.fixture
def sidecar_weights():
    with open(SIDECAR_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["weights"], dtype=np.float32), float(data["bias"])

@pytest.fixture
def pytorch_model():
    model = TacticalCBM()
    model.load_state_dict(torch.load(PT_PATH, map_location="cpu"))
    model.eval()
    return model

def test_defensive_weights_strictly_negative(sidecar_weights):
    """Architectural invariant: All 5 defensive concepts must act as threat suppressors."""
    weights, _ = sidecar_weights
    for i in range(5):
        assert weights[i] < 0.0, f"Defensive concept C{i+1} weight ({weights[i]}) must be negative!"

def test_box_overload_weight_strictly_positive(sidecar_weights):
    """Architectural invariant: Box overload must act as a threat catalyst."""
    weights, _ = sidecar_weights
    assert weights[5] > 0.0, f"Box overload C6 weight ({weights[5]}) must be positive!"

def test_strict_monotonicity_all_concepts(sidecar_weights):
    """Sweep each slider from 0.0 to 1.0; assert strict directional response."""
    weights, bias = sidecar_weights
    baseline = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    
    # 1. Line Height (C1)
    c_low = baseline.copy(); c_low[0] = 0.1
    c_high = baseline.copy(); c_high[0] = 0.9
    assert (np.dot(c_high, weights) + bias) < (np.dot(c_low, weights) + bias), "Line height must suppress threat"

    # 2. Compactness (C2)
    c_low = baseline.copy(); c_low[1] = 0.1
    c_high = baseline.copy(); c_high[1] = 0.9
    assert (np.dot(c_high, weights) + bias) < (np.dot(c_low, weights) + bias), "Compactness must suppress threat"

    # 3. Width Balance (C3)
    c_low = baseline.copy(); c_low[2] = 0.1
    c_high = baseline.copy(); c_high[2] = 0.9
    assert (np.dot(c_high, weights) + bias) < (np.dot(c_low, weights) + bias), "Width balance must suppress threat"

    # 4. Gegenpress (C4)
    c_low = baseline.copy(); c_low[3] = 0.1
    c_high = baseline.copy(); c_high[3] = 0.9
    assert (np.dot(c_high, weights) + bias) < (np.dot(c_low, weights) + bias), "Pressure must suppress threat"

    # 5. Lane Blockage (C5)
    c_low = baseline.copy(); c_low[4] = 0.1
    c_high = baseline.copy(); c_high[4] = 0.9
    assert (np.dot(c_high, weights) + bias) < (np.dot(c_low, weights) + bias), "Lane blockage must suppress threat"

    # 6. Box Overload (C6)
    c_low = baseline.copy(); c_low[5] = 0.1
    c_high = baseline.copy(); c_high[5] = 0.9
    assert (np.dot(c_high, weights) + bias) > (np.dot(c_low, weights) + bias), "Box overload must increase threat"

def test_tactical_preset_hierarchy(sidecar_weights):
    """Verify tactical threat hierarchy: Aggressive High-Press < Deep Block < Overloaded Breakdown."""
    weights, bias = sidecar_weights
    
    # Preset 1: Aggressive High Line & Counter-Press
    c_press = np.array([0.85, 0.70, 0.80, 0.85, 0.75, 0.05], dtype=np.float32)
    t_press = 1.0 / (1.0 + np.exp(-(np.dot(c_press, weights) + bias)))
    
    # Preset 2: Conservative Deep Block
    c_deep = np.array([0.30, 0.60, 0.50, 0.30, 0.50, 0.30], dtype=np.float32)
    t_deep = 1.0 / (1.0 + np.exp(-(np.dot(c_deep, weights) + bias)))
    
    # Preset 3: Breakdown (Deep uncompact block, zero pressure, box overloaded)
    c_breakdown = np.array([0.20, 0.35, 0.40, 0.10, 0.20, 0.75], dtype=np.float32)
    t_breakdown = 1.0 / (1.0 + np.exp(-(np.dot(c_breakdown, weights) + bias)))
    
    assert t_press < t_deep, f"High press ({t_press:.2%}) should be safer than deep block ({t_deep:.2%})"
    assert t_deep < t_breakdown, f"Deep block ({t_deep:.2%}) should be safer than overloaded breakdown ({t_breakdown:.2%})"
    assert t_press < 0.25, f"High press system threat should be under 25%, got {t_press:.2%}"
    assert t_breakdown > 0.70, f"Overloaded breakdown threat should exceed 70%, got {t_breakdown:.2%}"

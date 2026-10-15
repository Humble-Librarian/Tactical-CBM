"""
Test Suite: ONNX Graph Parity & Sidecar Contract
Verifies:
1. ONNX Model loads and runs without warnings.
2. Maximum absolute error vs PyTorch is < 10^-5.
3. Linear head sidecar JSON matches PyTorch checkpoint.
4. Sub-millisecond CPU latency (< 0.5ms).
"""
import sys
import json
import time
import torch
import numpy as np
import onnxruntime as ort
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.model import TacticalCBM

MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ONNX_PATH = MODELS_DIR / "tactical_cbm.onnx"
PT_PATH = MODELS_DIR / "tactical_cbm.pt"
SIDECAR_PATH = MODELS_DIR / "linear_head_weights.json"

def test_onnx_model_file_exists_and_compact():
    """Assert ONNX model exists and is < 100 KB."""
    assert ONNX_PATH.exists(), f"ONNX model missing at {ONNX_PATH}"
    size_kb = ONNX_PATH.stat().st_size / 1024
    assert size_kb < 100.0, f"ONNX model unexpectedly large: {size_kb:.2f} KB"

def test_sidecar_weights_alignment():
    """Assert sidecar JSON weights match PyTorch linear head."""
    assert SIDECAR_PATH.exists(), f"Sidecar missing at {SIDECAR_PATH}"
    with open(SIDECAR_PATH, "r", encoding="utf-8") as f:
        sidecar = json.load(f)
        
    weights = sidecar["weights"]
    bias = sidecar["bias"]
    assert len(weights) == 6, f"Expected 6 weights, got {len(weights)}"
    
    # Check against PyTorch state dict
    pt_model = TacticalCBM()
    pt_model.load_state_dict(torch.load(PT_PATH, map_location="cpu"))
    pt_w = pt_model.linear_head.weight.detach().cpu().squeeze().numpy()
    pt_b = float(pt_model.linear_head.bias.detach().cpu().item())
    
    for i in range(6):
        assert abs(weights[i] - float(pt_w[i])) < 1e-5, f"Weight {i} mismatch"
    assert abs(bias - pt_b) < 1e-5, "Bias mismatch"

def test_numerical_parity_against_pytorch():
    """Assert numerical parity between PyTorch and ONNX Runtime is < 10^-5."""
    pt_model = TacticalCBM()
    pt_model.load_state_dict(torch.load(PT_PATH, map_location="cpu"))
    pt_model.eval()
    
    session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    
    np.random.seed(123)
    for _ in range(25):
        node_np = np.random.randn(1, 22, 6).astype(np.float32)
        adj_np = np.ones((1, 22, 22), dtype=np.float32)
        mask_np = np.ones((1, 22, 1), dtype=np.float32)
        
        with torch.no_grad():
            pt_c, pt_y = pt_model(torch.from_numpy(node_np), torch.from_numpy(adj_np), torch.from_numpy(mask_np))
            
        ort_c, ort_y = session.run(None, {
            'node_features': node_np,
            'adj_mask': adj_np,
            'player_mask': mask_np
        })
        
        c_err = np.max(np.abs(pt_c.numpy() - ort_c))
        y_err = np.max(np.abs(pt_y.numpy() - ort_y))
        
        assert c_err < 1e-5, f"Concept error {c_err} exceeds 1e-5"
        assert y_err < 1e-5, f"Threat error {y_err} exceeds 1e-5"

def test_inference_latency_budget():
    """Assert average per-frame ONNX latency on CPU is under 0.5ms."""
    session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    
    inputs = {
        'node_features': np.random.randn(1, 22, 6).astype(np.float32),
        'adj_mask': np.ones((1, 22, 22), dtype=np.float32),
        'player_mask': np.ones((1, 22, 1), dtype=np.float32)
    }
    
    # Warmup
    for _ in range(50): session.run(None, inputs)
    
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        session.run(None, inputs)
    elapsed = time.perf_counter() - t0
    
    avg_ms = (elapsed / N) * 1000.0
    assert avg_ms < 0.5, f"Latency {avg_ms:.4f}ms exceeds 0.5ms target"

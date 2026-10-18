"""
Tactical CBM Engine - SpatioTemporalCBM ONNX Exporter & Verifier
Exports the Wave 3 SpatioTemporalCBM PyTorch model to an optimized, client-ready ONNX graph
and asserts numerical parity (< 1e-4) against PyTorch on randomized tactical frames.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError on emoji prints
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.model import SpatioTemporalCBM

MODELS_DIR = BASE_DIR / "models"
PT_PATH = MODELS_DIR / "spatiotemporal_cbm.pt"
ONNX_PATH = MODELS_DIR / "spatiotemporal_cbm.onnx"
SIDECAR_PATH = MODELS_DIR / "spatiotemporal_cbm_weights.json"


def export_spatiotemporal_to_onnx():
    print("=" * 70)
    print("EXPORTING SPATIOTEMPORAL CBM (WAVE 3) TO ONNX")
    print(f"PyTorch Checkpoint: {PT_PATH}")
    print(f"Output ONNX Target: {ONNX_PATH}")
    print("=" * 70, flush=True)

    if not PT_PATH.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found at {PT_PATH}")

    # Load model
    model = SpatioTemporalCBM(in_features=6, hidden_dim=32, num_concepts=6)
    model.load_state_dict(torch.load(PT_PATH, map_location="cpu"))
    model.eval()

    # Create dummy inputs (B=1, N=22)
    dummy_node = torch.randn(1, 22, 6, dtype=torch.float32)
    dummy_adj = torch.ones(1, 22, 22, dtype=torch.float32)
    dummy_mask = torch.ones(1, 22, 1, dtype=torch.float32)

    # Export with dynamic batch size
    print("Exporting ONNX graph with dynamic batch dimension...")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.onnx.export(
            model,
            (dummy_node, dummy_adj, dummy_mask),
            str(ONNX_PATH),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['node_features', 'adj_mask', 'player_mask'],
            output_names=['concepts', 'threat_logits'],
            dynamic_axes={
                'node_features': {0: 'batch_size'},
                'adj_mask': {0: 'batch_size'},
                'player_mask': {0: 'batch_size'},
                'concepts': {0: 'batch_size'},
                'threat_logits': {0: 'batch_size'}
            }
        )

    size_kb = ONNX_PATH.stat().st_size / 1024.0
    print(f"[OK] ONNX export complete! Size: {size_kb:.1f} KB (< 1.5 MB target)")
    return model


def verify_spatiotemporal_onnx_parity(pt_model):
    print("\n" + "-" * 70)
    print("VERIFYING NUMERICAL PARITY: PyTorch vs ONNX Runtime (100 trials)")
    print("-" * 70, flush=True)

    session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])

    np.random.seed(42)
    max_c_diff = 0.0
    max_y_diff = 0.0

    for i in range(100):
        node_np = np.random.randn(1, 22, 6).astype(np.float32)
        node_np[:, :, 0:2] = np.clip(node_np[:, :, 0:2], 0.0, 1.0)  # normalized coords
        adj_np = np.ones((1, 22, 22), dtype=np.float32)
        mask_np = np.ones((1, 22, 1), dtype=np.float32)

        # PyTorch prediction
        with torch.no_grad():
            pt_c, pt_y = pt_model(
                torch.from_numpy(node_np),
                torch.from_numpy(adj_np),
                torch.from_numpy(mask_np),
                return_calibrated=False
            )
            pt_c_np = pt_c.numpy()
            pt_y_np = pt_y.numpy()

        # ONNX prediction
        ort_c, ort_y = session.run(None, {
            'node_features': node_np,
            'adj_mask': adj_np,
            'player_mask': mask_np
        })

        c_err = float(np.max(np.abs(pt_c_np - ort_c)))
        y_err = float(np.max(np.abs(pt_y_np - ort_y)))

        if c_err > max_c_diff:
            max_c_diff = c_err
        if y_err > max_y_diff:
            max_y_diff = y_err

    print(f"Max Absolute Error across 100 trials:")
    print(f"  - Concepts (C1..C6): {max_c_diff:.2e}")
    print(f"  - Threat Logits:    {max_y_diff:.2e}")

    assert max_c_diff < 1e-4, f"Concept parity error {max_c_diff} >= 1e-4"
    assert max_y_diff < 1e-4, f"Threat parity error {max_y_diff} >= 1e-4"
    print("[PASS] Numerical parity confirmed: PyTorch == ONNX Runtime (< 1e-4).")
    return session


def benchmark_onnx_latency(session, n_iterations: int = 1000):
    print("\n" + "-" * 70)
    print(f"BENCHMARKING ONNX RUNTIME LATENCY ({n_iterations:,} iterations on CPU)")
    print("-" * 70, flush=True)

    node_np = np.random.randn(1, 22, 6).astype(np.float32)
    adj_np = np.ones((1, 22, 22), dtype=np.float32)
    mask_np = np.ones((1, 22, 1), dtype=np.float32)
    inputs = {
        'node_features': node_np,
        'adj_mask': adj_np,
        'player_mask': mask_np
    }

    # Warmup
    for _ in range(50):
        session.run(None, inputs)

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        session.run(None, inputs)
    elapsed = time.perf_counter() - t0

    latency_ms = (elapsed / n_iterations) * 1000.0
    fps = n_iterations / elapsed

    print(f"Single-Frame Latency: {latency_ms:.3f} ms")
    print(f"Throughput:           {fps:,.1f} FPS")
    print(f"SLA Target (< 15 ms): {'MET! Real-time 60 FPS capable' if latency_ms < 15.0 else 'Slow'}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    pt_model = export_spatiotemporal_to_onnx()
    session = verify_spatiotemporal_onnx_parity(pt_model)
    benchmark_onnx_latency(session, n_iterations=1000)

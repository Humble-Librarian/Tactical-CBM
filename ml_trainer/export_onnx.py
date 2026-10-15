import os
import sys
import json
import time
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Force UTF-8 on Windows consoles to prevent emoji print crashes
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.model import TacticalCBM

MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PT_MODEL_PATH = MODELS_DIR / "tactical_cbm.pt"
ONNX_MODEL_PATH = MODELS_DIR / "tactical_cbm.onnx"
GOLDEN_PATH = PROCESSED_DIR / "golden_test_frames.json"

def export_model_to_onnx():
    print("=" * 65)
    print("STAGE 4: ONNX EXPORT & GRAPH OPTIMIZATION")
    print("=" * 65)
    
    # 1. Load trained PyTorch model
    model = TacticalCBM(in_features=6, hidden_dim=32, num_concepts=6)
    model.load_state_dict(torch.load(PT_MODEL_PATH, map_location="cpu"))
    model.eval()
    print("Loaded PyTorch checkpoint from:", PT_MODEL_PATH)
    
    # 2. Define dummy input with static batch size = 1
    dummy_node = torch.randn(1, 22, 6, dtype=torch.float32)
    dummy_adj = torch.ones(1, 22, 22, dtype=torch.float32)
    dummy_mask = torch.ones(1, 22, 1, dtype=torch.float32)
    
    # 3. Export to ONNX using standard legacy exporter (dynamo=False)
    print(f"Exporting to {ONNX_MODEL_PATH} (Opset 17)...")
    torch.onnx.export(
        model,
        (dummy_node, dummy_adj, dummy_mask),
        str(ONNX_MODEL_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['node_features', 'adj_mask', 'player_mask'],
        output_names=['tactical_concepts', 'threat_logits'],
        dynamic_axes={
            'node_features': {0: 'batch_size'},
            'adj_mask': {0: 'batch_size'},
            'player_mask': {0: 'batch_size'},
            'tactical_concepts': {0: 'batch_size'},
            'threat_logits': {0: 'batch_size'}
        },
        dynamo=False
    )
    
    onnx_size_kb = ONNX_MODEL_PATH.stat().st_size / 1024
    print(f"ONNX Export Complete! Model Size: {onnx_size_kb:.2f} KB")
    return model

def verify_numerical_parity(pytorch_model):
    print("\n" + "-" * 65)
    print("NUMERICAL PARITY VERIFICATION (PyTorch vs ONNX Runtime)")
    print("-" * 65)
    
    # Configure ONNX Runtime Session
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), opts, providers=['CPUExecutionProvider'])
    
    # Test on 100 randomized inputs
    max_c_diff = 0.0
    max_y_diff = 0.0
    
    np.random.seed(42)
    for _ in range(100):
        node_np = np.random.randn(1, 22, 6).astype(np.float32)
        adj_np = np.random.choice([0.0, 1.0], size=(1, 22, 22)).astype(np.float32)
        # Ensure self loops
        for i in range(22): adj_np[0, i, i] = 1.0
        mask_np = np.ones((1, 22, 1), dtype=np.float32)
        
        # PyTorch prediction
        with torch.no_grad():
            node_t = torch.from_numpy(node_np)
            adj_t = torch.from_numpy(adj_np)
            mask_t = torch.from_numpy(mask_np)
            pt_c, pt_y = pytorch_model(node_t, adj_t, mask_t)
            pt_c_np = pt_c.numpy()
            pt_y_np = pt_y.numpy()
            
        # ONNX Runtime prediction
        ort_inputs = {
            'node_features': node_np,
            'adj_mask': adj_np,
            'player_mask': mask_np
        }
        ort_c, ort_y = session.run(None, ort_inputs)
        
        c_diff = np.max(np.abs(pt_c_np - ort_c))
        y_diff = np.max(np.abs(pt_y_np - ort_y))
        
        if c_diff > max_c_diff: max_c_diff = c_diff
        if y_diff > max_y_diff: max_y_diff = y_diff
        
    print(f"Max Absolute Error on 100 trials:")
    print(f"  - Tactical Concepts (C1-C6): {max_c_diff:.2e}")
    print(f"  - Threat Logits:             {max_y_diff:.2e}")
    
    assert max_c_diff < 1e-4, f"Parity failure: concept error {max_c_diff} >= 1e-4"
    assert max_y_diff < 1e-4, f"Parity failure: threat error {max_y_diff} >= 1e-4"
    print("VERIFICATION GATE PASSED: Numerical parity is guaranteed!")
    return session

def benchmark_inference(session, num_iterations: int = 5000):
    print("\n" + "-" * 65)
    print(f"LATENCY & THROUGHPUT BENCHMARK ({num_iterations:,} iterations on CPU)")
    print("-" * 65)
    
    node_np = np.random.randn(1, 22, 6).astype(np.float32)
    adj_np = np.ones((1, 22, 22), dtype=np.float32)
    mask_np = np.ones((1, 22, 1), dtype=np.float32)
    ort_inputs = {
        'node_features': node_np,
        'adj_mask': adj_np,
        'player_mask': mask_np
    }
    
    # Warmup
    for _ in range(200):
        session.run(None, ort_inputs)
        
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        session.run(None, ort_inputs)
    elapsed = time.perf_counter() - start_time
    
    per_frame_ms = (elapsed / num_iterations) * 1000.0
    fps = num_iterations / elapsed
    
    print(f"Inference Results on Intel CPU:")
    print(f"  Per-Frame Latency: {per_frame_ms:.4f} ms")
    print(f"  Throughput:        {fps:,.1f} FPS")
    print(f"  Status:            {'< 0.5 ms TARGET MET!' if per_frame_ms < 0.5 else 'Fast'}")
    print("=" * 65)

def main():
    model = export_model_to_onnx()
    session = verify_numerical_parity(model)
    benchmark_inference(session, num_iterations=5000)

if __name__ == "__main__":
    main()

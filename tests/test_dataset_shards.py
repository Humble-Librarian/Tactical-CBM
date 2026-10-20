"""
Test Suite: Binary Dataset Shards
Tests integrity of train.bin and val.bin on disk via PyTorch DataLoader.
"""
import sys
import torch
import numpy as np
import pytest
from pathlib import Path
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.dataset import BinaryShardDataset

PROCESSED_DIR = BASE_DIR / "data" / "processed"
TRAIN_BIN = PROCESSED_DIR / "train.bin"
VAL_BIN = PROCESSED_DIR / "val.bin"

def test_shards_exist_and_non_empty():
    assert TRAIN_BIN.exists(), "train.bin does not exist"
    assert VAL_BIN.exists(), "val.bin does not exist"
    assert TRAIN_BIN.stat().st_size > 0
    assert VAL_BIN.stat().st_size > 0

def test_train_dataset_dataloader_batching():
    """Verify train.bin loads via PyTorch DataLoader and produces correct shapes."""
    ds = BinaryShardDataset(TRAIN_BIN)
    assert len(ds) > 10000, f"Expected > 10k frames, got {len(ds)}"
    
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    batch = next(iter(loader))
    
    assert batch['node_features'].shape == (32, 22, 6)
    assert batch['adj_mask'].shape == (32, 22, 22)
    assert batch['player_mask'].shape == (32, 22, 1)
    assert batch['c_true'].shape == (32, 6)
    assert batch['y_threat'].shape == (32,)
    
    # Assert 0 NaNs in batch
    for k, v in batch.items():
        assert not torch.isnan(v).any(), f"NaN found in batch field {k}"

def test_val_dataset_concept_ranges():
    """Verify validation split concepts are all within [0.0, 1.0]."""
    ds = BinaryShardDataset(VAL_BIN)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    
    for i, batch in enumerate(loader):
        if i >= 5: break  # Sample first 5 batches
        c = batch['c_true']
        assert torch.all(c >= 0.0) and torch.all(c <= 1.0), "Concepts outside [0, 1] range"

def test_zero_copy_pointer_sharing():
    """Verify that PyTorch tensors share the exact physical memory pointer with the memmap buffer."""
    ds = BinaryShardDataset(TRAIN_BIN)
    rec = ds.data[0]
    sample = ds[0]
    
    # Assert physical pointer equality (true zero-copy)
    assert sample['node_features'].data_ptr() == rec['node_features'].ctypes.data, "node_features must share memory pointer"
    assert sample['adj_mask'].data_ptr() == rec['adj_mask'].ctypes.data, "adj_mask must share memory pointer"
    assert sample['c_true'].data_ptr() == rec['c_true'].ctypes.data, "c_true must share memory pointer"

def test_zero_copy_memory_allocation():
    """Verify that iterating through 5,000 frames produces near-zero heap memory allocations."""
    import tracemalloc
    ds = BinaryShardDataset(TRAIN_BIN)
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Iterate through 5,000 frames
    for i in range(5000):
        _ = ds[i]
        
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_allocated = sum(s.size_diff for s in stats if s.size_diff > 0)
    
    # Total heap allocation for 5,000 frames must be < 500 KB (with .copy() it was > 40 MB)
    assert total_allocated < 500 * 1024, f"Allocated {total_allocated / 1024:.2f} KB, expected < 500 KB (Zero-Copy violation)"

def test_dataset_audit_schema_and_demographics():
    """Verify Phase 7 Step 1 dataset audit outputs match schema contracts and invariants."""
    import json
    data_dir = BASE_DIR / "data"
    audit_json = data_dir / "audit_summary.json"
    dataset_md = data_dir / "DATASET_STATS.md"
    
    assert audit_json.exists(), "audit_summary.json must exist"
    assert dataset_md.exists(), "DATASET_STATS.md must exist"
    assert dataset_md.stat().st_size > 500, "DATASET_STATS.md must be populated"
    
    with open(audit_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    summary = data.get("summary", {})
    assert summary.get("total_matches") == 181, f"Expected 181 matches, got {summary.get('total_matches')}"
    assert summary.get("total_frames", 0) > 500000, f"Expected > 500k frames, got {summary.get('total_frames')}"
    assert summary.get("total_goals", 0) > 400, "Expected > 400 goals across 181 matches"
    
    comps = data.get("competitions", {})
    expected_comps = [
        "UEFA Euro 2024",
        "1. Bundesliga 2023/24 (Leverkusen & Bayern)",
        "Ligue 1 2022/23 (PSG Domination)",
        "FIFA World Cup 2022"
    ]
    for c in expected_comps:
        assert c in comps, f"Missing competition: {c}"
        assert comps[c]["matches"] > 0
        assert comps[c]["total_frames"] > 50000
        
    # Verify player visibility distribution covers realistic occlusion range
    vis = data.get("player_visibility_distribution", {})
    assert "20" in vis and vis["20"] > 10000, "Expected > 10k frames with 20 players visible"
    assert "22" in vis, "Expected 22 player count in visibility distribution"

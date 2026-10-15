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

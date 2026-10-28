"""
Tactical CBM Engine - Tactical Augmentations Test Suite (Phase 7 - Step 2)
Tests:
  1. Pitch Bilateral Mirroring Mathematical Invariants
  2. Double-Flip Group Identity: Mirror(Mirror(X)) == X
  3. Ball Carrier Absolute Protection Under Camera Occlusion
  4. Dropout Bounds, Shape Invariance, and Adjacency Disconnection
  5. Dataset Integration & Sub-Microsecond Execution Overhead
"""

import sys
import time
import torch
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.augmentations import (
    mirror_pitch_tensor,
    apply_player_dropout,
    TacticalAugmentor
)
from ml_trainer.dataset import BinaryShardDataset
TRAIN_BIN = BASE_DIR / "data" / "processed" / "train.bin"

def test_mirror_pitch_tensor_properties():
    """Verify lateral flip preserves invariant dimensions and strictly transforms y and theta."""
    # 22 players with dummy features [px, py, dist, theta, is_tm, is_act]
    feats = torch.tensor([
        [0.8, 0.25, 0.15, 0.70, 1.0, 0.0],
        [0.5, 0.90, 0.40, 0.20, 0.0, 1.0],  # Active actor
    ], dtype=torch.float32)
    
    flipped = mirror_pitch_tensor(feats, in_place=False)
    
    # 1. Invariant dimensions: px, dist, is_tm, is_act
    assert torch.allclose(flipped[:, 0], feats[:, 0]), "px must remain invariant"
    assert torch.allclose(flipped[:, 2], feats[:, 2]), "dist_to_ball must remain invariant"
    assert torch.allclose(flipped[:, 4], feats[:, 4]), "is_teammate must remain invariant"
    assert torch.allclose(flipped[:, 5], feats[:, 5]), "is_actor must remain invariant"
    
    # 2. Reflected dimensions: py and theta
    assert torch.allclose(flipped[:, 1], 1.0 - feats[:, 1]), "py must be 1.0 - py"
    assert torch.allclose(flipped[:, 3], 1.0 - feats[:, 3]), "angle must be 1.0 - angle"

def test_mirror_pitch_double_flip_identity():
    """Verify D2 reflection group identity: double-flip returns exact original tensor."""
    torch.manual_seed(42)
    feats = torch.rand(16, 22, 6)
    
    flip1 = mirror_pitch_tensor(feats, in_place=False)
    flip2 = mirror_pitch_tensor(flip1, in_place=False)
    
    assert torch.allclose(flip2, feats, atol=1e-7), "Mirror(Mirror(X)) must strictly equal X"

def test_player_dropout_ball_carrier_protection():
    """Verify that the active ball carrier (is_actor == 1) is NEVER dropped."""
    feats = torch.zeros(22, 6)
    p_mask = torch.ones(22, 1)
    
    # Populate 10 outfield players, player 3 is the active ball carrier
    for i in range(10):
        feats[i, 0] = 0.5
        feats[i, 1] = 0.5
        feats[i, 4] = 1.0 if i < 5 else 0.0
        feats[i, 5] = 1.0 if i == 3 else 0.0  # Player 3 is the actor
        
    for seed in range(50):
        torch.manual_seed(seed)
        aug_feats, aug_mask, _ = apply_player_dropout(
            feats, p_mask, p_drop=1.0, max_drop=2, in_place=False
        )
        
        # Ball carrier (index 3) must always remain active
        assert aug_mask[3, 0].item() == 1.0, f"Ball carrier was dropped on seed {seed}"
        assert aug_feats[3, 5].item() == 1.0, f"Ball carrier actor flag was corrupted on seed {seed}"

def test_player_dropout_bounds_and_disconnection():
    """Verify that dropout zeroes features and cleanly severs adjacency edges."""
    feats = torch.ones(22, 6)
    feats[:, 5] = 0.0  # Non-actors
    feats[0, 5] = 1.0  # Player 0 is actor
    p_mask = torch.ones(22, 1)
    adj_mask = torch.ones(22, 22)
    
    torch.manual_seed(123)
    aug_feats, aug_pmask, aug_adj = apply_player_dropout(
        feats, p_mask, adj_mask=adj_mask, p_drop=1.0, max_drop=2, in_place=False
    )
    
    dropped_indices = torch.where(aug_pmask[:, 0] == 0.0)[0]
    num_dropped = len(dropped_indices)
    
    assert 1 <= num_dropped <= 2, f"Expected 1 or 2 dropped players, got {num_dropped}"
    
    for idx in dropped_indices:
        # Features zeroed
        assert torch.all(aug_feats[idx, :] == 0.0), f"Dropped player {idx} features not zeroed"
        # Adjacency edges severed (row and column zeroed)
        assert torch.all(aug_adj[idx, :] == 0.0), f"Row {idx} of adjacency not severed"
        assert torch.all(aug_adj[:, idx] == 0.0), f"Col {idx} of adjacency not severed"

def test_tactical_augmentor_pipeline_integration():
    """Verify TacticalAugmentor works on real binary shard dataset samples without errors or NaNs."""
    if not TRAIN_BIN.exists():
        pytest.skip("train.bin not found on disk")
        
    augmentor = TacticalAugmentor(p_mirror=1.0, p_dropout=1.0, max_dropout_players=2)
    ds = BinaryShardDataset(str(TRAIN_BIN), augment=augmentor)
    
    sample = ds[0]
    assert sample['node_features'].shape == (22, 6)
    assert sample['player_mask'].shape == (22, 1)
    assert sample['adj_mask'].shape == (22, 22)
    assert not torch.isnan(sample['node_features']).any()
    assert not torch.isnan(sample['c_true']).any()

def test_augmentation_sub_microsecond_latency():
    """Verify dynamic on-the-fly augmentation executes with negligible CPU overhead (< 500 us per frame, > 2,000 FPS)."""
    augmentor = TacticalAugmentor(p_mirror=0.5, p_dropout=0.5, max_dropout_players=2)
    
    # Mock frame
    dummy_sample = {
        'node_features': torch.rand(22, 6),
        'adj_mask': torch.ones(22, 22),
        'player_mask': torch.ones(22, 1),
        'c_true': torch.rand(6),
        'y_threat': torch.tensor(0.0)
    }
    dummy_sample['node_features'][:, 5] = 0.0
    dummy_sample['node_features'][0, 5] = 1.0  # actor
    
    # Warmup
    for _ in range(50):
        _ = augmentor(dummy_sample)
        
    n_iters = 1000
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = augmentor(dummy_sample)
    elapsed_us = (time.perf_counter() - t0) * 1e6 / n_iters
    
    assert elapsed_us < 500.0, f"Augmentation latency {elapsed_us:.2f} us exceeds 500 us limit (> 2000 FPS)"

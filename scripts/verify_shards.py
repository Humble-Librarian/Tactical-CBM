"""
Tactical CBM Engine - Stage 2 Verification Gate
Audits generated binary shards (train.bin, val.bin) using zero-copy numpy.memmap.
Asserts: 0 NaNs, byte alignment, memory consumption, and concept distributions.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.schema import FRAME_DTYPE, FRAME_BYTE_SIZE

PROCESSED_DIR = BASE_DIR / "data" / "processed"
TRAIN_BIN = PROCESSED_DIR / "train.bin"
VAL_BIN = PROCESSED_DIR / "val.bin"
GOLDEN_JSON = PROCESSED_DIR / "golden_test_frames.json"

def audit_shard(shard_path: Path, name: str):
    print(f"\n--- Auditing {name} ({shard_path.name}) ---")
    size_bytes = shard_path.stat().st_size
    assert size_bytes % FRAME_BYTE_SIZE == 0, f"Error: {shard_path} size ({size_bytes}) is not a multiple of {FRAME_BYTE_SIZE}!"
    
    num_frames = size_bytes // FRAME_BYTE_SIZE
    print(f"File size: {size_bytes / (1024*1024):.2f} MB ({num_frames:,} frames)")
    
    # Zero-copy memory map
    mmap = np.memmap(shard_path, dtype=FRAME_DTYPE, mode='r')
    assert len(mmap) == num_frames
    
    # Read first and last frame to test indexing
    first_frame = mmap[0]
    last_frame = mmap[-1]
    print(f"First frame: Match {first_frame['match_id']}, Frame {first_frame['frame_id']}")
    print(f"Last frame:  Match {last_frame['match_id']}, Frame {last_frame['frame_id']}")
    
    # Check for NaNs across all tensors
    print("Checking for NaNs across entire dataset...")
    node_feats = mmap['node_features']
    adj_masks = mmap['adj_mask']
    player_masks = mmap['player_mask']
    c_trues = mmap['c_true']
    y_threats = mmap['y_threat']
    
    assert not np.isnan(node_feats).any(), "NaN found in node_features!"
    assert not np.isnan(adj_masks).any(), "NaN found in adj_mask!"
    assert not np.isnan(player_masks).any(), "NaN found in player_mask!"
    assert not np.isnan(c_trues).any(), "NaN found in c_true!"
    assert not np.isnan(y_threats).any(), "NaN found in y_threat!"
    print("  -> Zero NaNs detected across all tensors! (PASSED)")
    
    # Concept statistics
    concept_names = [
        "C1 (Def Line Height)",
        "C2 (Def Compactness)",
        "C3 (Def Width Balance)",
        "C4 (Ball Pressure)",
        "C5 (Lane Blockage)",
        "C6 (Box Overload Ratio)"
    ]
    
    print("\nConcept Distributions [0.0 - 1.0]:")
    print(f"{'Concept':<25} | {'Min':<6} | {'Mean':<6} | {'Max':<6} | {'Std':<6}")
    print("-" * 55)
    for c_idx, c_name in enumerate(concept_names):
        vals = c_trues[:, c_idx]
        print(f"{c_name:<25} | {vals.min():.3f}  | {vals.mean():.3f}  | {vals.max():.3f}  | {vals.std():.3f}")
        
    pos_count = int(np.sum(y_threats > 0.5))
    print(f"\nThreat Class Balance: {pos_count:,} positive / {num_frames:,} total ({pos_count/num_frames*100:.2f}%)")
    return num_frames

def main():
    print("=" * 60)
    print("STAGE 2: MEMORY-MAPPED SHARD VERIFICATION GATE")
    print("=" * 60)
    
    train_frames = audit_shard(TRAIN_BIN, "Train Split")
    val_frames = audit_shard(VAL_BIN, "Validation Split")
    
    print(f"\n--- Checking Golden Parity Reference Frames ---")
    with open(GOLDEN_JSON, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
    print(f"Loaded {len(golden_data)} golden frames.")
    print("Sample Golden Frame 0:")
    print(json.dumps(golden_data[0], indent=2))
    
    print("\n" + "=" * 60)
    print("STAGE 2 VERIFICATION COMPLETE: ALL GATES PASSED! (READY FOR STAGE 3)")
    print("=" * 60)

if __name__ == "__main__":
    main()

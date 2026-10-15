"""
Tactical CBM Engine - Binary Shard Contract Specification
Defines the memory-mapped struct for frames.bin shared between C++ and Python.
"""
import numpy as np

# Exact Dimensions
MAX_PLAYERS = 22
NODE_FEAT_DIM = 6
ADJ_DIM = 22 * 22  # 484
NUM_CONCEPTS = 6

# Fixed numpy structured dtype (Zero-copy layout)
# Byte calculation:
# match_id:    int32      = 4 bytes
# frame_id:    int32      = 4 bytes
# node_features: 22 x 6 f32 = 528 bytes
# adj_mask:      22 x 22 f32 = 1936 bytes
# player_mask:   22 f32     = 88 bytes
# c_true:        6 f32      = 24 bytes
# y_threat:      1 f32      = 4 bytes
# Total: 2,588 bytes per frame

FRAME_DTYPE = np.dtype([
    ('match_id', np.int32),
    ('frame_id', np.int32),
    ('node_features', np.float32, (MAX_PLAYERS, NODE_FEAT_DIM)),
    ('adj_mask', np.float32, (MAX_PLAYERS, MAX_PLAYERS)),
    ('player_mask', np.float32, (MAX_PLAYERS,)),
    ('c_true', np.float32, (NUM_CONCEPTS,)),
    ('y_threat', np.float32)
], align=False)

FRAME_BYTE_SIZE = FRAME_DTYPE.itemsize

if __name__ == "__main__":
    print(f"=== Binary Contract Verification ===")
    print(f"Total bytes per frame: {FRAME_BYTE_SIZE} bytes")
    print(f"Fields breakdown:")
    for name, (dt, offset) in FRAME_DTYPE.fields.items():
        print(f"  - {name:<15}: offset = {offset:<4} bytes | type = {dt}")
    print(f"Frames per 100 MB: {100 * 1024 * 1024 // FRAME_BYTE_SIZE:,} frames")

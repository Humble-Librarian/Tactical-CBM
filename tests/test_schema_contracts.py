"""
Test Suite: Schema & Binary Contract Invariants
Verifies that Python and C++ memory layouts match byte-for-byte.
"""
import sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.schema import FRAME_DTYPE, FRAME_BYTE_SIZE, MAX_PLAYERS, NODE_FEAT_DIM, NUM_CONCEPTS

def test_total_frame_size():
    """Assert total frame size is exactly 2,588 bytes."""
    assert FRAME_BYTE_SIZE == 2588, f"Expected 2588 bytes, got {FRAME_BYTE_SIZE}"
    assert FRAME_DTYPE.itemsize == 2588

def test_field_byte_offsets():
    """Assert exact byte offsets for all fields to guarantee zero-copy C++ compatibility."""
    fields = FRAME_DTYPE.fields
    assert fields['match_id'][1] == 0, "match_id offset must be 0"
    assert fields['frame_id'][1] == 4, "frame_id offset must be 4"
    assert fields['node_features'][1] == 8, "node_features offset must be 8"
    assert fields['adj_mask'][1] == 536, "adj_mask offset must be 536"
    assert fields['player_mask'][1] == 2472, "player_mask offset must be 2472"
    assert fields['c_true'][1] == 2560, "c_true offset must be 2560"
    assert fields['y_threat'][1] == 2584, "y_threat offset must be 2584"

def test_field_shapes_and_types():
    """Assert dimensions and float32/int32 data types."""
    fields = FRAME_DTYPE.fields
    assert fields['match_id'][0] == np.dtype('int32')
    assert fields['frame_id'][0] == np.dtype('int32')
    assert fields['node_features'][0] == np.dtype(('<f4', (22, 6)))
    assert fields['adj_mask'][0] == np.dtype(('<f4', (22, 22)))
    assert fields['player_mask'][0] == np.dtype(('<f4', (22,)))
    assert fields['c_true'][0] == np.dtype(('<f4', (6,)))
    assert fields['y_threat'][0] == np.dtype('float32')

def test_header_parity_check():
    """Assert C++ shard_format.hpp matches schema.py."""
    header_path = BASE_DIR / "cpp_engine" / "include" / "shard_format.hpp"
    assert header_path.exists(), "C++ header shard_format.hpp must exist"
    
    with open(header_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "sizeof(FrameRecord) == 2588" in content
    assert "offsetof(FrameRecord, node_features) == 8" in content
    assert "offsetof(FrameRecord, adj_mask) == 536" in content
    assert "offsetof(FrameRecord, player_mask) == 2472" in content
    assert "offsetof(FrameRecord, c_true) == 2560" in content
    assert "offsetof(FrameRecord, y_threat) == 2584" in content

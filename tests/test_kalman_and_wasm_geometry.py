"""
Test Suite: Bounded Kalman Tracker & C++ Geometry Algorithmic Parity
Authors: Senior Architect & Ponytail Quality Assurance
"""
import sys
import math
import numpy as np
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.kalman_tracker import TacticalTrack, TacticalKalmanTracker
from scripts.geometry import compute_c1_line_height, compute_c2_compactness

def test_kalman_velocity_clamping():
    """Assert Kalman filter enforces physical sprint ceiling <= 10.0 m/s."""
    track = TacticalTrack(track_id=1, x=50.0, y=40.0, is_teammate=False, is_keeper=False, timestamp=0.0)
    
    # Simulate unrealistic detection jump: 50m in 1 second (= 50 m/s)
    track.update(obs_x=100.0, obs_y=40.0, timestamp=1.0)
    
    # Predict forward
    track.predict(dt=1.0)
    speed = math.hypot(track.x[2], track.x[3])
    
    # Must be clamped to <= 10.0 m/s
    assert speed <= 10.0001, f"Extrapolated velocity {speed} exceeds human sprint limit (10 m/s)"

def test_kalman_pitch_boundary_clamping():
    """Assert Kalman filter clamps player positions to pitch dimensions [0, 120] x [0, 80]."""
    track = TacticalTrack(track_id=1, x=118.0, y=40.0, is_teammate=False, is_keeper=False, timestamp=0.0)
    track.x[2] = 10.0  # Sprinting forward towards goal line
    
    # Predict forward 2 seconds: 118 + 20 = 138m (out of bounds)
    track.predict(dt=2.0)
    
    assert track.x[0] <= 120.0, "X position must not exceed 120m"
    assert track.x[1] <= 80.0, "Y position must not exceed 80m"

def test_kalman_occlusion_recovery_horizon():
    """Assert occluded tracks are retained for dt <= 2.0s and purged for dt > 2.0s."""
    tracker = TacticalKalmanTracker()
    
    # Frame 1 at t = 0.0s
    frame1 = [{"location": [60.0, 40.0], "teammate": False, "keeper": False}]
    res1 = tracker.step(frame1, timestamp=0.0, possession_id=1)
    assert len(res1) == 1
    assert not res1[0][4], "Initial observation must not be extrapolated"
    
    # Frame 2 at t = 1.0s (detection occluded / out of camera frame)
    frame2 = []
    res2 = tracker.step(frame2, timestamp=1.0, possession_id=1)
    assert len(res2) == 1, "Track must persist during temporary 1.0s occlusion"
    assert res2[0][4], "Occluded track must be flagged as extrapolated"
    
    # Frame 3 at t = 3.5s (occlusion gap exceeds 2.0s horizon)
    frame3 = []
    res3 = tracker.step(frame3, timestamp=3.5, possession_id=1)
    assert len(res3) == 0, "Track must be pruned when occlusion gap exceeds horizon"

def test_kalman_optical_flow_fusion_in_scrum():
    """Assert optical flow vectors update track velocity and resolve scrum ambiguity."""
    track = TacticalTrack(track_id=1, x=105.0, y=35.0, is_teammate=False, is_keeper=False, timestamp=0.0)
    
    # In a corner kick scrum, player moves with optical flow vx=-2.0 m/s (clearing ball), vy=1.0 m/s
    track.update(obs_x=104.5, obs_y=35.2, timestamp=0.2, flow_vx=-2.0, flow_vy=1.0, in_scrum=True)
    
    # Velocity must reflect fused optical flow direction
    assert track.x[2] < 0.0, "X velocity must be negative (clearing out of box)"
    assert abs(track.x[2] - (-2.0)) < 1.0, "Velocity should fuse optical flow measurement"

def test_kalman_flow_only_update_during_total_occlusion():
    """Assert optical flow updates track when centroid position is completely occluded in a pileup."""
    track = TacticalTrack(track_id=1, x=105.0, y=35.0, is_teammate=False, is_keeper=False, timestamp=0.0)
    
    # Centroid occluded (None, None), but optical flow tracking patch detects motion vx=3.0, vy=0.0
    track.update(obs_x=None, obs_y=None, timestamp=0.2, flow_vx=3.0, flow_vy=0.0, in_scrum=True)
    
    # Track velocity updated via optical flow
    assert track.x[2] > 1.0, "Optical flow must update velocity during total position occlusion"

def test_wasm_binary_integrity():
    """Assert compiled tactical_geometry.wasm exists and has valid WebAssembly magic header."""
    wasm_path = BASE_DIR / "dashboard" / "tactical_geometry.wasm"
    assert wasm_path.exists(), "tactical_geometry.wasm must exist"
    
    with open(wasm_path, "rb") as f:
        magic = f.read(8)
    assert magic == b"\x00asm\x01\x00\x00\x00", "Must have valid WebAssembly 1.0 magic header"

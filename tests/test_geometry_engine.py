"""
Test Suite: Geometry & Tactical Concepts Math
Tests C1-C6 calculations under regular, boundary, and degenerate conditions.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.geometry import (
    compute_c1_line_height,
    compute_c2_compactness,
    compute_c3_width_balance,
    compute_c4_pressure,
    compute_c5_lane_blockage,
    compute_c6_box_overload,
    build_delaunay_adjacency
)

def test_c1_line_height_boundaries():
    """C1 must be normalized to [0, 1] and react inversely to defensive depth."""
    # Deep low block (at X=14m): 14 / 70 = 0.20
    deep_def = np.array([[14.0, 30.0], [15.0, 40.0], [14.0, 50.0]])
    c1_deep = compute_c1_line_height(deep_def)
    assert 0.19 <= c1_deep <= 0.21
    
    # High press line (at X=63m): 63 / 70 = 0.90
    high_def = np.array([[63.0, 30.0], [64.0, 40.0], [65.0, 50.0]])
    c1_high = compute_c1_line_height(high_def)
    assert 0.89 <= c1_high <= 0.91
    
    # Over-field clamp (> 70m)
    ultra_high = np.array([[85.0, 40.0]])
    assert compute_c1_line_height(ultra_high) == 1.0
    
    # Empty defenders fallback
    assert compute_c1_line_height(np.empty((0, 2))) == 0.5

def test_c2_compactness_degeneracies():
    """C2 must handle normal blocks, collinear points, and degenerate 1-2 player inputs."""
    # 4 defenders packed in tight 10m x 10m square (area = 100 m^2)
    # Compactness = 1.0 - (100 / 1500) = ~0.933
    tight_def = np.array([[40.0, 35.0], [40.0, 45.0], [50.0, 35.0], [50.0, 45.0]])
    c2_tight = compute_c2_compactness(tight_def)
    assert 0.90 <= c2_tight <= 0.96
    
    # Completely stretched defenders (area > 1500 m^2)
    stretched_def = np.array([[20.0, 10.0], [20.0, 70.0], [80.0, 10.0], [80.0, 70.0]])
    c2_stretched = compute_c2_compactness(stretched_def)
    assert c2_stretched <= 0.1
    
    # Degenerate: Exactly collinear points (ConvexHull volume=0 in 2D or raises QHull error)
    collinear_def = np.array([[30.0, 40.0], [40.0, 40.0], [50.0, 40.0]])
    c2_collinear = compute_c2_compactness(collinear_def)
    assert 0.0 <= c2_collinear <= 1.0
    
    # Degenerate: Fewer than 3 points
    assert compute_c2_compactness(np.array([[30.0, 40.0], [35.0, 45.0]])) == 0.5
    assert compute_c2_compactness(np.empty((0, 2))) == 0.5

def test_c3_width_balance():
    """C3 measures defensive balance relative to ball Y."""
    defenders = np.array([[40.0, 30.0], [40.0, 50.0]])
    
    # Ball centered at Y=40 (identical to defense mean Y=40)
    c3_centered = compute_c3_width_balance(defenders, 40.0)
    
    # Ball on far wing Y=75 (defense isolated on other side)
    c3_isolated = compute_c3_width_balance(defenders, 75.0)
    assert c3_centered > c3_isolated
    assert 0.0 <= c3_centered <= 1.0
    assert 0.0 <= c3_isolated <= 1.0

def test_c4_pressure_decay():
    """C4 must decrease smoothly as defenders get farther from the ball carrier."""
    ball = np.array([60.0, 40.0])
    
    # 1 defender right on the ball (0.5m)
    near_def = np.array([[60.5, 40.0]])
    c4_near = compute_c4_pressure(near_def, ball)
    
    # 1 defender 7m away
    far_def = np.array([[67.0, 40.0]])
    c4_far = compute_c4_pressure(far_def, ball)
    
    # 1 defender outside 8m radius (12m away)
    outside_def = np.array([[72.0, 40.0]])
    c4_outside = compute_c4_pressure(outside_def, ball)
    
    assert c4_near > c4_far
    assert c4_outside == 0.0
    assert 0.0 <= c4_near <= 1.0

def test_c5_lane_blockage_interception():
    """C5 tests ray intersection within 2.5m tolerance."""
    ball = np.array([70.0, 40.0])
    box_attacker = np.array([[105.0, 40.0]])  # Straight horizontal ray along Y=40
    
    # Defender standing directly on the passing lane (X=85, Y=40)
    blocking_def = np.array([[85.0, 40.0]])
    c5_blocked = compute_c5_lane_blockage(ball, box_attacker, blocking_def)
    assert c5_blocked == 1.0
    
    # Defender standing far away from the lane (X=85, Y=65)
    open_def = np.array([[85.0, 65.0]])
    c5_open = compute_c5_lane_blockage(ball, box_attacker, open_def)
    assert c5_open == 0.0

def test_c6_box_overload_ratio():
    """C6 evaluates attacker-to-defender ratio in the danger zone (X >= 95)."""
    # 3 attackers and 1 defender in box -> 3 / 4 = 0.75
    attackers = np.array([[98.0, 35.0], [102.0, 40.0], [100.0, 45.0]])
    defenders = np.array([[105.0, 40.0]])
    c6_overload = compute_c6_box_overload(attackers, defenders)
    assert c6_overload == 0.75
    
    # 0 in box
    c6_empty = compute_c6_box_overload(np.empty((0, 2)), np.empty((0, 2)))
    assert c6_empty == 0.0

def test_delaunay_adjacency_symmetry_and_self_loops():
    """Delaunay adjacency matrix must be symmetric with self-loops."""
    coords = np.array([
        [30.0, 20.0], [32.0, 40.0], [30.0, 60.0],
        [50.0, 25.0], [52.0, 45.0], [60.0, 35.0]
    ], dtype=np.float32)
    
    adj = build_delaunay_adjacency(coords, max_players=22)
    assert adj.shape == (22, 22)
    
    # Self-loops for all real players
    for i in range(len(coords)):
        assert adj[i, i] == 1.0
        
    # Symmetry A_ij == A_ji
    assert np.allclose(adj, adj.T)
    
    # Empty padded slots must have no connections
    for i in range(len(coords), 22):
        assert np.sum(adj[i, :]) == 0.0

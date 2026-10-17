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
    """C1 must be normalized to [0, 1] relative to defended goal at X=120 with 70m ceiling."""
    # Deep low block (defenders at X=109m, 11m from own goal): 11 / 70 = ~0.157
    deep_def = np.array([[109.0, 30.0], [108.0, 40.0], [109.0, 50.0]])
    c1_deep = compute_c1_line_height(deep_def)
    assert 0.15 <= c1_deep <= 0.17
    
    # High press line (defenders at X=71m, 49m from own goal): 49 / 70 = 0.70
    high_def = np.array([[71.0, 30.0], [70.0, 40.0], [69.5, 50.0]])
    c1_high = compute_c1_line_height(high_def)
    assert 0.69 <= c1_high <= 0.71
    
    # Over-field clamp (defenders pushed to X=40m, 80m from goal > 70m ceiling)
    ultra_high = np.array([[40.0, 40.0]])
    assert compute_c1_line_height(ultra_high) == 1.0
    
    # Empty defenders fallback
    assert compute_c1_line_height(np.empty((0, 2))) == 0.5

    # Breached line: Ball carrier penetrates behind defensive line (ball at X=105, defenders at X=71)
    # Line height must collapse from 0.70 to (120 - 105) / 70 = ~0.214
    c1_breached = compute_c1_line_height(high_def, ball_loc=np.array([105.0, 40.0]))
    assert 0.20 <= c1_breached <= 0.23, f"Breached line C1 ({c1_breached}) must reflect penetration"
    assert c1_breached < c1_high, "Breached line must have strictly lower protective line height than intact line"

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

def test_c2_frustum_scaling():
    """C2 frustum scaling penalizes partial observations to avoid artificial compactness."""
    # 5 defenders visible spanning 300 m^2
    defenders_5 = np.array([[40.0, 30.0], [40.0, 50.0], [55.0, 30.0], [55.0, 50.0], [47.5, 40.0]])
    c2_unscaled = compute_c2_compactness(defenders_5, apply_frustum_scaling=False)
    c2_scaled = compute_c2_compactness(defenders_5, apply_frustum_scaling=True)
    
    # Scaled area = 300 * sqrt(10 / 5) = 300 * 1.414 = 424.26 m^2 -> compactness is appropriately lower
    assert c2_scaled < c2_unscaled
    assert 0.0 <= c2_scaled <= 1.0

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
    
    # 1 defender 5m away
    far_def = np.array([[65.0, 40.0]])
    c4_far = compute_c4_pressure(far_def, ball)
    
    # 1 defender 15m away
    very_far_def = np.array([[75.0, 40.0]])
    c4_very_far = compute_c4_pressure(very_far_def, ball)
    
    assert c4_near > c4_far > c4_very_far
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
    """C6 evaluates attacker-to-defender ratio in the 18-yard penalty box (X >= 102)."""
    # 3 attackers and 1 defender in box -> 3 / 4 = 0.75
    attackers = np.array([[104.0, 35.0], [105.0, 40.0], [106.0, 45.0]])
    defenders = np.array([[108.0, 40.0]])
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

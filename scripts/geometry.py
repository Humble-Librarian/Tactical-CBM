"""
Tactical CBM Engine - Geometry & Concept Math
Computes C1-C6 tactical bottleneck concepts and adjacency graphs from player coordinates.
"""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from typing import List, Dict, Any, Tuple

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0  # Attacking goal
OWN_GOAL_X = 0.0

def compute_c1_line_height(defenders: np.ndarray) -> float:
    """
    C1: Defensive Line Height.
    Distance of the deepest outfield defender from their own goal line (X=0), divided by 70m.
    High line (>0.7), Low block (<0.3).
    """
    if len(defenders) == 0:
        return 0.5
    # Deepest defender is the one with minimum X (closest to own goal X=0)
    min_x = np.min(defenders[:, 0])
    c1 = min_x / 70.0
    return float(np.clip(c1, 0.0, 1.0))

def compute_c2_compactness(defenders: np.ndarray) -> float:
    """
    C2: Defensive Compactness.
    1.0 - (Area of Defenders' Convex Hull / 1500 m^2).
    Tight packed block (>0.8), stretched/loose block (<0.4).
    """
    if len(defenders) < 3:
        return 0.5
    try:
        # Check if points are collinear
        hull = ConvexHull(defenders[:, :2])
        area = hull.volume  # in 2D, volume is area
        c2 = 1.0 - (area / 1500.0)
        return float(np.clip(c2, 0.0, 1.0))
    except Exception:
        # Fallback if points are collinear or 2D coplanar degenerate
        return 0.5

def compute_c3_width_balance(defenders: np.ndarray, ball_y: float) -> float:
    """
    C3: Defensive Width Balance.
    Lateral spread (max Y - min Y) of defenders relative to the ball's Y channel.
    Normalized to [0, 1]. Measures if defense is over-shifted or balanced.
    """
    if len(defenders) < 2:
        return 0.5
    y_vals = defenders[:, 1]
    lateral_spread = np.max(y_vals) - np.min(y_vals)
    # Pitch width is 80m. Spread usually ranges 20m - 60m.
    # Center of defense relative to ball_y
    def_center_y = np.mean(y_vals)
    y_offset = abs(def_center_y - ball_y) / 40.0
    
    # Combined score: good balance means adequate spread (not too narrow) and shifted towards ball
    spread_norm = np.clip(lateral_spread / 60.0, 0.0, 1.0)
    c3 = 0.5 * spread_norm + 0.5 * (1.0 - np.clip(y_offset, 0.0, 1.0))
    return float(np.clip(c3, 0.0, 1.0))

def compute_c4_pressure(defenders: np.ndarray, ball_loc: np.ndarray) -> float:
    """
    C4: Ball-Carrier Pressure.
    sum(exp(-d_i / 4.0)) for all defenders within 8m of the ball carrier.
    Normalized by 3.0 (max plausible pressure in soccer).
    """
    if len(defenders) == 0:
        return 0.0
    dists = np.linalg.norm(defenders[:, :2] - ball_loc[:2], axis=1)
    close_mask = dists <= 8.0
    if not np.any(close_mask):
        return 0.0
    pressure = np.sum(np.exp(-dists[close_mask] / 4.0))
    return float(np.clip(pressure / 2.5, 0.0, 1.0))

def compute_c5_lane_blockage(ball_loc: np.ndarray, attackers: np.ndarray, defenders: np.ndarray) -> float:
    """
    C5: Central Lane Blockage.
    Fraction of direct rays from ball-carrier to attackers inside the penalty box 
    intercepted within 2.0m by a defender.
    """
    # Box attackers: X >= 102 (penalty area starts at 102m in 120m pitch), 18 <= Y <= 62
    if len(attackers) == 0 or len(defenders) == 0:
        return 1.0
        
    box_attackers = attackers[(attackers[:, 0] >= 95.0) & (attackers[:, 1] >= 18.0) & (attackers[:, 1] <= 62.0)]
    if len(box_attackers) == 0:
        # If no attackers in deep box, check forward attackers ahead of ball
        box_attackers = attackers[attackers[:, 0] > ball_loc[0]]
        if len(box_attackers) == 0:
            return 1.0  # Safe/blocked
            
    blocked_count = 0
    p1 = ball_loc[:2]
    
    for att in box_attackers:
        p2 = att[:2]
        segment_vec = p2 - p1
        seg_len_sq = np.dot(segment_vec, segment_vec)
        if seg_len_sq < 1e-6:
            continue
            
        is_blocked = False
        for d in defenders:
            p3 = d[:2]
            # Projection factor t of p3 onto line p1-p2
            t = np.dot(p3 - p1, segment_vec) / seg_len_sq
            if 0.0 <= t <= 1.0:
                projection = p1 + t * segment_vec
                dist_to_ray = np.linalg.norm(p3 - projection)
                if dist_to_ray <= 2.5:  # Intercepted within 2.5m
                    is_blocked = True
                    break
        if is_blocked:
            blocked_count += 1
            
    c5 = blocked_count / max(1, len(box_attackers))
    return float(np.clip(c5, 0.0, 1.0))

def compute_c6_box_overload(attackers: np.ndarray, defenders: np.ndarray) -> float:
    """
    C6: Box Overload Ratio.
    Ratio of attackers to defenders inside the final 25m zone (X >= 95.0).
    Higher means attacking overload (>0.5 is dangerous).
    """
    att_in_box = np.sum(attackers[:, 0] >= 95.0) if len(attackers) > 0 else 0
    def_in_box = np.sum(defenders[:, 0] >= 95.0) if len(defenders) > 0 else 0
    
    total = att_in_box + def_in_box
    if total == 0:
        return 0.0
    c6 = att_in_box / total
    return float(np.clip(c6, 0.0, 1.0))

def build_delaunay_adjacency(coords: np.ndarray, max_players: int = 22) -> np.ndarray:
    """
    Builds a 22x22 adjacency matrix using Delaunay triangulation + distance gating (<25m).
    """
    adj = np.zeros((max_players, max_players), dtype=np.float32)
    n = len(coords)
    if n < 3:
        # Fallback to proximity
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(coords[i] - coords[j]) <= 20.0:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        return adj

    try:
        tri = Delaunay(coords)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = simplex[i], simplex[j]
                    dist = np.linalg.norm(coords[u] - coords[v])
                    if dist <= 30.0:  # Gated Delaunay edge
                        adj[u, v] = 1.0
                        adj[v, u] = 1.0
    except Exception:
        # Fallback to pairwise distance <= 18m
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(coords[i] - coords[j]) <= 18.0:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

    # Self-loops for GAT attention
    for i in range(n):
        adj[i, i] = 1.0

    return adj

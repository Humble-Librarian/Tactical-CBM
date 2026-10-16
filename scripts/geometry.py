"""
Tactical CBM Engine - Professional Football Geometry & Concept Math
Engineered to elite coaching standards (Hansi Flick / Pep Guardiola pressing & offside line).

Pitch Dimensions (StatsBomb 360 standard):
- Length X: [0.0, 120.0] meters. Attacking team attacks towards X = 120.0.
- Width Y:  [0.0, 80.0] meters. Center is Y = 40.0.
- Defended Goal: Centered at X = 120.0, Y = 40.0.
- Penalty Box: X in [102.0, 120.0], Y in [18.0, 62.0].
"""
from typing import Optional
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
DEFENDED_GOAL_X = 120.0
PENALTY_BOX_X_START = 102.0

def compute_c1_line_height(outfield_defenders: np.ndarray, ball_loc: Optional[np.ndarray] = None) -> float:
    """
    C1: Defensive Line Height (Offside Trap Line).
    In modern high-pressing tactics (Hansi Flick), line height is measured by 
    the distance of the deepest outfield defender (excluding goalkeeper) 
    from their defended goal line (X = 120.0).
    
    If the ball carrier has broken behind the deepest defender (ball_x > max_x),
    the defensive line is breached and cannot offer depth protection.
    
    - Deep Low Block: Defending deep inside 18-yard box (X >= 105m -> dist <= 15m) -> C1 ~ 0.20-0.30
    - Mid Block:      Defending around 35m from goal (X ~ 85m -> dist ~ 35m)       -> C1 ~ 0.60
    - High Line:      Offside line pushed to midfield (X ~ 65m -> dist ~ 55m)      -> C1 ~ 0.90-1.00
    """
    if len(outfield_defenders) == 0:
        return 0.5
    # The deepest defender closest to defended goal (X=120) has maximum X
    max_x = float(np.max(outfield_defenders[:, 0]))
    if ball_loc is not None and len(ball_loc) >= 1:
        effective_x = max(max_x, float(ball_loc[0]))
    else:
        effective_x = max_x
        
    distance_from_goal = DEFENDED_GOAL_X - effective_x
    
    # High line ceiling is 55m from goal (near midfield line X=60-65m)
    c1 = distance_from_goal / 55.0
    return float(np.clip(c1, 0.0, 1.0))

def compute_c2_compactness(outfield_defenders: np.ndarray, apply_frustum_scaling: bool = False) -> float:
    """
    C2: Defensive Compactness (Block Volume).
    Measures how compressed the outfield block is via Convex Hull area.
    Elite compact blocks compress pitch volume to < 600 m^2.
    Stretched, broken blocks span > 1400 m^2.
    
    If apply_frustum_scaling=True (e.g. broadcast 360 tracking feeds with partial camera frustum),
    scales the area by sqrt(10 / max(N_vis, 4)) to prevent camera zoom-ins from artificially
    inflating compactness scores.
    """
    if len(outfield_defenders) < 3:
        return 0.5
    try:
        hull = ConvexHull(outfield_defenders[:, :2])
        area = hull.volume  # 2D area
        
        if apply_frustum_scaling:
            n_vis = len(outfield_defenders)
            if n_vis < 10:
                scale_factor = math.sqrt(10.0 / max(n_vis, 4))
                area = area * scale_factor
                
        # 300m^2 is ultra-compact (1.0), 1400m^2 is completely stretched (0.0)
        c2 = 1.0 - (area / 1200.0)
        return float(np.clip(c2, 0.0, 1.0))
    except Exception:
        # Fallback for collinear defenders
        return 0.5

def compute_c3_width_balance(outfield_defenders: np.ndarray, ball_y: float) -> float:
    """
    C3: Defensive Width Balance & Ball-Oriented Shifting.
    Measures if the defensive block properly shifts laterally toward the ball corridor
    while preserving adequate team spread.
    """
    if len(outfield_defenders) < 2:
        return 0.5
    y_vals = outfield_defenders[:, 1]
    lateral_spread = np.max(y_vals) - np.min(y_vals)
    
    # Defensive centroid offset from ball's Y
    centroid_y = np.mean(y_vals)
    y_offset = abs(centroid_y - ball_y) / 35.0  # Normalized offset
    
    # Ideal lateral spread is 25m - 50m
    spread_score = np.clip(lateral_spread / 45.0, 0.0, 1.0)
    shift_score = 1.0 - np.clip(y_offset, 0.0, 1.0)
    
    c3 = 0.5 * spread_score + 0.5 * shift_score
    return float(np.clip(c3, 0.0, 1.0))

def compute_c4_pressure(defenders: np.ndarray, ball_loc: np.ndarray) -> float:
    """
    C4: Ball-Carrier Pressure (Gegenpressing Intensity).
    Uses exponential distance decay with a sharp drop-off:
    - Defender at 1.5m: ~0.55 pressure
    - Defender at 3.0m: ~0.30 pressure
    - Defender at 6.0m: ~0.09 pressure
    - Defenders > 8m have 0 pressure.
    Sum of pressures is normalized so a coordinated double-team equals 1.0.
    """
    if len(defenders) == 0:
        return 0.0
    dists = np.linalg.norm(defenders[:, :2] - ball_loc[:2], axis=1)
    close_mask = dists <= 8.0
    if not np.any(close_mask):
        return 0.0
    raw_pressure = np.sum(np.exp(-dists[close_mask] / 2.5))
    return float(np.clip(raw_pressure / 1.5, 0.0, 1.0))

def compute_c5_lane_blockage(ball_loc: np.ndarray, attackers: np.ndarray, defenders: np.ndarray) -> float:
    """
    C5: Central Lane Blockage (Forward Passing Ray Interception).
    Evaluates whether defenders intercept direct ground passing lanes from the ball carrier 
    to forward attackers ahead of the ball.
    """
    if len(attackers) == 0 or len(defenders) == 0:
        return 1.0
        
    # Attackers ahead of the ball (towards defended goal X=120)
    forward_attackers = attackers[attackers[:, 0] > (ball_loc[0] + 1.0)]
    if len(forward_attackers) == 0:
        # If no forward attackers, check all teammates
        forward_attackers = attackers
        if len(forward_attackers) == 0:
            return 1.0
            
    p1 = ball_loc[:2]
    blocked_count = 0
    
    for att in forward_attackers:
        p2 = att[:2]
        segment_vec = p2 - p1
        seg_len_sq = float(np.dot(segment_vec, segment_vec))
        if seg_len_sq < 1.0:
            continue
            
        is_blocked = False
        for d in defenders:
            p3 = d[:2]
            t = float(np.dot(p3 - p1, segment_vec) / seg_len_sq)
            if 0.05 <= t <= 0.95:  # Defender between ball and receiver
                proj = p1 + t * segment_vec
                dist_to_ray = float(np.linalg.norm(p3 - proj))
                if dist_to_ray <= 2.2:  # Passing lane cut within 2.2m
                    is_blocked = True
                    break
        if is_blocked:
            blocked_count += 1
            
    c5 = blocked_count / max(1, len(forward_attackers))
    return float(np.clip(c5, 0.0, 1.0))

def compute_c6_box_overload(attackers: np.ndarray, defenders: np.ndarray) -> float:
    """
    C6: Box Overload Ratio (Penalty Area Superiority).
    Evaluates attacking vs defensive headcount in the 18-yard penalty box 
    (X >= 102.0, 18.0 <= Y <= 62.0).
    Higher means attacking dominance (e.g. 3 vs 1 = 0.75).
    """
    def in_box(coords):
        if len(coords) == 0:
            return 0
        x, y = coords[:, 0], coords[:, 1]
        mask = (x >= 102.0) & (y >= 18.0) & (y <= 62.0)
        return int(np.sum(mask))
        
    att_in_box = in_box(attackers)
    def_in_box = in_box(defenders)
    
    total = att_in_box + def_in_box
    if total == 0:
        return 0.0
    c6 = att_in_box / total
    return float(np.clip(c6, 0.0, 1.0))

def build_delaunay_adjacency(coords: np.ndarray, max_players: int = 22) -> np.ndarray:
    """
    Constructs a 22x22 adjacency matrix using Delaunay triangulation + distance gating (<= 28m).
    Includes self-loops for Graph Attention Networks.
    """
    adj = np.zeros((max_players, max_players), dtype=np.float32)
    n = len(coords)
    if n < 3:
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(coords[i] - coords[j]) <= 25.0:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        for i in range(n): adj[i, i] = 1.0
        return adj

    try:
        tri = Delaunay(coords)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = simplex[i], simplex[j]
                    dist = np.linalg.norm(coords[u] - coords[v])
                    if dist <= 28.0:
                        adj[u, v] = 1.0
                        adj[v, u] = 1.0
    except Exception:
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(coords[i] - coords[j]) <= 20.0:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

    for i in range(n):
        adj[i, i] = 1.0

    return adj

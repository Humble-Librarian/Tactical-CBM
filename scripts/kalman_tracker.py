"""
Tactical CBM Engine - Bounded Kinematic Kalman Tracker (Stage 2 Component)
Authors: Senior Architect & Ponytail

Provides temporal tracking smoothing and occlusion recovery for camera-frustum broadcast data.
Implements bounded covariance, velocity clamping (max human sprint <= 10 m/s),
and strict 2.0s horizon expiration to eliminate phantom player hallucinations.
"""
import math
import numpy as np
from typing import List, Dict, Optional, Tuple

class TacticalTrack:
    """Represents a single player track with bounded 2D kinematic Kalman filtering."""
    def __init__(self, track_id: int, x: float, y: float, is_teammate: bool, is_keeper: bool, timestamp: float):
        self.track_id = track_id
        self.is_teammate = is_teammate
        self.is_keeper = is_keeper
        self.last_seen_time = timestamp
        self.time_since_update = 0.0
        
        # State: [px, py, vx, vy]^T
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        
        # Error covariance matrix P (4x4)
        self.P = np.diag([1.0, 1.0, 9.0, 9.0]).astype(np.float32)
        
        # Measurement noise covariance R (2x2) - ~0.5m camera detection jitter
        self.R = np.diag([0.25, 0.25]).astype(np.float32)
        
        # Acceleration spectral density q (m^2/s^3)
        self.q = 4.0

    def predict(self, dt: float):
        """Predicts state forward in time with physical acceleration bounding."""
        if dt <= 0.0:
            return
            
        self.time_since_update += dt
        
        # State transition matrix F
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Discrete process noise Q
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        Q = self.q * np.array([
            [dt4, 0.0, dt3, 0.0],
            [0.0, dt4, 0.0, dt3],
            [dt3, 0.0, dt2, 0.0],
            [0.0, dt3, 0.0, dt2]
        ], dtype=np.float32)
        
        # State and covariance propagation
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        
        # Physical Velocity Clamping (Max human sprint speed = 10.0 m/s)
        speed = math.hypot(self.x[2], self.x[3])
        if speed > 10.0:
            scale = 10.0 / speed
            self.x[2] *= scale
            self.x[3] *= scale
            
        # Pitch Boundary Clamping ([0, 120] x [0, 80])
        self.x[0] = np.clip(self.x[0], 0.0, 120.0)
        self.x[1] = np.clip(self.x[1], 0.0, 80.0)

    def update(self, obs_x: float, obs_y: float, timestamp: float):
        """Standard Kalman measurement update from observed detection."""
        z = np.array([obs_x, obs_y], dtype=np.float32)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ H) @ self.P
        
        self.last_seen_time = timestamp
        self.time_since_update = 0.0

    @property
    def is_valid(self) -> bool:
        """Track remains valid if unobserved horizon <= 2.0s and covariance trace <= 25.0."""
        return (self.time_since_update <= 2.0) and (np.trace(self.P[:2, :2]) <= 25.0)


class TacticalKalmanTracker:
    """
    Multi-player tracking manager with temporal smoothing and frustum occlusion recovery.
    """
    def __init__(self, max_association_dist: float = 6.0):
        self.tracks: List[TacticalTrack] = []
        self.next_id = 1
        self.last_timestamp: Optional[float] = None
        self.last_possession: Optional[int] = None
        self.max_association_dist = max_association_dist

    def reset(self):
        """Resets tracker state on turnover or match boundary."""
        self.tracks = []
        self.last_timestamp = None
        self.last_possession = None

    def step(
        self,
        detections: List[Dict],
        timestamp: float,
        possession_id: Optional[int] = None
    ) -> List[Tuple[float, float, bool, bool, bool]]:
        """
        Processes a freeze frame:
        1. Predicts existing tracks forward to current timestamp.
        2. Associates observations using greedy nearest-neighbor gating.
        3. Extrapolates occluded tracks up to 2.0s.
        
        Returns:
            List of (x, y, is_teammate, is_actor, is_extrapolated)
        """
        # Turnover or long jump reset
        if self.last_possession is not None and possession_id is not None and possession_id != self.last_possession:
            self.reset()
            
        dt = (timestamp - self.last_timestamp) if (self.last_timestamp is not None and timestamp >= self.last_timestamp) else 0.0
        if dt > 3.0:
            self.reset()
            dt = 0.0
            
        self.last_timestamp = timestamp
        self.last_possession = possession_id

        # 1. Predict all active tracks
        for t in self.tracks:
            t.predict(dt)

        # 2. Greedy association
        unmatched_obs = list(range(len(detections)))
        matched_tracks = set()

        for obs_idx in list(unmatched_obs):
            obs = detections[obs_idx]
            loc = obs.get("location")
            if not loc or len(loc) < 2:
                unmatched_obs.remove(obs_idx)
                continue
                
            ox, oy = float(loc[0]), float(loc[1])
            is_tm = bool(obs.get("teammate", False))
            
            best_track = None
            best_dist = self.max_association_dist
            
            for t in self.tracks:
                if t.track_id in matched_tracks or t.is_teammate != is_tm:
                    continue
                d = math.hypot(t.x[0] - ox, t.x[1] - oy)
                if d < best_dist:
                    best_dist = d
                    best_track = t
                    
            if best_track is not None:
                best_track.update(ox, oy, timestamp)
                matched_tracks.add(best_track.track_id)
                unmatched_obs.remove(obs_idx)

        # 3. Initialize new tracks for unmatched observations
        for obs_idx in unmatched_obs:
            obs = detections[obs_idx]
            loc = obs.get("location")
            ox, oy = float(loc[0]), float(loc[1])
            new_track = TacticalTrack(
                track_id=self.next_id,
                x=ox,
                y=oy,
                is_teammate=bool(obs.get("teammate", False)),
                is_keeper=bool(obs.get("keeper", False)),
                timestamp=timestamp
            )
            self.next_id += 1
            self.tracks.append(new_track)

        # 4. Prune dead tracks
        self.tracks = [t for t in self.tracks if t.is_valid]

        # 5. Emit synchronized 22-player tracking output
        results = []
        for t in self.tracks:
            is_extrapolated = (t.time_since_update > 0.05)
            # Find if track corresponds to the actor
            results.append((float(t.x[0]), float(t.x[1]), t.is_teammate, t.is_keeper, is_extrapolated))
            
        return results

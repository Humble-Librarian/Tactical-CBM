"""
Tactical CBM Engine - Pitch Control & Temporal Lookahead Test Suite (Wave 5: Pathway A)
Authors: Senior Architect & Principal Computational Sports Physicist

Enforces Real-World Football Kinematics & Domain Invariants:
  1. Sprinting Fullback Recovery vs. Stationary Striker (Velocity dominance over distance)
  2. Sweeper-Keeper 18-Yard Box Handling Sphere vs. Outside Foot Sweep
  3. Biomechanical Rotational Turning Inertia on Flanks (tau_turn ~ theta^2)
  4. Pitch Control Surface Mathematical Invariants (Boundedness [0, 1], Team Completeness = 1.0)
  5. Pitch Bilateral Y-Axis Symmetry Equivariance
  6. Trajectory Extrapolation with Neuromuscular / Friction Damping
  7. Dynamic Offside Wave Front & Step-Up Trap Projection
  8. Passing Window of Opportunity [t_open, t_closed]
  9. Strict Zero-Player-Names Invariant Across all Pitch Control Definitions
"""

import sys
import math
import pytest
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.pitch_control import (
    calculate_player_tti,
    compute_pitch_control_surface,
    extrapolate_player_trajectories,
    compute_dynamic_offside_wave,
    compute_passing_window_of_opportunity,
    ROLE_KINEMATICS
)


class TestKinematicTimeToIntrecept:
    """Tests the 4-phase ballistic kinematics of player motion."""

    def test_sprinting_fullback_recovery_beats_stationary_striker(self):
        """
        Velocity Dominance over Proximity:
        A stationary forward is at X=90m, Y=34m (5m from target X=95m, Y=34m).
        A sprinting recovery fullback is at X=80m, Y=34m (15m from target, 3x farther!),
        sprinting at vx = +9.2 m/s.
        In static Voronoi, forward wins 100%.
        Under true TTI kinematics, the sprinting fullback reaches the ball first!
        """
        target = np.array([95.0, 34.0])

        # Stationary forward (CF): 5m away, standing still (v = 0)
        cf_pos = np.array([90.0, 34.0])
        cf_vel = np.array([0.0, 0.0])
        tau_striker = calculate_player_tti(cf_pos, cf_vel, target, role_tag="CF")

        # Sprinting fullback (FB): 15m away, moving at 9.2 m/s forward
        fb_pos = np.array([80.0, 34.0])
        fb_vel = np.array([9.2, 0.0])
        tau_fullback = calculate_player_tti(fb_pos, fb_vel, target, role_tag="FB")

        # Striker must react (0.60s) and accelerate from 0: tau ~ 2.0s
        # Fullback has momentum: tau ~ 1.6s
        assert tau_fullback < tau_striker, (
            f"Sprinting fullback ({tau_fullback:.2f}s) must beat stationary striker ({tau_striker:.2f}s)"
        )
        assert tau_striker - tau_fullback > 0.25, "Fullback should arrive with clear margin"

    def test_sweeper_keeper_handling_sphere_intercept(self):
        """
        Goalkeeper Reach Mechanics:
        Inside the 18-yard box, the Sweeper-Keeper has a 2.5m spherical handling reach.
        Outside the box, handling collapses to standard 0.8m foot reach.
        """
        # Target at X=105m, Y=34m (inside penalty box: X in [102, 120], Y in [18, 62])
        target_in_box = np.array([105.0, 34.0])
        gk_pos = np.array([112.0, 34.0])  # In box
        gk_vel = np.array([-4.0, 0.0])    # Charging out

        tau_gk_box = calculate_player_tti(gk_pos, gk_vel, target_in_box, role_tag="GK", is_goalkeeper=True)

        # Target at X=95m, Y=34m (outside penalty box: foot sweep only)
        target_out_box = np.array([95.0, 34.0])
        tau_gk_out = calculate_player_tti(gk_pos, gk_vel, target_out_box, role_tag="GK", is_goalkeeper=True)

        # Inside box handling reach reduces effective travel distance by 2.5m
        assert tau_gk_box < 1.8, f"GK should intercept in-box target quickly, got {tau_gk_box:.2f}s"
        assert tau_gk_out > tau_gk_box, "Outside-box foot sweep requires longer distance"

    def test_rotational_inertia_turn_penalty_on_flank(self):
        """
        Mechanical Turning Penalty:
        A player caught sprinting away from the target (theta = 180 deg)
        must dissipate momentum and pivot body mass, paying full tau_turn penalty.
        """
        target = np.array([85.0, 20.0])
        pos = np.array([75.0, 20.0])  # 10m to target

        # Scenario A: Sprinting directly toward target (theta = 0 deg)
        vel_towards = np.array([7.0, 0.0])
        tau_towards = calculate_player_tti(pos, vel_towards, target, role_tag="FB")

        # Scenario B: Sprinting in the exact opposite direction (theta = 180 deg)
        vel_opposite = np.array([-7.0, 0.0])
        tau_opposite = calculate_player_tti(pos, vel_opposite, target, role_tag="FB")

        # Turning around from a full backward sprint requires substantially longer
        assert tau_opposite > tau_towards + 0.50, (
            f"180-degree pivot ({tau_opposite:.2f}s) must severely penalize time compared to forward sprint ({tau_towards:.2f}s)"
        )


class TestPitchControlSurfaceAndInvariants:
    """Tests the 2D pitch control surface and mathematical invariants."""

    def test_pitch_control_surface_boundaries_and_invariants(self):
        """
        Physical Invariants:
          1. Boundedness: P_att and P_def in [0.0, 1.0] everywhere.
          2. Team Completeness: P_att(x, y) + P_def(x, y) == 1.0 everywhere.
          3. Certainty in [0.0, 1.0].
        """
        attackers = [
            {"x": 75.0, "y": 34.0, "vx": 3.0, "vy": 0.0, "role": "CF"},
            {"x": 65.0, "y": 20.0, "vx": 5.0, "vy": 1.0, "role": "W"},
        ]
        defenders = [
            {"x": 80.0, "y": 34.0, "vx": -1.0, "vy": 0.0, "role": "CB"},
            {"x": 70.0, "y": 20.0, "vx": -2.0, "vy": 0.0, "role": "FB"},
            {"x": 115.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "role": "GK"},
        ]

        surface = compute_pitch_control_surface(defenders, attackers, grid_shape=(30, 20))
        p_att = surface["p_att"]
        p_def = surface["p_def"]
        certainty = surface["certainty"]

        assert np.all(p_att >= 0.0) and np.all(p_att <= 1.0), "P_att must be bounded in [0, 1]"
        assert np.all(p_def >= 0.0) and np.all(p_def <= 1.0), "P_def must be bounded in [0, 1]"
        assert np.allclose(p_att + p_def, 1.0, atol=1e-5), "P_att + P_def must strictly sum to 1.0"
        assert np.all(certainty >= 0.0) and np.all(certainty <= 1.0), "Certainty must be in [0, 1]"

    def test_pitch_bilateral_y_axis_symmetry_pitch_control(self):
        """
        Bilateral Symmetry:
        Mirroring all player positions and Y-velocities across Y=34m
        must produce an identically mirrored pitch control surface.
        """
        pitch_w = 68.0
        attackers_left = [{"x": 70.0, "y": 20.0, "vx": 4.0, "vy": 1.0, "role": "W"}]
        defenders_left = [{"x": 75.0, "y": 22.0, "vx": -1.0, "vy": 0.5, "role": "FB"}]

        # Mirrored right side: y' = 68 - y, vy' = -vy
        attackers_right = [{"x": 70.0, "y": pitch_w - 20.0, "vx": 4.0, "vy": -1.0, "role": "W"}]
        defenders_right = [{"x": 75.0, "y": pitch_w - 22.0, "vx": -1.0, "vy": -0.5, "role": "FB"}]

        surf_left = compute_pitch_control_surface(defenders_left, attackers_left, grid_shape=(30, 20), pitch_width=pitch_w)
        surf_right = compute_pitch_control_surface(defenders_right, attackers_right, grid_shape=(30, 20), pitch_width=pitch_w)

        # Mirrored along row axis (Y-axis)
        p_att_left = surf_left["p_att"]
        p_att_right_flipped = np.flip(surf_right["p_att"], axis=0)

        assert np.allclose(p_att_left, p_att_right_flipped, atol=1e-3), (
            "Mirrored pitch configuration must yield identical mirrored control surface"
        )


class TestTemporalLookaheadAndPassingWindow:
    """Tests forward-in-time trajectory rollouts and passing windows."""

    def test_trajectory_extrapolation_friction_damping(self):
        """
        Forward trajectory extrapolation:
        Velocity decays under damping e^(-gamma * dt).
        """
        players = [{"x": 60.0, "y": 30.0, "vx": 8.0, "vy": 0.0}]
        
        # Rollout at t + 1.0s
        fut_1s = extrapolate_player_trajectories(players, delta_t=1.0, gamma_friction=0.12)[0]
        # Rollout at t + 2.0s
        fut_2s = extrapolate_player_trajectories(players, delta_t=2.0, gamma_friction=0.12)[0]

        assert fut_1s["x"] > 60.0
        assert fut_2s["x"] > fut_1s["x"]
        assert fut_1s["vx"] < 8.0, "Velocity must decay with damping"
        assert fut_2s["vx"] < fut_1s["vx"]

    def test_dynamic_offside_wave_step_up_trap(self):
        """
        Dynamic Offside Line Wave Front:
        When defenders step up (vx < 0), the offside line moves forward (lower X)
        over future horizons t + 0.5s and t + 1.0s.
        """
        defenders = [
            {"x": 75.0, "y": 20.0, "vx": -2.0, "vy": 0.0, "role": "FB"},
            {"x": 76.0, "y": 30.0, "vx": -2.5, "vy": 0.0, "role": "CB"},
            {"x": 76.0, "y": 40.0, "vx": -2.5, "vy": 0.0, "role": "CB"},
            {"x": 75.0, "y": 50.0, "vx": -2.0, "vy": 0.0, "role": "FB"},
            {"x": 115.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "role": "GK"}
        ]
        ball_x = 55.0

        offside_t0 = compute_dynamic_offside_wave(defenders, ball_x, delta_t=0.0)
        offside_t05 = compute_dynamic_offside_wave(defenders, ball_x, delta_t=0.5)
        offside_t10 = compute_dynamic_offside_wave(defenders, ball_x, delta_t=1.0)

        # Offside line is last outfield defender (ignoring GK at 115m)
        assert offside_t0 == pytest.approx(76.0, abs=0.1)
        assert offside_t05 < offside_t0, "Stepping up moves offside line forward (lower X)"
        assert offside_t10 < offside_t05, "Offside line continues moving forward at t+1.0s"

    def test_passing_window_of_opportunity_detection(self):
        """
        Passing Window of Opportunity:
        Evaluates a through-ball from an Advanced Playmaker (AM) to a sprinting Winger (W)
        with an open window before a defender can intercept.
        """
        passer = {"x": 65.0, "y": 34.0, "vx": 1.0, "vy": 0.0, "role": "AM"}
        # Winger sprinting into open space
        receiver = {"x": 82.0, "y": 18.0, "vx": 7.0, "vy": 0.0, "role": "W"}
        # Covering defender positioned far inside
        defenders = [
            {"x": 85.0, "y": 32.0, "vx": 0.0, "vy": -2.0, "role": "CB"},
            {"x": 115.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "role": "GK"}
        ]

        window = compute_passing_window_of_opportunity(passer, receiver, defenders, ball_speed=18.0)
        
        assert window["is_viable"] is True
        assert window["window_duration"] >= 0.25
        assert window["t_open"] <= window["t_closed"]


class TestZeroPlayerNamesEnforcement:
    """Strict architectural constraint: Zero celebrity player names anywhere in the module."""

    def test_strict_zero_player_names_in_pitch_control(self):
        """Verifies that scripts/pitch_control.py contains zero hardcoded player names."""
        forbidden_names = [
            "messi", "ronaldo", "haaland", "mbappe", "debruyne", "kane",
            "salah", "neymar", "modric", "lewandowski", "kroos", "rodri",
            "walker", "neuer", "bellingham", "vinicius", "van dijk", "alisson"
        ]

        script_path = BASE_DIR / "scripts" / "pitch_control.py"
        assert script_path.exists()

        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
            for name in forbidden_names:
                assert name not in content, (
                    f"Found forbidden player name '{name}' in scripts/pitch_control.py! "
                    f"Must strictly adhere to Position & Role Taxonomy."
                )

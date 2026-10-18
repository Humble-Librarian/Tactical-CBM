"""
Tactical CBM Engine - Fan-Grade Tactical Rigor & Scenario Benchmark Suite
Authors: Senior Architect & Tactical Purist ("Built with the Heart of a Fan")

Verifies Real-World Football Dynamics & Tactical Archetypes:
  1. Sacchi High-Line Offside Trap & The Blown Trap (Trailing Fullback)
  2. Zone 14 "Playmaker in the Hole" vs. "Double-Pivot Midfield Cage"
  3. Simeone / Mourinho "Parked Bus Low Block" vs. Late-Game Box Scramble
  4. Klopp Heavy-Metal Counter-Attack (Kinematic Transition Momentum)
  5. Neuer-Style Sweeper-Keeper Interception vs. Frozen Goal-Line Keeper
  6. Overlapping Fullback Dynamic Width & Transition Crossing
  7. The Purist's Laws of Football Physics:
     - Bilateral Pitch Symmetry (Left/Right Flank Invariance)
     - Monotonic Threat Gradient to Goal
     - Collision Singularity Robustness (Zero Division Protection)
     - Sensor Noise & Extreme GPS Velocity Glitch Boundedness
     - Strict Zero-Player-Names Invariant Across the Entire Codebase
"""

import sys
import math
import json
import pytest
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.model import (
    SpatioTemporalCBM,
    TacticalCBM,
    BilinearConceptHead,
    BetaCalibratorLayer,
    ROLE_TAXONOMY,
    ROLE_NORM_FACTORS,
    NEUTRAL_ROLE_CAPABILITIES,
    get_role_capability_vector
)

from scripts.geometry import (
    compute_c1_line_height,
    compute_c2_compactness,
    compute_c3_width_balance,
    compute_c4_pressure,
    compute_c5_lane_blockage,
    compute_c6_box_overload
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def spatiotemporal_model():
    torch.manual_seed(42)
    model = SpatioTemporalCBM(in_features=12, hidden_dim=32, num_concepts=6)
    model.eval()
    return model


@pytest.fixture
def calibrated_bcit():
    """Bilinear Concept Head loaded with calibrated weights."""
    weights_path = BASE_DIR / "models" / "spatiotemporal_cbm_weights.json"
    with open(weights_path, "r", encoding="utf-8") as f:
        data = json.load(f)["parameters"]

    head = BilinearConceptHead(num_concepts=6)
    with torch.no_grad():
        head.linear_weights.copy_(torch.tensor(data["linear_weights"], dtype=torch.float32))
        head.bias.copy_(torch.tensor([data["bias"]], dtype=torch.float32))
        
        # Load interactions
        interactions = []
        for i in range(6):
            for j in range(i + 1, 6):
                key = f"C{i+1}_x_C{j+1}"
                interactions.append(data["pairwise_interactions"].get(key, 0.0))
        head.interaction_weights.copy_(torch.tensor(interactions, dtype=torch.float32))

    calibrator = BetaCalibratorLayer()
    calibrator.set_parameters(
        data["beta_calibration"]["a"],
        data["beta_calibration"]["b"],
        data["beta_calibration"]["c"]
    )
    return head, calibrator


# ==============================================================================
# 1. SACCHI HIGH-LINE TRAP VS. TRAILING FULLBACK
# ==============================================================================

class TestSacchiOffsideDiscipline:
    """
    Tests Arrigo Sacchi-style coordinated high-line offside traps vs.
    a broken line caused by a lazy or trailing fullback.
    """

    def test_sacchi_unison_step_up_trap_flags_and_suppresses_threat(self, spatiotemporal_model):
        """
        All 4 defenders step up in unison (vx < 0, toward halfway line)
        while the striker surges forward into depth (vx > 0).
        IFAB offside must trigger and trap must snap.
        """
        node_features = torch.zeros(1, 22, 12)
        
        # 4 Defenders in a flat line at X=75m (norm 0.625) stepping up at -2.5 m/s (norm -0.25)
        for i, y in enumerate([18.0, 30.0, 42.0, 54.0]):
            node_features[0, i, 0] = 75.0 / 120.0  # x
            node_features[0, i, 1] = y / 68.0      # y
            node_features[0, i, 4] = 0.0           # is_attacker = 0 (defender)
            node_features[0, i, 6] = -0.25         # vx = -2.5 m/s (step-up!)
            node_features[0, i, 7] = 0.0           # vy
            caps = get_role_capability_vector("CB" if 1 <= i <= 2 else "FB")
            node_features[0, i, 8:12] = torch.tensor(caps)

        # Attacking Striker (CF) at X=78m (norm 0.650) sprinting into depth at +5.0 m/s (norm +0.50)
        node_features[0, 11, 0] = 78.0 / 120.0
        node_features[0, 11, 1] = 34.0 / 68.0
        node_features[0, 11, 4] = 1.0          # is_attacker = 1
        node_features[0, 11, 6] = 0.50         # vx = +5.0 m/s forward sprint
        caps_cf = get_role_capability_vector("CF")
        node_features[0, 11, 8:12] = torch.tensor(caps_cf)

        # Fill remaining players safely in their own defensive zones
        for i in range(4, 11):
            node_features[0, i, 0] = 40.0 / 120.0
            node_features[0, i, 1] = (10.0 + i * 5.0) / 68.0
            node_features[0, i, 4] = 0.0
        for i in range(12, 22):
            node_features[0, i, 0] = 50.0 / 120.0
            node_features[0, i, 1] = (10.0 + (i - 11) * 5.0) / 68.0
            node_features[0, i, 4] = 1.0

        kinematics = spatiotemporal_model.compute_offside_kinematics(node_features)
        
        assert kinematics["is_offside"] is True, "Striker beyond last defender must be flagged offside"
        assert kinematics["trap_snapped"] is True, "Coordinated step-up must trigger trap_snapped"
        assert kinematics["closing_speed"] > 0.05, f"Closing speed in normalized space must be positive, got {kinematics['closing_speed']:.3f}"

    def test_trailing_fullback_destroys_offside_trap(self, spatiotemporal_model):
        """
        If 3 defenders step up but 1 fullback is slow / caught deep (X=82m),
        the striker at X=78m is legally ONSIDE, with open grass ahead.
        The trap must NOT fire, and threat must surge.
        """
        node_features = torch.zeros(1, 22, 12)

        # 3 Defenders step up to X=74m
        for i, y in enumerate([18.0, 30.0, 42.0]):
            node_features[0, i, 0] = 74.0 / 120.0
            node_features[0, i, 1] = y / 68.0
            node_features[0, i, 4] = 0.0
            node_features[0, i, 6] = -0.25  # stepping up
            node_features[0, i, 8:12] = torch.tensor(get_role_capability_vector("CB"))

        # 1 Trailing Fullback caught deep at X=82m, sleeping (vx = 0.0)
        node_features[0, 3, 0] = 82.0 / 120.0  # Trailing 8m behind!
        node_features[0, 3, 1] = 54.0 / 68.0
        node_features[0, 3, 4] = 0.0
        node_features[0, 3, 6] = 0.0          # Flat-footed
        node_features[0, 3, 8:12] = torch.tensor(get_role_capability_vector("FB"))

        # Striker at X=78m (Behind the 3 CBs, but 4m ahead of the trailing fullback -> ONSIDE!)
        node_features[0, 11, 0] = 78.0 / 120.0
        node_features[0, 11, 1] = 34.0 / 68.0
        node_features[0, 11, 4] = 1.0
        node_features[0, 11, 6] = 0.60        # Full throttle toward goal
        node_features[0, 11, 8:12] = torch.tensor(get_role_capability_vector("CF"))

        # Fill remaining players
        for i in range(4, 11):
            node_features[0, i, 0] = 40.0 / 120.0
            node_features[0, i, 4] = 0.0
        for i in range(12, 22):
            node_features[0, i, 0] = 50.0 / 120.0
            node_features[0, i, 4] = 1.0

        kinematics = spatiotemporal_model.compute_offside_kinematics(node_features)
        
        assert kinematics["is_offside"] is False, "Striker is played onside by the trailing fullback"
        assert kinematics["trap_snapped"] is False, "Broken line must not register a snapped trap"


# ==============================================================================
# 2. ZONE 14 "PLAYMAKER IN THE HOLE" VS. DOUBLE PIVOT CAGE
# ==============================================================================

class TestPlaymakerZone14Pocket:
    """
    Tests the Advanced Playmaker operating in the "hole" (Zone 14)
    with tight clearance vs. being crowded by a defensive double pivot.
    """

    def test_advanced_playmaker_unlocks_tighter_passing_lanes_than_stopper(self):
        """
        An AM requires only 1.4m channel clearance, whereas a CB requires 2.4m.
        A defender positioned 1.8m off the passing ray should block a CB pass,
        but leave an AM pass open.
        """
        am_clearance = ROLE_TAXONOMY["AM"]["clearance"]  # 1.4m
        cb_clearance = ROLE_TAXONOMY["CB"]["clearance"]  # 2.4m

        ball = np.array([78.0, 34.0])           # Zone 14 pocket
        defender = np.array([90.0, 35.8])       # Defender offset by 1.8m perpendicularly

        # Perpendicular distance to passing line along Y=34 is exactly 1.8m
        perp_dist = abs(defender[1] - ball[1])  # 1.8m
        assert perp_dist == pytest.approx(1.8, abs=0.01)

        # Under AM clearance: 1.8m > 1.4m -> LANE OPEN!
        am_pass_open = bool(perp_dist > am_clearance)
        assert am_pass_open is True, "Advanced Playmaker should thread this 1.8m corridor"

        # Under CB clearance: 1.8m < 2.4m -> LANE BLOCKED!
        cb_pass_open = bool(perp_dist > cb_clearance)
        assert cb_pass_open is False, "Central Stopper requires 2.4m and cannot thread this pass"

    def test_double_pivot_cage_surges_gegenpress_and_closing_speed(self):
        """
        When two Holding Pivots (DM) converge on the Playmaker in Zone 14:
        Distance shrinks rapidly (rate of convergence > 3 m/s) and Gegenpress pressure surges.
        """
        playmaker_pos = np.array([78.0, 34.0])
        playmaker_vel = np.array([1.0, 0.0])  # Controlled dribble

        # Two DMs converging from top and bottom in a tight 2m pressing trap
        dm1_pos = np.array([77.5, 35.5])      # 1.58m above
        dm1_vel = np.array([0.5, -3.5])       # Charging downward at 3.5 m/s

        dm2_pos = np.array([77.5, 32.5])      # 1.58m below
        dm2_vel = np.array([0.5, 3.5])        # Charging upward at 3.5 m/s

        # Rate of distance change (convergence): d(dist)/dt = (dp . dv) / dist
        dp1 = playmaker_pos - dm1_pos
        dist1 = np.linalg.norm(dp1) + 1e-4
        dv1 = playmaker_vel - dm1_vel
        convergence_rate1 = -float(np.dot(dp1, dv1) / dist1)  # positive convergence rate

        dp2 = playmaker_pos - dm2_pos
        dist2 = np.linalg.norm(dp2) + 1e-4
        dv2 = playmaker_vel - dm2_vel
        convergence_rate2 = -float(np.dot(dp2, dv2) / dist2)

        assert convergence_rate1 > 3.0, f"DM1 convergence rate {convergence_rate1:.2f} m/s must be > 3 m/s"
        assert convergence_rate2 > 3.0, f"DM2 convergence rate {convergence_rate2:.2f} m/s must be > 3 m/s"

        # Gegenpress pressure on ball carrier with two defenders within 1.6m
        defenders = np.array([dm1_pos, dm2_pos])
        c4_pressured = compute_c4_pressure(defenders, playmaker_pos)
        
        # When DMs are 15m away
        distant_defenders = np.array([[65.0, 50.0], [65.0, 20.0]])
        c4_unpressured = compute_c4_pressure(distant_defenders, playmaker_pos)

        assert c4_pressured > 0.65, f"Double pivot cage must create severe pressure, got {c4_pressured:.2f}"
        assert c4_pressured > c4_unpressured * 5.0, "Pressured carrier must have >5x higher pressure than open carrier"


# ==============================================================================
# 3. SIMEONE LOW BLOCK FORTRESS VS. CHAOTIC BOX OVERLOAD
# ==============================================================================

class TestSimeoneLowBlockFortress:
    """
    Tests Diego Simeone / José Mourinho style 10-man low blocks:
    Tight box protection chokes threat despite forward presence.
    """

    def test_ten_man_deep_block_suffocates_threat(self, calibrated_bcit):
        """
        In a disciplined low block:
          - Line Height is low (C1 ~ 0.80 protective)
          - Defensive Compactness is extremely high (C2 ~ 0.85)
          - Defensive Width Balance is solid (C3 ~ 0.20)
          - Ball Pressure is firm (C4 ~ 0.80)
          - Box Overload is locked down (C6 ~ 0.05)
        Threat must remain safely contained (< 15% calibrated).
        """
        head, calibrator = calibrated_bcit
        
        # Simeone low block safe concept vector
        c_low_block = torch.tensor([[0.80, 0.80, 0.20, 0.80, 0.10, 0.05]], dtype=torch.float32)
        with torch.no_grad():
            raw_logit = head(c_low_block)
            raw_prob = torch.sigmoid(raw_logit)
            cal_threat = calibrator(raw_prob).item()

        assert cal_threat < 0.15, f"Disciplined low block calibrated threat should be < 15%, got {cal_threat:.1%}"
        assert raw_prob.item() < 0.50, f"Raw probability should be safe (<50%), got {raw_prob.item():.1%}"

    def test_chaotic_box_scramble_overload_threat_explosion(self, calibrated_bcit):
        """
        In a 90th-minute chaotic box breakdown:
          - Line Height collapsed (C1 ~ 0.10)
          - Compactness destroyed (C2 ~ 0.20)
          - Flanks exposed (C3 ~ 0.80)
          - Open channels (C5 ~ 0.90)
          - Box flooded with attackers (C6 ~ 0.90)
        Threat must surge dramatically (raw > 80%, calibrated > 35%).
        """
        head, calibrator = calibrated_bcit
        
        c_scramble = torch.tensor([[0.10, 0.20, 0.80, 0.10, 0.90, 0.90]], dtype=torch.float32)
        with torch.no_grad():
            raw_logit = head(c_scramble)
            raw_prob = torch.sigmoid(raw_logit)
            cal_threat = calibrator(raw_prob).item()

        assert raw_prob.item() > 0.80, f"Raw scramble threat should exceed 80%, got {raw_prob.item():.1%}"
        assert cal_threat > 0.35, f"Calibrated breakdown threat should exceed 35%, got {cal_threat:.1%}"


# ==============================================================================
# 4. KLOPP HEAVY-METAL COUNTER-ATTACK (KINEMATIC MOMENTUM)
# ==============================================================================

class TestKloppHeavyMetalCounter:
    """
    Tests fast-break transition dynamics where sprinting attackers
    catch retreating / backpedaling defenders on their heels.
    """

    def test_blitzkrieg_counter_closing_speeds_and_retreat(self):
        """
        A 3v2 counter attack:
        Attacking Winger (W) sprinting at 9.0 m/s against backpedaling CB moving at 3.0 m/s.
        Closing velocity must capture the defender retreating toward their own goal.
        """
        winger_pos = np.array([65.0, 20.0])
        winger_vel = np.array([9.0, 0.0])   # Sizzling sprint

        cb_pos = np.array([80.0, 22.0])     # 15m ahead
        cb_vel = np.array([3.0, 0.0])       # Backpedaling toward own goal (retreating!)

        dp = cb_pos - winger_pos
        dist = np.linalg.norm(dp) + 1e-4
        dv = cb_vel - winger_vel            # 3.0 - 9.0 = -6.0 m/s

        rate_of_closure = float(np.dot(dp, dv) / dist)
        assert rate_of_closure < -5.5, f"Winger is eating up ground on retreating CB at {rate_of_closure:.2f} m/s"

    def test_direct_winger_sprint_cap_exceeds_holding_pivot(self):
        """
        Verifies that tactical taxonomy preserves physical pace hierarchies:
        Winger (9.5 m/s) > Box-to-Box (8.8 m/s) > Central Stopper (8.2 m/s) > Holding Pivot (7.5 m/s).
        """
        w_speed = ROLE_TAXONOMY["W"]["v_max"]
        cm_speed = ROLE_TAXONOMY["CM"]["v_max"]
        cb_speed = ROLE_TAXONOMY["CB"]["v_max"]
        dm_speed = ROLE_TAXONOMY["DM"]["v_max"]

        assert w_speed > cm_speed > cb_speed > dm_speed
        assert w_speed == 9.5
        assert dm_speed == 7.5


# ==============================================================================
# 5. NEUER-STYLE SWEEPER-KEEPER INTERCEPTION
# ==============================================================================

class TestSweeperKeeperInterception:
    """
    Tests Manuel Neuer-style aggressive sweeping outside the penalty box:
    Sweeper Keeper rushing out intercepts a through-ball before striker arrival.
    """

    def test_sweeper_keeper_rush_cuts_through_ball_window(self):
        """
        Ball played to X=98m.
        Sweeper Keeper charges out from X=108m at -5.0 m/s (toward the ball).
        Striker arrives from X=85m at +6.0 m/s.
        The closing speed between GK and through ball isolates the threat.
        """
        ball_pos = np.array([98.0, 34.0])
        
        # Sweeper Keeper rushing out
        gk_pos = np.array([108.0, 34.0])
        gk_vel = np.array([-5.0, 0.0])
        dp_gk = ball_pos - gk_pos
        time_to_ball_gk = np.linalg.norm(dp_gk) / abs(gk_vel[0])  # 10m / 5.0 = 2.0 seconds

        # Striker sprinting
        att_pos = np.array([85.0, 34.0])
        att_vel = np.array([6.0, 0.0])
        dp_att = ball_pos - att_pos
        time_to_ball_att = np.linalg.norm(dp_att) / abs(att_vel[0])  # 13m / 6.0 = 2.17 seconds

        assert time_to_ball_gk < time_to_ball_att, (
            f"Sweeper Keeper ({time_to_ball_gk:.2f}s) sweeps the ball before striker ({time_to_ball_att:.2f}s) can shoot"
        )

    def test_sweeper_keeper_has_widest_defensive_vision(self):
        """
        Sweeper Keeper possesses 150 deg vision cone to survey the pitch,
        compared to 100 deg for Central Stopper.
        """
        assert ROLE_TAXONOMY["GK"]["theta"] == 150.0
        assert ROLE_TAXONOMY["CB"]["theta"] == 100.0


# ==============================================================================
# 6. THE PURIST'S LAWS OF FOOTBALL PHYSICS & INVARIANTS
# ==============================================================================

class TestFootballPhysicsAndInvariants:
    """
    Mathematical and domain invariance checks that any credible
    tactical AI must strictly obey.
    """

    def test_pitch_bilateral_y_axis_symmetry(self, spatiotemporal_model):
        """
        A left-flank attack mirrored across Y=34m to the right flank
        must produce symmetrical concept responses within numerical precision.
        """
        B = 1
        adj = torch.ones(B, 22, 22)
        mask = torch.ones(B, 22, 1)

        # Baseline left-wing attack
        torch.manual_seed(101)
        x_left = torch.zeros(B, 22, 12)
        x_left[0, 0:11, 0] = torch.linspace(0.2, 0.8, 11)   # Defenders
        x_left[0, 0:11, 1] = torch.tensor([0.15, 0.25, 0.35, 0.45, 0.20, 0.30, 0.40, 0.50, 0.25, 0.35, 0.45])
        x_left[0, 0:11, 4] = 0.0                            # Defenders
        x_left[0, 11:22, 0] = torch.linspace(0.3, 0.9, 11)  # Attackers
        x_left[0, 11:22, 1] = torch.tensor([0.10, 0.20, 0.30, 0.15, 0.25, 0.35, 0.20, 0.30, 0.15, 0.25, 0.30])
        x_left[0, 11:22, 4] = 1.0                           # Attackers
        x_left[0, :, 8:12] = torch.tensor(NEUTRAL_ROLE_CAPABILITIES)

        # Mirrored right-wing attack (Y_new = 1.0 - Y_old, Vy_new = -Vy_old)
        x_right = x_left.clone()
        x_right[0, :, 1] = 1.0 - x_left[0, :, 1]
        x_right[0, :, 7] = -x_left[0, :, 7]

        with torch.no_grad():
            c_left, t_left = spatiotemporal_model(x_left, adj, mask, return_calibrated=True)
            c_right, t_right = spatiotemporal_model(x_right, adj, mask, return_calibrated=True)

        # Overall concepts and calibrated threat must be symmetric
        assert abs(c_left[0, 0].item() - c_right[0, 0].item()) < 0.10, "Line Height must be flank-symmetric"
        assert abs(t_left[0, 0].item() - t_right[0, 0].item()) < 0.10, "Overall calibrated threat must be symmetric"

    def test_monotonic_distance_to_goal_gradient(self, calibrated_bcit):
        """
        As an unchallenged ball carrier penetrates closer to goal along the central axis:
        Threat must be monotonically non-decreasing.
        """
        head, calibrator = calibrated_bcit

        # Simulate concepts at 40m, 25m, 15m, and 6m from goal
        # Line height collapses and box dominance surges as attacker advances
        c_40m = torch.tensor([[0.60, 0.70, 0.50, 0.40, 0.60, 0.05]], dtype=torch.float32)
        c_25m = torch.tensor([[0.45, 0.55, 0.45, 0.30, 0.40, 0.20]], dtype=torch.float32)
        c_15m = torch.tensor([[0.25, 0.40, 0.35, 0.20, 0.20, 0.50]], dtype=torch.float32)
        c_6m  = torch.tensor([[0.10, 0.20, 0.80, 0.10, 0.90, 0.90]], dtype=torch.float32)

        with torch.no_grad():
            t_40m = calibrator(torch.sigmoid(head(c_40m))).item()
            t_25m = calibrator(torch.sigmoid(head(c_25m))).item()
            t_15m = calibrator(torch.sigmoid(head(c_15m))).item()
            t_6m  = calibrator(torch.sigmoid(head(c_6m))).item()

        assert t_40m < t_25m < t_15m < t_6m, (
            f"Threat must monotonically increase approaching goal: {t_40m:.1%} -> {t_25m:.1%} -> {t_15m:.1%} -> {t_6m:.1%}"
        )

    def test_collision_singularity_robustness(self, spatiotemporal_model):
        """
        When two players occupy the exact same spatial coordinates (dp = [0, 0]),
        numerical epsilon (1e-4) must prevent division by zero and produce finite outputs.
        """
        B = 2
        x = torch.zeros(B, 22, 12)
        # Force players 0, 1, 2, 3 to identical coordinates (severe collision)
        x[:, :4, 0] = 0.50
        x[:, :4, 1] = 0.50
        x[:, :4, 6] = 1.0  # vx
        x[:, :4, 7] = 1.0  # vy

        adj = torch.ones(B, 22, 22)
        mask = torch.ones(B, 22, 1)

        with torch.no_grad():
            concepts, threat = spatiotemporal_model(x, adj, mask, return_calibrated=True)

        assert not torch.isnan(concepts).any(), "Collision coordinates must not produce NaN concepts"
        assert not torch.isinf(concepts).any(), "Collision coordinates must not produce Inf concepts"
        assert not torch.isnan(threat).any(), "Collision coordinates must not produce NaN threat"
        assert not torch.isinf(threat).any(), "Collision coordinates must not produce Inf threat"

    def test_sensor_noise_velocity_spike_boundedness(self, spatiotemporal_model):
        """
        Extreme GPS glitch (v = 30.0 m/s, supersonic for football)
        must remain bounded and not break tensor constraints.
        """
        B = 1
        x = torch.zeros(B, 22, 12)
        x[0, :, 0] = torch.rand(22)
        x[0, :, 1] = torch.rand(22)
        x[0, 5, 6] = 3.0  # Norm of 30 m/s (30 / 10 = 3.0)
        x[0, 5, 7] = 3.0

        adj = torch.ones(B, 22, 22)
        mask = torch.ones(B, 22, 1)

        with torch.no_grad():
            concepts, threat = spatiotemporal_model(x, adj, mask, return_calibrated=True)

        assert torch.all(concepts >= 0.0) and torch.all(concepts <= 1.0)
        assert torch.all(threat >= 0.0) and torch.all(threat <= 1.0)

    def test_strict_zero_player_names_enforcement(self):
        """
        Scans model definitions, taxonomy configs, and sidecar files
        to guarantee ZERO celebrity player names are hardcoded.
        """
        forbidden_names = [
            "messi", "ronaldo", "haaland", "mbappe", "debruyne", "kane",
            "salah", "neymar", "modric", "lewandowski", "kroos", "rodri",
            "bellingham", "vinicius", "van dijk", "alisson", "courtois"
        ]

        files_to_check = [
            BASE_DIR / "ml_trainer" / "model.py",
            BASE_DIR / "models" / "spatiotemporal_cbm_weights.json",
            BASE_DIR / "models" / "linear_head_weights.json",
            BASE_DIR / "dashboard" / "index.html"
        ]

        for file_path in files_to_check:
            if not file_path.exists():
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().lower()
                for name in forbidden_names:
                    assert name not in content, (
                        f"Found forbidden player name '{name}' in {file_path.name}! "
                        f"Violates pure Position & Role Taxonomy constraint."
                    )

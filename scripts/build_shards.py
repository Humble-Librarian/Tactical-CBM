"""
Tactical CBM Engine - Shard Builder (Stage 2)
Streams raw StatsBomb 360 JSONs, computes C1-C6 geometric concepts,
and outputs memory-mapped binary shards (train.bin, val.bin) + golden_test_frames.json.
"""
import os
import sys
import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.schema import FRAME_DTYPE, FRAME_BYTE_SIZE, MAX_PLAYERS, NODE_FEAT_DIM
from scripts.geometry import (
    compute_c1_line_height,
    compute_c2_compactness,
    compute_c3_width_balance,
    compute_c4_pressure,
    compute_c5_lane_blockage,
    compute_c6_box_overload,
    build_delaunay_adjacency
)

BASE_DIR = Path(__file__).resolve().parent.parent
EVENTS_DIR = BASE_DIR / "data" / "raw" / "events"
THREESIXTY_DIR = BASE_DIR / "data" / "raw" / "three-sixty"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def process_match(match_id: int) -> List[np.ndarray]:
    """
    Processes one match and returns an array of structured numpy records (2,588 bytes each).
    """
    ev_path = EVENTS_DIR / f"{match_id}.json"
    ts_path = THREESIXTY_DIR / f"{match_id}.json"
    
    if not ev_path.exists() or not ts_path.exists():
        return []
        
    with open(ev_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    with open(ts_path, "r", encoding="utf-8") as f:
        frames_360 = json.load(f)
        
    event_idx_map = {ev["id"]: i for i, ev in enumerate(events)}
    
    # Precompute shot events
    n_events = len(events)
    is_shot = np.zeros(n_events, dtype=bool)
    event_team_ids = np.zeros(n_events, dtype=np.int64)
    
    for i, ev in enumerate(events):
        event_team_ids[i] = ev.get("team", {}).get("id", -1)
        if ev.get("type", {}).get("name") == "Shot":
            is_shot[i] = True
            
    records = []
    
    for frame_id, f in enumerate(frames_360):
        ev_uuid = f.get("event_uuid")
        if ev_uuid not in event_idx_map:
            continue
            
        ev_idx = event_idx_map[ev_uuid]
        current_team = event_team_ids[ev_idx]
        
        # Calculate Y label: Shot occurs within next 10 actions by attacking team
        window_end = min(ev_idx + 10, n_events)
        team_shots = is_shot[ev_idx:window_end] & (event_team_ids[ev_idx:window_end] == current_team)
        y_threat = 1.0 if np.any(team_shots) else 0.0
        
        freeze_frame = f.get("freeze_frame", [])
        if len(freeze_frame) < 8:  # Camera frustum filter
            continue
            
        # Parse players
        teammate_locs = []
        opponent_locs = []
        outfield_opponent_locs = []
        players_data = []
        actor_loc = None
        
        for p in freeze_frame:
            loc = p.get("location")
            if not loc or len(loc) < 2:
                continue
            x, y = float(loc[0]), float(loc[1])
            is_teammate = bool(p.get("teammate", False))
            is_actor = bool(p.get("actor", False))
            is_keeper = bool(p.get("keeper", False))
            
            players_data.append((x, y, is_teammate, is_actor))
            if is_teammate:
                teammate_locs.append([x, y])
            else:
                opponent_locs.append([x, y])
                if not is_keeper:
                    outfield_opponent_locs.append([x, y])
                
            if is_actor:
                actor_loc = [x, y]
                
        # Must have at least 3 opponents to evaluate defensive geometry
        if len(opponent_locs) < 3:
            continue
            
        # Fallback for ball carrier / actor location
        if actor_loc is None:
            if len(teammate_locs) > 0:
                actor_loc = teammate_locs[0]
            else:
                actor_loc = [60.0, 40.0]
                
        opponents_np = np.array(opponent_locs, dtype=np.float32)
        outfield_opponents_np = np.array(outfield_opponent_locs, dtype=np.float32) if len(outfield_opponent_locs) >= 2 else opponents_np
        teammates_np = np.array(teammate_locs, dtype=np.float32) if len(teammate_locs) > 0 else np.empty((0, 2), dtype=np.float32)
        actor_loc_np = np.array(actor_loc, dtype=np.float32)
        
        # Compute the 6 Bottleneck Concepts (C1 - C6) with exact tactical physics
        c1 = compute_c1_line_height(outfield_opponents_np)
        c2 = compute_c2_compactness(outfield_opponents_np)
        c3 = compute_c3_width_balance(outfield_opponents_np, actor_loc_np[1])
        c4 = compute_c4_pressure(opponents_np, actor_loc_np)
        c5 = compute_c5_lane_blockage(actor_loc_np, teammates_np, opponents_np)
        c6 = compute_c6_box_overload(teammates_np, opponents_np)
        c_true = np.array([c1, c2, c3, c4, c5, c6], dtype=np.float32)
        
        # Build 22-slot node features
        node_features = np.zeros((MAX_PLAYERS, NODE_FEAT_DIM), dtype=np.float32)
        player_mask = np.zeros(MAX_PLAYERS, dtype=np.float32)
        player_coords = []
        
        num_to_take = min(len(players_data), MAX_PLAYERS)
        for i in range(num_to_take):
            px, py, is_tm, is_act = players_data[i]
            player_coords.append([px, py])
            
            dist_to_ball = math.hypot(px - actor_loc[0], py - actor_loc[1])
            dx_goal = 120.0 - px
            dy_goal = 40.0 - py
            angle_to_goal = math.atan2(dy_goal, dx_goal) / math.pi
            
            node_features[i, 0] = px / 120.0
            node_features[i, 1] = py / 80.0
            node_features[i, 2] = min(dist_to_ball / 120.0, 1.0)
            node_features[i, 3] = (angle_to_goal + 1.0) / 2.0  # Normalized [0, 1]
            node_features[i, 4] = 1.0 if is_tm else 0.0
            node_features[i, 5] = 1.0 if is_act else 0.0
            player_mask[i] = 1.0
            
        adj_mask = build_delaunay_adjacency(np.array(player_coords, dtype=np.float32), MAX_PLAYERS)
        
        # Fill structured record
        rec = np.zeros(1, dtype=FRAME_DTYPE)[0]
        rec['match_id'] = match_id
        rec['frame_id'] = frame_id
        rec['node_features'] = node_features
        rec['adj_mask'] = adj_mask
        rec['player_mask'] = player_mask
        rec['c_true'] = c_true
        rec['y_threat'] = y_threat
        
        records.append(rec)
        
    return records

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    event_files = sorted(list(EVENTS_DIR.glob("*.json")))
    match_ids = [int(f.stem) for f in event_files]
    
    print(f"Total available matches: {len(match_ids)}")
    if len(match_ids) < 2:
        print("Error: Need at least 2 matches for train/val split!")
        return
        
    # Match-level split: 80% train, 20% validation
    split_idx = max(1, int(len(match_ids) * 0.8))
    train_matches = match_ids[:split_idx]
    val_matches = match_ids[split_idx:]
    
    print(f"Train matches ({len(train_matches)}): {train_matches}")
    print(f"Val matches   ({len(val_matches)}): {val_matches}")
    
    train_bin_path = PROCESSED_DIR / "train.bin"
    val_bin_path = PROCESSED_DIR / "val.bin"
    
    # Remove existing shards if present
    if train_bin_path.exists(): train_bin_path.unlink()
    if val_bin_path.exists(): val_bin_path.unlink()
    
    golden_frames = []
    
    def process_and_write(m_list, out_path, is_train=True):
        total_records = 0
        total_threats = 0
        with open(out_path, "wb") as f_out:
            for mid in m_list:
                recs = process_match(mid)
                if not recs:
                    continue
                arr = np.array(recs, dtype=FRAME_DTYPE)
                f_out.write(arr.tobytes())
                
                count = len(recs)
                threats = sum(1 for r in recs if r['y_threat'] > 0.5)
                total_records += count
                total_threats += threats
                print(f"  Match {mid}: {count:,} frames processed ({threats} threat frames, {threats/count*100:.1f}%)")
                
                if is_train and len(golden_frames) < 10 and len(recs) >= 10:
                    for i in range(10):
                        r = recs[i]
                        golden_frames.append({
                            "match_id": int(r['match_id']),
                            "frame_id": int(r['frame_id']),
                            "c_true": [float(x) for x in r['c_true']],
                            "y_threat": float(r['y_threat']),
                            "num_players": int(np.sum(r['player_mask']))
                        })
                        
        return total_records, total_threats

    print("\n--- Writing train.bin ---")
    n_train, t_train = process_and_write(train_matches, train_bin_path, is_train=True)
    
    print("\n--- Writing val.bin ---")
    n_val, t_val = process_and_write(val_matches, val_bin_path, is_train=False)
    
    # Save golden test frames
    golden_path = PROCESSED_DIR / "golden_test_frames.json"
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden_frames, f, indent=2)
    print(f"\nSaved {len(golden_frames)} reference frames to {golden_path}")
    
    print("\n" + "=" * 60)
    print("STAGE 2 SHARDS SUMMARY:")
    print(f"  train.bin: {n_train:,} frames ({train_bin_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"             Threat rate: {t_train}/{n_train} ({t_train/max(1,n_train)*100:.2f}%)")
    print(f"  val.bin:   {n_val:,} frames ({val_bin_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"             Threat rate: {t_val}/{n_val} ({t_val/max(1,n_val)*100:.2f}%)")
    print(f"  Total:     {n_train + n_val:,} frames")
    print("=" * 60)

if __name__ == "__main__":
    main()

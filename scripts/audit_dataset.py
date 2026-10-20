"""
Tactical CBM Engine - Dataset Audit & Demographics Collector (Phase 7 - Step 1)
Scans all 181 modern matches and gathers:
  - Tournament-by-tournament match and frame counts
  - Event distribution (shots, open play threat events, baseline non-threat)
  - Player tracking completeness & occlusion rates (how often 22 vs <22 players appear)
  - Team demographic representation
Outputs summary data to be documented in data/DATASET_STATS.md.
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EVENTS_DIR = RAW_DIR / "events"
THREESIXTY_DIR = RAW_DIR / "three-sixty"
MATCHES_DIR = RAW_DIR / "matches"
MANIFEST_PATH = MATCHES_DIR / "modern_dataset_manifest.json"

def run_dataset_audit():
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        matches = json.load(f)

    print(f"Loaded {len(matches)} matches from manifest.")
    
    comp_stats = defaultdict(lambda: {
        "matches": 0,
        "total_frames": 0,
        "threat_frames": 0,
        "teams": set(),
        "goals": 0,
        "player_counts": Counter(),
        "keeper_present_count": 0,
    })

    overall_player_counts = Counter()
    total_frames = 0
    total_threat_frames = 0
    total_goals = 0

    print("Auditing matches across all four modern competitions...")
    for idx, m in enumerate(matches):
        mid = int(m["match_id"])
        comp_name = m.get("competition", "Unknown Competition")
        home_team = m.get("home_team", "Home")
        away_team = m.get("away_team", "Away")
        home_score = m.get("home_score", 0) or 0
        away_score = m.get("away_score", 0) or 0

        ev_file = EVENTS_DIR / f"{mid}.json"
        ts_file = THREESIXTY_DIR / f"{mid}.json"

        if not ev_file.exists() or not ts_file.exists():
            continue

        with open(ev_file, "r", encoding="utf-8") as f:
            events = json.load(f)
        with open(ts_file, "r", encoding="utf-8") as f:
            frames_360 = json.load(f)

        # Map events to check threat status (shot or goal in possession)
        event_map = {ev["id"]: ev for ev in events}
        
        # Track goals
        match_goals = home_score + away_score
        total_goals += match_goals

        stats = comp_stats[comp_name]
        stats["matches"] += 1
        stats["teams"].add(home_team)
        stats["teams"].add(away_team)
        stats["goals"] += match_goals

        m_frames = len(frames_360)
        stats["total_frames"] += m_frames
        total_frames += m_frames

        for f360 in frames_360:
            freeze = f360.get("freeze_frame", [])
            n_players = len(freeze)
            # Cap count for histogram binning
            stats["player_counts"][n_players] += 1
            overall_player_counts[n_players] += 1

            # Check if goalkeeper is visible in freeze frame
            has_keeper = any(p.get("keeper", False) for p in freeze)
            if has_keeper:
                stats["keeper_present_count"] += 1

            # Check threat status via linked event
            ev_id = f360.get("event_uuid")
            if ev_id in event_map:
                ev = event_map[ev_id]
                # If event type is Shot or leads directly to shot
                is_threat = (ev.get("type", {}).get("name") == "Shot" or 
                             ev.get("shot") is not None)
                if is_threat:
                    stats["threat_frames"] += 1
                    total_threat_frames += 1

        if (idx + 1) % 40 == 0 or (idx + 1) == len(matches):
            print(f"  Processed {idx + 1}/{len(matches)} matches ({total_frames:,} frames audited)...")

    # Serialize results to a structured JSON for reporting
    report = {
        "summary": {
            "total_matches": len(matches),
            "total_frames": total_frames,
            "total_threat_frames": total_threat_frames,
            "threat_rate_pct": round(total_threat_frames / max(1, total_frames) * 100, 2),
            "total_goals": total_goals,
            "average_goals_per_match": round(total_goals / max(1, len(matches)), 2),
            "average_frames_per_match": round(total_frames / max(1, len(matches))),
        },
        "competitions": {},
        "player_visibility_distribution": {
            str(k): v for k, v in sorted(overall_player_counts.items())
        }
    }

    for comp_name, data in comp_stats.items():
        m_count = data["matches"]
        f_count = data["total_frames"]
        t_count = data["threat_frames"]
        report["competitions"][comp_name] = {
            "matches": m_count,
            "teams_count": len(data["teams"]),
            "teams": sorted(list(data["teams"])),
            "total_frames": f_count,
            "threat_frames": t_count,
            "threat_rate_pct": round(t_count / max(1, f_count) * 100, 2),
            "goals": data["goals"],
            "goals_per_match": round(data["goals"] / max(1, m_count), 2),
            "keeper_visibility_pct": round(data["keeper_present_count"] / max(1, f_count) * 100, 2),
            "average_visible_players": round(sum(k * v for k, v in data["player_counts"].items()) / max(1, f_count), 1),
        }

    output_json = DATA_DIR / "audit_summary.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nAudit complete! Saved detailed summary to {output_json}")
    return report

if __name__ == "__main__":
    run_dataset_audit()

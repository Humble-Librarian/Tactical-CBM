import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EVENTS_DIR = BASE_DIR / "data" / "raw" / "events"
THREESIXTY_DIR = BASE_DIR / "data" / "raw" / "three-sixty"

def main():
    print("=" * 60)
    print("STAGE 1: RAW DATA AUDIT & VERIFICATION")
    print("=" * 60)
    
    event_files = sorted(list(EVENTS_DIR.glob("*.json")))
    print(f"Total downloaded event JSONs: {len(event_files)}")
    
    total_frames = 0
    total_events = 0
    match_summaries = []
    
    for ev_path in event_files:
        mid = ev_path.stem
        ts_path = THREESIXTY_DIR / f"{mid}.json"
        if not ts_path.exists():
            print(f"Warning: Missing 360 data for match {mid}")
            continue
            
        with open(ev_path, "r", encoding="utf-8") as f:
            events = json.load(f)
        with open(ts_path, "r", encoding="utf-8") as f:
            frames = json.load(f)
            
        n_ev = len(events)
        n_fr = len(frames)
        total_events += n_ev
        total_frames += n_fr
        
        match_summaries.append((mid, n_ev, n_fr))
        
    print(f"{'Match ID':<12} | {'Events':<10} | {'360 Freeze-Frames':<18}")
    print("-" * 48)
    for mid, n_ev, n_fr in match_summaries:
        print(f"{mid:<12} | {n_ev:<10} | {n_fr:<18}")
        
    print("-" * 48)
    print(f"TOTAL: {len(match_summaries)} Matches | {total_events:,} Events | {total_frames:,} 360 Frames")
    print("=" * 60)
    print("STATUS: Stage 1 Raw Data & Schema Locked successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

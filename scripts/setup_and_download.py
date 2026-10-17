"""
Tactical CBM Engine - Modern Data Ingestion Pipeline
Fetches elite modern tracking data (2022-2024) with full 360 freeze frames from StatsBomb Open Data:
  1. UEFA Euro 2024 (51 matches)
  2. 1. Bundesliga 2023/2024 (34 matches - Bayer Leverkusen Invincible Season & Bayern Munich)
  3. Ligue 1 2022/2023 (32 matches - Paris Saint-Germain domestic dominance)
  4. FIFA World Cup 2022 (64 matches - Global transitional meta)

Total: 181 matches | ~550,000 freeze frames
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import requests
from typing import List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
EVENTS_DIR = RAW_DIR / "events"
THREESIXTY_DIR = RAW_DIR / "three-sixty"
MATCHES_DIR = RAW_DIR / "matches"
MANIFEST_FILE = MATCHES_DIR / "modern_dataset_manifest.json"

TARGET_COMPETITIONS = [
    {"name": "UEFA Euro 2024", "c_id": 55, "s_id": 282, "tag": "euro_2024"},
    {"name": "1. Bundesliga 2023/24 (Leverkusen & Bayern)", "c_id": 9, "s_id": 281, "tag": "bundesliga_2023_24"},
    {"name": "Ligue 1 2022/23 (PSG Domination)", "c_id": 7, "s_id": 235, "tag": "ligue1_psg_2022_23"},
    {"name": "FIFA World Cup 2022", "c_id": 43, "s_id": 106, "tag": "worldcup_2022"},
]

def setup_directories():
    for p in [EVENTS_DIR, THREESIXTY_DIR, MATCHES_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def fetch_target_matches(comps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_matches = []
    print("\n--- Step 1: Querying Competition Match Registries ---")
    for comp in comps:
        url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{comp['c_id']}/{comp['s_id']}.json"
        req = urllib.request.Request(url, headers={"User-Agent": "Tactical-CBM-Collector/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  [ERROR] Failed to query {comp['name']}: {e}")
            continue
            
        m_360 = [m for m in data if m.get("match_status_360") == "available"]
        print(f"  {comp['name']}: {len(m_360)} matches with 360 tracking (total matches: {len(data)})")
        for m in m_360:
            all_matches.append({
                "match_id": m["match_id"],
                "competition": comp["name"],
                "tag": comp["tag"],
                "match_date": m.get("match_date"),
                "home_team": m["home_team"]["home_team_name"],
                "away_team": m["away_team"]["away_team_name"],
                "home_score": m.get("home_score"),
                "away_score": m.get("away_score"),
            })
    return all_matches

def download_file_with_retry(session: requests.Session, url: str, target_path: Path, max_retries: int = 4) -> Tuple[bool, str]:
    if target_path.exists() and target_path.stat().st_size > 1000:
        return True, "cached"
        
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                with open(target_path, "wb") as f:
                    f.write(resp.content)
                return True, "downloaded"
            elif resp.status_code == 429:
                wait_time = 2.0 * (attempt + 1)
                time.sleep(wait_time)
            elif resp.status_code == 404:
                return False, "404_not_found"
            else:
                time.sleep(1.0)
        except Exception as e:
            if attempt == max_retries - 1:
                return False, str(e)
            time.sleep(1.5)
    return False, "retries_exceeded"

def download_match_pair(session: requests.Session, m: Dict[str, Any]) -> Tuple[int, str, bool, str]:
    mid = m["match_id"]
    desc = f"{m['competition']} | {m['home_team']} vs {m['away_team']}"
    
    url_ev = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{mid}.json"
    url_360 = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{mid}.json"
    
    path_ev = EVENTS_DIR / f"{mid}.json"
    path_360 = THREESIXTY_DIR / f"{mid}.json"
    
    ok_ev, msg_ev = download_file_with_retry(session, url_ev, path_ev)
    ok_360, msg_360 = download_file_with_retry(session, url_360, path_360)
    
    return mid, desc, (ok_ev and ok_ts_pair(ok_ev, ok_360)), f"ev:{msg_ev}, 360:{msg_360}"

def ok_ts_pair(ok_ev, ok_360):
    return ok_ev and ok_360

def main():
    parser = argparse.ArgumentParser(description="StatsBomb Modern Era Tracking Collector")
    parser.add_argument("--comp", type=str, default="all", choices=["all", "euro", "bundesliga", "ligue1", "worldcup"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of matches to download")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent download worker threads")
    args = parser.parse_args()

    setup_directories()

    comp_map = {
        "euro": [TARGET_COMPETITIONS[0]],
        "bundesliga": [TARGET_COMPETITIONS[1]],
        "ligue1": [TARGET_COMPETITIONS[2]],
        "worldcup": [TARGET_COMPETITIONS[3]],
        "all": TARGET_COMPETITIONS
    }
    
    target_comps = comp_map[args.comp]
    matches = fetch_target_matches(target_comps)
    
    if args.limit:
        matches = matches[:args.limit]
        
    print(f"\nTargeting {len(matches)} matches across {len(target_comps)} competition(s).")
    
    # Save manifest
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2)
    print(f"Dataset manifest written to {MANIFEST_FILE}")
    
    print(f"\n--- Step 2: Downloading Match Pairs (Events + 360) with {args.workers} Workers ---")
    session = requests.Session()
    session.headers.update({"User-Agent": "Tactical-CBM-Collector/1.0"})
    
    t0 = time.time()
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_match_pair, session, m): m for m in matches}
        completed = 0
        total = len(futures)
        
        for fut in as_completed(futures):
            mid, desc, ok, msg = fut.result()
            completed += 1
            if ok:
                success_count += 1
                status = "OK"
            else:
                fail_count += 1
                status = "FAIL"
            
            # Print periodic progress
            if completed % 10 == 0 or completed == total or not ok:
                print(f"[{completed:3d}/{total:3d}] [{status}] Match {mid} ({desc}) -> {msg}")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"INGESTION SUMMARY:")
    print(f"  Successfully processed: {success_count}/{len(matches)} matches")
    print(f"  Failed:                 {fail_count}")
    print(f"  Total time:             {elapsed:.1f}s ({elapsed/max(1, len(matches)):.2f}s per match)")
    print("=" * 60)

if __name__ == "__main__":
    main()

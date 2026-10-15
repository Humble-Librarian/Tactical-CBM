import os
import json
import time
import requests
from pathlib import Path
from statsbombpy import sb
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

EVENTS_DIR = RAW_DIR / "events"
THREESIXTY_DIR = RAW_DIR / "three-sixty"
MATCHES_DIR = RAW_DIR / "matches"

def setup_directories():
    for p in [EVENTS_DIR, THREESIXTY_DIR, MATCHES_DIR, PROCESSED_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    print("Project directories initialized successfully.")

def download_file_if_not_exists(url, target_path):
    if target_path.exists() and target_path.stat().st_size > 0:
        return True, "already_exists"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(resp.content)
            return True, "downloaded"
        else:
            return False, f"HTTP_{resp.status_code}"
    except Exception as e:
        return False, str(e)

def download_match_pair(match_id):
    url_events = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
    url_360 = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{match_id}.json"
    
    path_events = EVENTS_DIR / f"{match_id}.json"
    path_360 = THREESIXTY_DIR / f"{match_id}.json"
    
    ok_ev, msg_ev = download_file_if_not_exists(url_events, path_events)
    ok_360, msg_360 = download_file_if_not_exists(url_360, path_360)
    
    return match_id, ok_ev and ok_360, msg_ev, msg_360

def main(max_matches=10):
    setup_directories()
    
    print("\n--- Step 1: Fetching Euro 2020 Match List ---")
    matches = sb.matches(competition_id=55, season_id=43)
    matches_file = MATCHES_DIR / "euro_2020_matches.json"
    matches.to_json(matches_file, orient="records", indent=2)
    print(f"Saved {len(matches)} matches metadata to {matches_file}")
    
    match_ids = matches["match_id"].tolist()[:max_matches]
    print(f"\n--- Step 2: Downloading {len(match_ids)} Matches (Events + 360) in parallel ---")
    
    start_time = time.time()
    success_count = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_match_pair, mid): mid for mid in match_ids}
        for future in as_completed(futures):
            mid, ok, msg_ev, msg_360 = future.result()
            if ok:
                success_count += 1
                print(f"Match {mid}: OK (Events: {msg_ev}, 360: {msg_360})")
            else:
                print(f"Match {mid}: FAILED (Events: {msg_ev}, 360: {msg_360})")
                
    elapsed = time.time() - start_time
    print(f"\nCompleted: {success_count}/{len(match_ids)} match pairs ready in {elapsed:.2f}s.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Number of matches to download (default: 10)")
    args = parser.parse_args()
    main(max_matches=args.limit)

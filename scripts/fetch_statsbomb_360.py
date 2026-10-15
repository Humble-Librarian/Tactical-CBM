import os
import json
import requests
from statsbombpy import sb

def main():
    print("=== Exploring StatsBomb Open Data Competitions ===")
    competitions = sb.competitions()
    print("Total competitions in open data:", len(competitions))
    
    # Filter for competitions that have 360 data
    # In statsbomb open data, 'match_status_360' is often tracked
    print(competitions.columns)
    
    # Look for Euro 2020 (competition 55, season 43) or 360 tagged
    euro_matches = sb.matches(competition_id=55, season_id=43)
    print(f"Euro 2020 matches found: {len(euro_matches)}")
    print("Match sample columns:", euro_matches.columns.tolist()[:10])
    
    sample_match_id = euro_matches['match_id'].iloc[0]
    print(f"Sample match ID: {sample_match_id} ({euro_matches['home_team'].iloc[0]} vs {euro_matches['away_team'].iloc[0]})")
    
    # Check 360 data for this match directly from StatsBomb open-data GitHub repository
    # URL format: https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{match_id}.json
    url_360 = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{sample_match_id}.json"
    url_events = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{sample_match_id}.json"
    
    print(f"Testing 360 data download from: {url_360}")
    resp_360 = requests.get(url_360)
    if resp_360.status_code == 200:
        frames = resp_360.json()
        print(f"SUCCESS! Retrieved 360 data with {len(frames)} freeze-frames.")
        print("Sample frame structure keys:", frames[0].keys())
        print("Number of visible players in frame 0:", len(frames[0].get('freeze_frame', [])))
        print("Sample player object:", frames[0]['freeze_frame'][0] if frames[0].get('freeze_frame') else "None")
    else:
        print(f"360 fetch status code: {resp_360.status_code}")

if __name__ == "__main__":
    main()

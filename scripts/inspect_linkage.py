import requests
import json

def inspect_match_linkage(match_id=3788758):
    print(f"Fetching events for match {match_id}...")
    url_events = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
    url_360 = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{match_id}.json"
    
    events = requests.get(url_events).json()
    frames_360 = requests.get(url_360).json()
    
    print(f"Total events: {len(events)}")
    print(f"Total 360 freeze-frames: {len(frames_360)}")
    
    # Map event_uuid to event index
    event_index_map = {ev['id']: i for i, ev in enumerate(events)}
    
    # Count how many 360 frames map to events
    matched_count = sum(1 for f in frames_360 if f['event_uuid'] in event_index_map)
    print(f"Matched 360 frames to events: {matched_count}/{len(frames_360)}")
    
    # Inspect shot events to see distribution of threat
    shots = [i for i, ev in enumerate(events) if ev.get('type', {}).get('name') == 'Shot']
    print(f"Total shots in match: {len(shots)}")
    
    # Check "shot within next 10 actions" label for each 360 frame
    shot_within_10_count = 0
    valid_frames = 0
    player_counts = []
    
    for f in frames_360:
        event_id = f['event_uuid']
        if event_id not in event_index_map:
            continue
        idx = event_index_map[event_id]
        
        # Check if any event in [idx, idx + 10] is a Shot by the attacking team
        # An event belongs to a team: ev['team']['id']
        current_team_id = events[idx].get('team', {}).get('id')
        has_shot = False
        for next_i in range(idx, min(idx + 10, len(events))):
            next_ev = events[next_i]
            if next_ev.get('type', {}).get('name') == 'Shot' and next_ev.get('team', {}).get('id') == current_team_id:
                has_shot = True
                break
        
        if has_shot:
            shot_within_10_count += 1
            
        num_players = len(f.get('freeze_frame', []))
        player_counts.append(num_players)
        valid_frames += 1
        
    print(f"Valid paired frames: {valid_frames}")
    print(f"Frames resulting in a shot within 10 actions: {shot_within_10_count} ({shot_within_10_count / valid_frames * 100:.2f}%)")
    print(f"Player count per frame - Min: {min(player_counts)}, Max: {max(player_counts)}, Avg: {sum(player_counts)/len(player_counts):.1f}")
    
    # Show a matched sample
    sample_f = frames_360[10]
    sample_ev = events[event_index_map[sample_f['event_uuid']]]
    print("\n--- Sample Paired Frame ---")
    print(f"Event Type: {sample_ev.get('type', {}).get('name')}")
    print(f"Event Team: {sample_ev.get('team', {}).get('name')}")
    print(f"Location of event: {sample_ev.get('location')}")
    print(f"Freeze frame player count: {len(sample_f.get('freeze_frame', []))}")
    print("First 3 players in freeze frame:")
    for p in sample_f.get('freeze_frame', [])[:3]:
        print("  ", p)

if __name__ == "__main__":
    inspect_match_linkage()

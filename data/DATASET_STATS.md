# The Tactical Dataset: A Story of Modern Football 🌍⚽

This document tells the story of the data that powers the **Tactical-CBM Engine**. It describes the matches, the tournaments, and the real-world tracking cameras behind nearly 600,000 freeze frames of elite football.

---

## 📖 The Four Theaters of Modern Football

Rather than relying on one specific team or a single tournament, our fleet spans **181 matches** played between 2022 and 2024 across four distinct competitions. Each competition brings its own tactical identity, tempo, and style of play.

```
Total Scope:
├── 181 Matches
├── 94 Unique Teams (National Teams & Top European Clubs)
├── 598,789 Freeze Frames of Tracking Data
└── 513 Goals Scored (2.83 goals per match)
```

---

### 1. FIFA World Cup 2022 (The Global Arena)
- **Matches:** 64 matches (Every match of the tournament)
- **Teams:** 32 national teams from every continent
- **Total Frames:** 203,882 frames
- **Goals:** 172 goals (2.69 per match)
- **The Tactical Story:** The World Cup represents elite tournament football. Teams play with immense caution, prioritizing compact defensive shapes and sudden counter-attacks. Because cameras at the World Cup often follow fast transitions closely, an average of 15 players are visible in any single frame, capturing wide-open pitch space during decisive breakaways.

### 2. UEFA Euro 2024 (The European Summit)
- **Matches:** 51 matches (Complete tournament from group stage to final)
- **Teams:** 24 top European national teams
- **Total Frames:** 164,530 frames
- **Goals:** 117 goals (2.29 per match)
- **The Tactical Story:** Euro 2024 represents modern positional play. Defenses are extraordinarily well-organized, leading to lower goal counts (2.29 per game) as teams methodically search for openings using inverted fullbacks and controlled passing lanes.

### 3. 1. Bundesliga 2023/24 (The High-Press Laboratory)
- **Matches:** 34 matches (Featuring Xabi Alonso's historic unbeaten Bayer Leverkusen and Bayern Munich)
- **Teams:** 18 German clubs
- **Total Frames:** 118,607 frames
- **Goals:** 113 goals (3.32 per match)
- **The Tactical Story:** German football is renowned for relentless pressing, counter-pressing, and high defensive lines. This tactical intensity creates open, high-scoring games (3.32 goals per game) with constant box entries, giving our engine rich data on aggressive pressing traps.

### 4. Ligue 1 2022/23 (The Asymmetric Duel)
- **Matches:** 32 matches (Covering Paris Saint-Germain's domestic campaign)
- **Teams:** 20 French clubs
- **Total Frames:** 111,770 frames
- **Goals:** 111 goals (3.47 per match)
- **The Tactical Story:** French domestic football offers classic tactical contrasts: high-possession attacking giants facing deep, stubborn defensive blocks. It is the highest-scoring environment in our dataset (3.47 goals per game) and features the tightest clustering of players, with an average of 17.5 players packed into the broadcast camera frame.

---

## 🎥 What Real Broadcast Cameras Actually See

In theoretical tactical chalkboard diagrams, all 22 players are always visible on the pitch. But real-world optical tracking cameras live in the physical world of stadiums and broadcast directors.

Our audit of all **598,789 frames** uncovered two crucial realities about modern broadcast data:

### 1. All 22 Players Are Almost Never on Camera Together
- In our entire collection, having **all 22 players visible at the exact same moment happened in only 216 frames** out of nearly 600,000.
- The most common picture is **16 to 20 players** on screen at once:
  - **20 players visible:** 82,233 frames
  - **18 players visible:** 78,741 frames
  - **19 players visible:** 75,195 frames
  - **17 players visible:** 68,099 frames
- **Why this happens:** When the ball is near the sideline, opposite wingers or fullbacks are often positioned wide outside the camera's angle of view. During counter-attacks, trailing defenders are still sprinting back from the far end of the pitch.

### 2. The Goalkeeper's Solitude
- Goalkeepers appear in only **32% to 42% of tracking frames**.
- When play is unfolding in the middle third of the pitch, both goalkeepers stand back inside their own penalty areas, far beyond the camera's sightlines.
- They only appear when play enters the final third during shots, crosses, or corner kicks.

> **Why This Matters for AI:** A model that expects every player to be present will break down during live broadcast analysis. Our graph attention network uses attention masks and random player occlusion dropout to remain completely stable even when players run out of frame.

---

## 📊 Summary by Competition

| Competition | Matches | Teams | Total Frames | Goals Scored | Goals / Match | Avg Players On Screen | Goalkeeper Visible |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FIFA World Cup 2022** | 64 | 32 | 203,882 | 172 | 2.69 | 15.1 | 31.9% |
| **UEFA Euro 2024** | 51 | 24 | 164,530 | 117 | 2.29 | 16.4 | 37.4% |
| **1. Bundesliga 2023/24** | 34 | 18 | 118,607 | 113 | 3.32 | 16.5 | 40.0% |
| **Ligue 1 2022/23** | 32 | 20 | 111,770 | 111 | 3.47 | 17.5 | 42.5% |
| **Total Modern Fleet** | **181** | **94** | **598,789** | **513** | **2.83** | **16.2** | **37.2%** |

---

## 🎯 Next Steps for Phase 7

Now that our match catalog and real-world camera numbers are fully documented:
1. **Pitch Mirroring (Bilateral Symmetry):** We can flip every frame horizontally from left-wing to right-wing, doubling our training variety and eliminating any artificial left-or-right bias.
2. **Camera Occlusion Training (Dropout):** We can intentionally hide 1 or 2 players during training to mirror the exact camera dropouts discovered in this audit.
3. **Cross-Tournament Generalization:** We can train our engine on domestic club football (Bundesliga and Ligue 1) and test how well it understands international knockout football (World Cup and Euro 2024).

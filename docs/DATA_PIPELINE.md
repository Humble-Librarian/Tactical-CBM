# Data Pipeline & Shard Specification

## 1. Overview

The data pipeline ingests raw event and tracking JSON files from StatsBomb Open Data, extracts spatial graphs and ground-truth concepts, and writes structured binary shards for high-throughput zero-copy training.

```
data/
├── raw/
│   ├── events/               # Match event sequences ({match_id}.json)
│   ├── three-sixty/          # 360 freeze frames with player coordinates
│   └── matches/
│       └── modern_dataset_manifest.json  # 181-match curated manifest
└── processed/
    ├── train.bin             # 462,382 training frames (1,141.21 MB)
    ├── val.bin               # 112,178 validation frames (276.87 MB)
    └── golden_test_frames.json # Reference frames for regression testing
```

---

## 2. Ingestion Pipeline (`scripts/setup_and_download.py`)

Downloads match pairs with multi-worker parallelism and automatic backoff retry logic:
- **Competitions:**
  1. `UEFA Euro 2024` (comp_id=55, season_id=282): 51 matches
  2. `1. Bundesliga 2023/24` (comp_id=9, season_id=281): 34 matches (Bayer Leverkusen Invincible Season)
  3. `Ligue 1 2022/23` (comp_id=7, season_id=235): 32 matches (Paris Saint-Germain)
  4. `FIFA World Cup 2022` (comp_id=43, season_id=106): 64 matches
- **Total:** 181 matches (574,560 freeze frames).

Usage:
```bash
python scripts/setup_and_download.py --comp all --workers 8
```

---

## 3. Shard Builder (`scripts/build_shards.py`)

Transforms raw JSON files into binary shards:
- **Stratified Match Splitting:** Automatically balances training (80%) and validation (20%) across competition tags to prevent cross-competition covariate shift.
- **Worker Recycling:** Uses `multiprocessing.Pool(processes=8, maxtasksperchild=4)` and explicit `gc.collect()` to guarantee strictly bounded virtual memory usage.
- **Frustum Scaling:** Applies $s = \sqrt{10 / \max(N_{\text{vis}}, 4)}$ to Convex Hull area when unobserved players fall outside the broadcast camera frustum.
- **Bounded State Cache (BSPSC):** Preserves occluded defender coordinates for $\le 2.0\text{s}$ during active possession sequences.

Usage:
```bash
python scripts/build_shards.py --workers 8
```

---

## 4. PyTorch Dataset (`ml_trainer/dataset.py`)

Implements `BinaryShardDataset` providing true $O(1)$ zero-copy random access:
- Opens binary shards using `np.memmap(bin_path, dtype=FRAME_DTYPE, mode='c')`.
- Returns direct PyTorch views via `torch.from_numpy()` sharing the exact underlying buffer memory pointers (`t.data_ptr() == rec.ctypes.data`).
- Produces zero transient heap copies during sample iteration ($< 500\text{ KB}$ heap allocation over 5,000 frames).

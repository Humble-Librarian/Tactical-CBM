"""
Tactical CBM Engine - Memory-Mapped PyTorch Dataset
Provides zero-copy random access to train.bin and val.bin.
"""
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.schema import FRAME_DTYPE, FRAME_BYTE_SIZE

class BinaryShardDataset(Dataset):
    def __init__(self, bin_path: str):
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary shard not found at: {self.bin_path}")
            
        file_size = self.bin_path.stat().st_size
        self.num_frames = file_size // FRAME_BYTE_SIZE
        
        # Zero-copy memory map
        self.data = np.memmap(self.bin_path, dtype=FRAME_DTYPE, mode='r')

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int):
        rec = self.data[idx]
        
        node_features = torch.from_numpy(rec['node_features'].copy())  # [22, 6]
        adj_mask = torch.from_numpy(rec['adj_mask'].copy())            # [22, 22]
        player_mask = torch.from_numpy(rec['player_mask'].copy()).unsqueeze(-1) # [22, 1]
        c_true = torch.from_numpy(rec['c_true'].copy())                # [6]
        y_threat = torch.tensor(rec['y_threat'], dtype=torch.float32)  # scalar
        
        return {
            'node_features': node_features,
            'adj_mask': adj_mask,
            'player_mask': player_mask,
            'c_true': c_true,
            'y_threat': y_threat
        }

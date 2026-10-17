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
    def __init__(self, bin_path: str, augment: bool = False):
        self.bin_path = Path(bin_path)
        self.augment = augment
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary shard not found at: {self.bin_path}")
            
        file_size = self.bin_path.stat().st_size
        self.num_frames = file_size // FRAME_BYTE_SIZE
        
        # Zero-copy memory map with Copy-on-Write (mode='c')
        # Ensures NumPy array flags are writeable so PyTorch creates direct views
        # without emitting UserWarning or requiring expensive .copy() heap allocations.
        self.data = np.memmap(self.bin_path, dtype=FRAME_DTYPE, mode='c')

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int):
        rec = self.data[idx]
        
        # True zero-copy: direct torch tensor views over the memory-mapped buffer
        node_features = torch.from_numpy(rec['node_features'])  # [22, 6]
        adj_mask = torch.from_numpy(rec['adj_mask'])            # [22, 22]
        player_mask = torch.from_numpy(rec['player_mask']).unsqueeze(-1) # [22, 1]
        c_true = torch.from_numpy(rec['c_true'])                # [6]
        y_threat = torch.tensor(rec['y_threat'], dtype=torch.float32)  # scalar
        
        # D2 Klein Four-Group Lateral Pitch Symmetry Augmentation (DeepMind TacticAI pattern)
        # Reflects lateral position (y_norm -> 1.0 - y_norm) and angle to goal (angle_norm -> 1.0 - angle_norm)
        if self.augment and torch.rand(1).item() < 0.5:
            node_features = node_features.clone()
            node_features[:, 1] = 1.0 - node_features[:, 1]
            node_features[:, 3] = 1.0 - node_features[:, 3]
        
        return {
            'node_features': node_features,
            'adj_mask': adj_mask,
            'player_mask': player_mask,
            'c_true': c_true,
            'y_threat': y_threat
        }

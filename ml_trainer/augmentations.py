"""
Tactical CBM Engine - Tactical Data Augmentations (Phase 7 - Step 2)
Provides high-performance, dynamic on-the-fly tactical augmentations:
  1. Bilateral Lateral Reflection (y_norm -> 1.0 - y_norm, theta_goal -> 1.0 - theta_goal)
     - Preserves all 6 geometric concepts (C1 - C6) and threat label Y under D2 group symmetry.
  2. Realistic Player Occlusion Dropout (Camera Frustum & Shadow Simulation)
     - Empirically grounded in the Step 1 finding that broadcast cameras only see 16-20 players.
     - Protects active ball carrier (is_actor == 1) from ever being dropped.
     - Zeros node features and clears player_mask cleanly disconnecting them from GAT.
"""

import torch
from typing import Dict, Any, Optional, Tuple, Union

def mirror_pitch_tensor(
    node_features: torch.Tensor,
    in_place: bool = False
) -> torch.Tensor:
    """
    Applies bilateral lateral reflection (pitch width flip) to node features.
    Handles both unbatched [N, 6] and batched [B, N, 6] tensors.
    
    Feature mapping under reflection:
      dim 0: px / 120.0 (lengthwise coordinate) -> INVARIANT
      dim 1: py / 80.0  (widthwise coordinate)  -> 1.0 - py_norm
      dim 2: dist_to_ball / 120.0               -> INVARIANT (isometric)
      dim 3: angle_to_goal_norm                 -> 1.0 - angle_norm
      dim 4: is_teammate                        -> INVARIANT
      dim 5: is_actor (ball carrier)            -> INVARIANT
    """
    feats = node_features if in_place else node_features.clone()
    # Feature 1: normalized Y coordinate
    feats[..., 1] = 1.0 - feats[..., 1]
    # Feature 3: normalized angle to goal
    feats[..., 3] = 1.0 - feats[..., 3]
    return feats

def apply_player_dropout(
    node_features: torch.Tensor,
    player_mask: torch.Tensor,
    adj_mask: Optional[torch.Tensor] = None,
    p_drop: float = 0.15,
    max_drop: int = 2,
    in_place: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Simulates realistic camera frustum cutoff and tracking occlusion.
    Randomly drops 1 to max_drop peripheral outfield players (where is_actor == 0).
    Never drops the active ball carrier (is_actor == 1).
    
    Args:
        node_features: [..., 22, 6] tensor
        player_mask: [..., 22, 1] or [..., 22] tensor
        adj_mask: optional [..., 22, 22] tensor
        p_drop: probability of applying occlusion dropout to this frame
        max_drop: maximum number of peripheral players to occlude
        in_place: whether to modify tensors in-place
        
    Returns:
        (augmented_node_features, augmented_player_mask, augmented_adj_mask)
    """
    if p_drop <= 0.0 or torch.rand(1).item() > p_drop:
        return node_features, player_mask, adj_mask

    feats = node_features if in_place else node_features.clone()
    p_mask = player_mask if in_place else player_mask.clone()
    a_mask = (adj_mask if in_place else adj_mask.clone()) if adj_mask is not None else None

    # Handle batched vs unbatched
    is_batched = feats.ndim == 3
    batch_size = feats.shape[0] if is_batched else 1

    for b in range(batch_size):
        f_slice = feats[b] if is_batched else feats
        m_slice = p_mask[b] if is_batched else p_mask

        # Eligible players: currently visible (player_mask > 0) AND not the active ball carrier (is_actor < 0.5)
        # Note: dim 5 is is_actor
        mask_1d = m_slice.squeeze(-1) if m_slice.ndim > 1 else m_slice
        actor_flag = f_slice[:, 5]
        
        eligible_indices = torch.where((mask_1d > 0.5) & (actor_flag < 0.5))[0]
        n_eligible = len(eligible_indices)
        
        # We only drop players if we have sufficient context (at least 6 visible outfield players)
        if n_eligible >= 6:
            # Pick number of players to drop: 1 or up to max_drop
            k_drop = int(torch.randint(1, max_drop + 1, (1,)).item())
            k_drop = min(k_drop, n_eligible - 5)  # Ensure at least 5 outfield players remain
            
            if k_drop > 0:
                perm = torch.randperm(n_eligible)
                drop_indices = eligible_indices[perm[:k_drop]]
                
                # Zero out node features and clear player mask
                f_slice[drop_indices, :] = 0.0
                if m_slice.ndim > 1:
                    m_slice[drop_indices, 0] = 0.0
                else:
                    m_slice[drop_indices] = 0.0
                    
                # Optionally clear adjacency edges for dropped players
                if a_mask is not None:
                    adj_slice = a_mask[b] if is_batched else a_mask
                    adj_slice[drop_indices, :] = 0.0
                    adj_slice[:, drop_indices] = 0.0

    return feats, p_mask, a_mask

class TacticalAugmentor:
    """
    Unified, composable augmentation pipeline for Tactical CBM training.
    Applies on-the-fly pitch mirroring and camera occlusion dropout with zero disk overhead.
    """
    def __init__(
        self,
        p_mirror: float = 0.5,
        p_dropout: float = 0.15,
        max_dropout_players: int = 2
    ):
        self.p_mirror = p_mirror
        self.p_dropout = p_dropout
        self.max_dropout_players = max_dropout_players

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augments a single dictionary sample from BinaryShardDataset:
          - 'node_features': [22, 6]
          - 'adj_mask': [22, 22]
          - 'player_mask': [22, 1]
          - 'c_true': [6]
          - 'y_threat': scalar
        """
        feats = sample['node_features']
        p_mask = sample['player_mask']
        a_mask = sample['adj_mask']
        
        # 1. Bilateral Pitch Mirroring (D2 symmetry)
        if self.p_mirror > 0.0 and torch.rand(1).item() < self.p_mirror:
            feats = mirror_pitch_tensor(feats, in_place=False)
            
        # 2. Camera Occlusion Dropout
        if self.p_dropout > 0.0 and torch.rand(1).item() < self.p_dropout:
            feats, p_mask, a_mask = apply_player_dropout(
                feats,
                p_mask,
                adj_mask=a_mask,
                p_drop=1.0,  # Already sampled above
                max_drop=self.max_dropout_players,
                in_place=False
            )
            
        return {
            'node_features': feats,
            'adj_mask': a_mask,
            'player_mask': p_mask,
            'c_true': sample['c_true'],
            'y_threat': sample['y_threat']
        }

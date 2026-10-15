"""
Tactical CBM Engine - Training Loop (Stage 3)
Trains Dense-GAT Concept Bottleneck Model using joint Concept MSE + Threat BCE loss.
Evaluates Concept R2 and Threat ROC-AUC, saving tactical_cbm.pt and linear_head_weights.json.
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, r2_score
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_trainer.dataset import BinaryShardDataset
from ml_trainer.model import TacticalCBM

MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CONCEPT_NAMES = [
    "C1: Defensive Line Height",
    "C2: Defensive Compactness",
    "C3: Defensive Width Balance",
    "C4: Ball-Carrier Pressure",
    "C5: Central Lane Blockage",
    "C6: Box Overload Ratio"
]

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_c_pred = []
    all_c_true = []
    all_y_pred = []
    all_y_true = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch['node_features'].to(device)
            adj = batch['adj_mask'].to(device)
            mask = batch['player_mask'].to(device)
            c_true = batch['c_true'].to(device)
            y_true = batch['y_threat'].to(device)
            
            c_pred, y_logits = model(x, adj, mask)
            y_probs = torch.sigmoid(y_logits).squeeze(-1)
            
            all_c_pred.append(c_pred.cpu().numpy())
            all_c_true.append(c_true.cpu().numpy())
            all_y_pred.append(y_probs.cpu().numpy())
            all_y_true.append(y_true.cpu().numpy())
            
    c_pred_arr = np.concatenate(all_c_pred, axis=0)
    c_true_arr = np.concatenate(all_c_true, axis=0)
    y_pred_arr = np.concatenate(all_y_pred, axis=0)
    y_true_arr = np.concatenate(all_y_true, axis=0)
    
    # Calculate R2 per concept
    r2_scores = []
    for i in range(6):
        r2 = r2_score(c_true_arr[:, i], c_pred_arr[:, i])
        r2_scores.append(float(r2))
        
    mean_r2 = float(np.mean(r2_scores))
    roc_auc = float(roc_auc_score(y_true_arr, y_pred_arr))
    
    return mean_r2, r2_scores, roc_auc

def main(epochs: int = 5, batch_size: int = 128, lr: float = 1e-3):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    train_path = PROCESSED_DIR / "train.bin"
    val_path = PROCESSED_DIR / "val.bin"
    
    train_ds = BinaryShardDataset(train_path)
    val_ds = BinaryShardDataset(val_path)
    
    print(f"Loaded datasets: {len(train_ds):,} train frames, {len(val_ds):,} val frames.")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = TacticalCBM(in_features=6, hidden_dim=32, num_concepts=6).to(device)
    
    # Losses
    criterion_concept = nn.MSELoss()
    # Threat label imbalance ~5.9% positive, pos_weight ~8.0
    pos_weight = torch.tensor([8.0]).to(device)
    criterion_threat = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\n" + "=" * 65)
    print("STAGE 3: DENSE-GAT CONCEPT BOTTLENECK TRAINING")
    print("=" * 65)
    
    best_auc = 0.0
    best_score = -999.0
    best_metrics = {}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        c_loss_sum = 0.0
        y_loss_sum = 0.0
        t0 = time.time()
        
        # Warmup phase: Epochs 1-4 focus heavily on concept convergence (X -> C)
        # Epochs 5+ jointly balance threat prediction (C -> Y)
        y_weight = 0.05 if epoch <= 4 else 0.25
        
        for batch in train_loader:
            x = batch['node_features'].to(device)
            adj = batch['adj_mask'].to(device)
            mask = batch['player_mask'].to(device)
            c_true = batch['c_true'].to(device)
            y_true = batch['y_threat'].to(device).unsqueeze(-1)
            
            optimizer.zero_grad()
            c_pred, y_logits = model(x, adj, mask)
            
            loss_c = criterion_concept(c_pred, c_true)
            loss_y = criterion_threat(y_logits, y_true)
            loss = loss_c + y_weight * loss_y
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            c_loss_sum += loss_c.item()
            y_loss_sum += loss_y.item()
            
        scheduler.step()
        elapsed = time.time() - t0
        
        # Validation
        mean_r2, r2_scores, roc_auc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
              f"Loss: {total_loss/len(train_loader):.4f} (C_MSE: {c_loss_sum/len(train_loader):.4f}, Y_BCE: {y_loss_sum/len(train_loader):.4f}) | "
              f"Val Mean R2: {mean_r2:.3f} | Val ROC-AUC: {roc_auc:.4f}")
        
        # Track best model balancing both R2 and AUC
        combined_score = roc_auc + 0.5 * mean_r2
        if combined_score > best_score:
            best_score = combined_score
            best_metrics = {
                "epoch": epoch,
                "mean_r2": mean_r2,
                "r2_scores": dict(zip(CONCEPT_NAMES, r2_scores)),
                "roc_auc": roc_auc
            }
            # Save best PyTorch checkpoint
            torch.save(model.state_dict(), MODELS_DIR / "tactical_cbm.pt")
            
    print("\n" + "=" * 65)
    print("TRAINING FINISHED! Best Epoch Summary:")
    print(f"  Best Val ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"  Best Val Mean R2: {best_metrics['mean_r2']:.4f}")
    print("  Per-Concept R2 Breakdown:")
    for c_name, r2_val in best_metrics['r2_scores'].items():
        print(f"    - {c_name:<30}: R2 = {r2_val:.3f}")
        
    # Export Sidecar for Stage 5 Instant Slider Recomputation
    model.load_state_dict(torch.load(MODELS_DIR / "tactical_cbm.pt"))
    linear_w = model.linear_head.weight.detach().cpu().squeeze().tolist()
    linear_b = float(model.linear_head.bias.detach().cpu().item())
    
    sidecar_data = {
        "model_architecture": "TacticalCBM_DenseGATv2",
        "stage5_contract": "Threat = sigmoid(sum(W_i * C_i) + b)",
        "weights": linear_w,
        "bias": linear_b,
        "concepts": CONCEPT_NAMES,
        "validation_metrics": {
            "mean_r2": best_metrics['mean_r2'],
            "roc_auc": best_metrics['roc_auc']
        }
    }
    
    sidecar_path = MODELS_DIR / "linear_head_weights.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)
        
    print(f"\nExported linear head sidecar to: {sidecar_path}")
    print("Linear Head Weights (w1 ... w6):", [f"{w:.4f}" for w in linear_w])
    print(f"Linear Head Bias (b):            {linear_b:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)

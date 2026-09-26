"""
src/dl_utils.py — Unified Deep Learning Training & Evaluation Utilities for FloraGuard AI
Includes:
1. Hardware accelerator detection (MPS / CUDA / CPU).
2. Standardized epoch training loop with gradient clipping.
3. Comprehensive evaluation suite (Loss, Top-1 Accuracy, Macro-F1, Precision, Recall).
4. Early Stopping tracker to halt training upon plateau.
5. Checkpoint saving and loading utilities.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def get_device() -> torch.device:
    """
    Auto-detects the fastest hardware accelerator available:
    - Apple Silicon GPU (MPS)
    - NVIDIA CUDA GPU
    - CPU Fallback
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")



def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
    desc: str = "Training",
) -> Dict[str, float]:
    """
    Executes one full training epoch across all batches with real-time progress tracking.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True)
    except ImportError:
        pbar = dataloader

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and gradient clipping
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data).item()
        total_samples += images.size(0)

        # Update live progress bar display
        if hasattr(pbar, "set_postfix"):
            cur_loss = running_loss / max(total_samples, 1)
            cur_acc = correct_predictions / max(total_samples, 1)
            pbar.set_postfix({"loss": f"{cur_loss:.4f}", "acc": f"{cur_acc*100:.1f}%"})

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = correct_predictions / max(total_samples, 1)

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
) -> Dict[str, float]:
    """
    Evaluates model performance across evaluation dataset with real-time progress tracking.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True)
    except ImportError:
        pbar = dataloader

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            total_samples += images.size(0)

            if hasattr(pbar, "set_postfix"):
                cur_loss = running_loss / max(total_samples, 1)
                pbar.set_postfix({"loss": f"{cur_loss:.4f}"})

    val_loss = running_loss / max(total_samples, 1)
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    return {
        "loss": float(val_loss),
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
    }


class EarlyStopping:
    """
    Monitors a metric (e.g., validation loss or F1) and signals when training should halt.
    """

    def __init__(
        self, patience: int=5, 
        min_delta: float=1e-4,
        mode: str="min",
    ):
        self.patience=patience
        self.min_delta=min_delta
        self.mode=mode
        self.counter=0
        self.best_score: Optional[float]=None
        self.early_stop=False

        if self.mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'") 

    def __call__(self, current_metric: float)-> Tuple[bool, float]:
        """
        Returns (should_stop, best_score)
        """ 

        if self.best_score is None:
            self.best_score=current_metric
            return False, True

        is_improvement=False
        if self.mode=="min":
            is_improvement=current_metric<(self.best_score-self.min_delta)
        else:
            is_improvement=current_metric>(self.best_score+self.min_delta)

        
        if is_improvement:
            self.best_score=current_metric
            self.counter=0
            return False, True

        else:
            self.counter+=1
            if self.counter>=self.patience:
                self.early_stop=True
                return True, False

            return False, False 


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Path,
    is_best: bool = False,
    best_filepath: Optional[Path] = None,
):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)

    if is_best and best_filepath is not None:
        best_filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, best_filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer]=None,
    device: Optional[torch.device]=None,

) -> Dict[str, Any]:
     
    if device is None:
        device=get_device()

    checkpoint=torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DEEP LEARNING UTILITIES ENGINE")
    print("=" * 60)

    device = get_device()
    print(f"• Detected Compute Device:  {device.type.upper()}")

    # Test EarlyStopping tracker
    early_stopper = EarlyStopping(patience=3, mode="min")
    losses = [0.85, 0.70, 0.65, 0.66, 0.67, 0.68]  # Loss plateaus after 0.65
    print("\n• Testing EarlyStopping logic on simulated loss sequence:")

    for epoch, loss in enumerate(losses, start=1):
        should_stop, is_best = early_stopper(loss)
        status = "⭐ Best model saved" if is_best else f"No improvement (patience {early_stopper.counter}/{early_stopper.patience})"
        print(f"  Epoch {epoch}: Loss={loss:.2f} -> {status}")
        if should_stop:
            print(f"  🛑 Early stopping triggered at epoch {epoch}!")
            break

    print("=" * 60)
    print("✅ Deep Learning Utilities verified successfully!\n")

        
        

    

    
    
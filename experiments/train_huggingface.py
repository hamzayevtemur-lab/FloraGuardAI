"""
experiments/train_huggingface.py — Fine-tuning Pipeline for Hugging Face ViT-Base
Fine-tunes google/vit-base-patch16-224 on PlantVillage using a low LR
to preserve pretrained patch attention weights.
Saves best checkpoint to models/huggingface_vit_best.pt.
"""


import sys
import json
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import get_dataloaders
from src.models.huggingface_model import PlantDiseaseHFModel
from src.dl_utils import (
    get_device,
    train_one_epoch,
    evaluate_model,
    EarlyStopping,
    save_checkpoint,
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "model_name": "google/vit-base-patch16-224",
    "lr": 2e-5,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
}

def train_huggingface(epochs: int=10, batch_size: int=32, num_workers: int=2):
    device=get_device()

    print("=" * 70)
    print("FLORAGUARD AI — TRAINING: HUGGING FACE ViT-BASE")
    print(f"• Hardware Accelerator: {device.type.upper()}")
    print(f"• Checkpoint:          {CONFIG['model_name']}")
    print(f"• Max Epochs:          {epochs}")
    print(f"• Batch Size:          {batch_size} (smaller due to ViT memory demands)")
    print("=" * 70)


    train_loader, val_loader, _, class_names = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model=PlantDiseaseHFModel(
        model_name=CONFIG["model_name"],
        num_classes=len(class_names),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    early_stopper = EarlyStopping(patience=4, mode="max")

    history = []
    print(f"\n{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val F1':<8}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Epoch {epoch:02d}/{epochs} [Train]"
        )
        val_metrics = evaluate_model(
            model, val_loader, criterion, device,
            desc=f"Epoch {epoch:02d}/{epochs} [Val]"
        )

        scheduler.step()

        should_stop, is_best = early_stopper(val_metrics["f1_macro"])

        print(
            f"{epoch:<8} | {train_metrics['loss']:<12.4f} | {train_metrics['accuracy']*100:<12.2f} | "
            f"{val_metrics['loss']:<10.4f} | {val_metrics['accuracy']*100:<10.2f} | {val_metrics['f1_macro']:<8.4f}"
            + (" ⭐" if is_best else ""),
            flush=True
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        })

        if is_best:
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1_macro": val_metrics["f1_macro"],
                    "val_accuracy": val_metrics["accuracy"],
                    "class_names": class_names,
                    "config": CONFIG,
                },
                filepath=MODELS_DIR / "huggingface_vit_latest.pt",
                is_best=True,
                best_filepath=MODELS_DIR / "huggingface_vit_best.pt",
            )

        if should_stop:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}.")
            break

    history_path = REPORTS_DIR / "huggingface_vit_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ Training complete! Best Val F1: {early_stopper.best_score:.4f}")
    print(f"📁 Best checkpoint: {MODELS_DIR / 'huggingface_vit_best.pt'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    train_huggingface(epochs=10, batch_size=32)
    
       





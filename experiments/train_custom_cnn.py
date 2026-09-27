"""
experiments/train_custom_cnn.py — Full Training Pipeline for Custom CNN
Trains PlantDiseaseCNN with the best hyperparameters found from tuning.
Saves the best model checkpoint to models/custom_cnn_best.pt.
"""
import sys
import json
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import get_dataloaders
from src.models.custom_cnn import PlantDiseaseCNN
from src.dl_utils import (
    get_device,
    train_one_epoch,
    evaluate_model,
    EarlyStopping,
    save_checkpoint
)

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TUNING_RESULTS = PROJECT_ROOT / "reports" / "tuning" / "custom_cnn_tuning_results.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def train_custom_cnn(
    epochs: int = 24,
    batch_size: int = 64,
    num_workers: int = 2,
):
    device=get_device()

    print("=" * 70)
    print("FLORAGUARD AI — TRAINING: CUSTOM CNN")
    print(f"• Hardware Accelerator: {device.type.upper()}")
    print(f"• Max Epochs:          {epochs}")
    print(f"• Batch Size:          {batch_size}")
    print("=" * 70)

    ## 1. Load best hyperparameters from tuning results (with fallback defaults)
    if TUNING_RESULTS.exists():
        with open(TUNING_RESULTS) as f:
            tuning=json.load(f)

        best_config=tuning["best_config"]
        print(f"\n✅ Loaded best hyperparameters from tuning: {best_config}")

    else:
        best_config = {"lr": 1e-3, "optimizer": "AdamW", "dropout_rate": 0.4}
        print(f"\n⚠️  No tuning results found. Using defaults: {best_config}")

    ## 2. Load DataLoaders
    train_loader, val_loader, _, class_names=get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    ## 3. Initialize Model
    model=PlantDiseaseCNN(
        num_classes=len(class_names),
        dropout_rate=best_config["dropout_rate"],
    ).to(device)

    ## 4. Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()

    if best_config["optimizer"]=="AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_config["lr"], weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=best_config["lr"], momentum=0.9, weight_decay=1e-4)

    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopper = EarlyStopping(patience=7, mode="max")  # monitor Macro-F1

    # 5. Training Loop
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

        current_lr = scheduler.get_last_lr()[0]
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
            "lr": current_lr,
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
                    "config": best_config,
                },
                filepath=MODELS_DIR / "custom_cnn_latest.pt",
                is_best=True,
                best_filepath=MODELS_DIR / "custom_cnn_best.pt",
            )
        if should_stop:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}.")
            break


    # 6. Save training history
    history_path = REPORTS_DIR / "custom_cnn_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print("\n" + "=" * 70)
    print(f"✅ Training complete! Best Val F1: {early_stopper.best_score:.4f}")
    print(f"📁 Best checkpoint saved to: {MODELS_DIR / 'custom_cnn_best.pt'}")
    print(f"📁 Training history saved to: {history_path}")
    print("=" * 70 + "\n")



if __name__ == "__main__":
    train_custom_cnn(epochs=24, batch_size=64)



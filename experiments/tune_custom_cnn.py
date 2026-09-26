"""
experiments/tune_custom_cnn.py — Hyperparameter Search for Custom CNN
Searches across:
- Optimizers (AdamW vs SGD with Momentum)
- Learning Rates (1e-3, 5e-4, 1e-4)
- Dropout Rates (0.3, 0.5)
Outputs results and rankings to reports/tuning/custom_cnn_tuning_results.json.
"""

import sys
import os
import json
import itertools
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import get_dataloaders
from src.models.custom_cnn import PlantDiseaseCNN
from src.dl_utils import get_device, train_one_epoch, evaluate_model

# Output directory for tuning reports
REPORTS_DIR = PROJECT_ROOT / "reports" / "tuning"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def run_tuning(
    epochs_per_trial: int=2,
    batch_size: int=64,
    num_workers: int=2, 
):
    device = get_device()
    print("=" * 70)
    print("FLORAGUARD AI — HYPERPARAMETER TUNING: CUSTOM CNN")
    print(f"• Hardware Accelerator: {device.type.upper()}")
    print(f"• Epochs per Trial:    {epochs_per_trial}")
    print(f"• Batch Size:          {batch_size}")
    print("=" * 70)

    # 1.Load Dataloader
    print("\nLoading dataset splits...")
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 2. Define Search Space
    search_space = {
        "lr": [1e-3, 5e-4],
        "optimizer": ["AdamW", "SGD"],
        "dropout_rate": [0.3, 0.5],
    }

    # Generate all hyperparameter combinations
    keys, values = zip(*search_space.items())
    trial_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_trials = len(trial_configs)

    print(f"\nGenerated {total_trials} hyperparameter trials to evaluate.\n")

    criterion = nn.CrossEntropyLoss()
    results: List[Dict[str, Any]] = []

    #3. Execute Grid Search Trials
    for trial_idx, config in enumerate(trial_configs, start=1):
        print("-" * 70)
        print(f"Trial {trial_idx}/{total_trials} -> LR: {config['lr']}, Optimizer: {config['optimizer']}, Dropout: {config['dropout_rate']}")
        print("-" * 70)

        # Initialize fresh model for each trial
        model = PlantDiseaseCNN(num_classes=38, dropout_rate=config["dropout_rate"]).to(device)

        # Configure optimizer
        if config["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=1e-4)

        # Train for specified trial epochs
        for epoch in range(1, epochs_per_trial + 1):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate_model(model, val_loader, criterion, device)

            print(
                f"  Epoch {epoch}/{epochs_per_trial} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']*100:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']*100:.2f}%, Val F1: {val_metrics['f1_macro']:.4f}"
            )

        trial_record = {
            "trial_id": trial_idx,
            "config": config,
            "final_val_loss": val_metrics["loss"],
            "final_val_accuracy": val_metrics["accuracy"],
            "final_val_f1_macro": val_metrics["f1_macro"],
        }

        results.append(trial_record)


    # 4. Rank Results by Validation Macro-F1 (descending)
    results.sort(key=lambda x: x["final_val_f1_macro"], reverse=True)
    best_trial = results[0]

    # Save to JSON
    output_json_path = REPORTS_DIR / "custom_cnn_tuning_results.json"

    with open(output_json_path, "w") as f:
        json.dump(
            {
                "best_config": best_trial["config"],
                "best_val_f1_macro": best_trial["final_val_f1_macro"],
                "all_trials": results,
            },
            f,
            indent=2,
        )


    # 5. Print Leaderboard Summary
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING LEADERBOARD (Ranked by Validation Macro-F1)")
    print("=" * 70)
    print(f"{'Rank':<5} | {'LR':<8} | {'Optimizer':<10} | {'Dropout':<8} | {'Val Loss':<10} | {'Val Acc (%)':<12} | {'Val F1':<8}")
    print("-" * 70)
    for rank, trial in enumerate(results, start=1):
        cfg = trial["config"]
        print(
            f"{rank:<5} | {cfg['lr']:<8} | {cfg['optimizer']:<10} | {cfg['dropout_rate']:<8} | "
            f"{trial['final_val_loss']:<10.4f} | {trial['final_val_accuracy']*100:<12.2f} | {trial['final_val_f1_macro']:<8.4f}"
        )
    print("=" * 70)
    print(f"🏆 Winning Configuration: {best_trial['config']}")
    print(f"📁 Full Tuning Report saved to: {output_json_path}\n")

if __name__ == "__main__":
    run_tuning(epochs_per_trial=2, batch_size=64)


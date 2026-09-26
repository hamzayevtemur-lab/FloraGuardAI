"""
experiments/compare_models.py — Final Benchmark: Compare All Trained Models on Test Set
Loads best checkpoints for all four models and evaluates each on the held-out test set.
Outputs a comparison leaderboard and saves a summary report to reports/model_comparison.json.
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
from src.models.resnet import PlantDiseaseResNet18
from src.models.mobilenet import PlantDiseaseMobileNet
from src.models.huggingface_model import PlantDiseaseHFModel
from src.dl_utils import get_device, evaluate_model, load_checkpoint

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compare_all_models(batch_size: int=64, num_workers: int=2):
    device=get_device()

    print("=" * 70)
    print("FLORAGUARD AI — FINAL MODEL COMPARISON ON TEST SET")
    print(f"• Hardware Accelerator: {device.type.upper()}")
    print("=" * 70)

    _, _, test_loader, class_names = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    criterion = nn.CrossEntropyLoss()

    # Registry of all models and their checkpoint paths
    model_registry = [
        {
            "name": "Custom CNN (Scratch)",
            "checkpoint": MODELS_DIR / "custom_cnn_best.pt",
            "model_class": PlantDiseaseCNN,
            "init_kwargs": {"num_classes": len(class_names), "dropout_rate": 0.4},
        },
        {
            "name": "ResNet-18",
            "checkpoint": MODELS_DIR / "resnet18_best.pt",
            "model_class": PlantDiseaseResNet18,
            "init_kwargs": {"num_classes": len(class_names), "pretrained": False},
        },
        {
            "name": "MobileNetV3-Large",
            "checkpoint": MODELS_DIR / "mobilenet_best.pt",
            "model_class": PlantDiseaseMobileNet,
            "init_kwargs": {"num_classes": len(class_names), "pretrained": False},
        },
        {
            "name": "ViT-Base (Hugging Face)",
            "checkpoint": MODELS_DIR / "huggingface_vit_best.pt",
            "model_class": PlantDiseaseHFModel,
            "init_kwargs": {
                "model_name": "google/vit-base-patch16-224",
                "num_classes": len(class_names),
                "pretrained": False,
            },
        },
    ]


    results=[]
    for entry in model_registry:
        checkpoint_path=entry["checkpoint"]
        if not checkpoint_path.exists():
            print(f"\n⚠️  Skipping {entry['name']}: checkpoint not found at {checkpoint_path}")
            continue

        
        print(f"\n--- Evaluating: {entry['name']} ---")
        model=entry["model_class"](**entry["init_kwargs"])
        load_checkpoint(filepath=checkpoint_path, model=model, device=device)
        model=model.to(device)

        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        best_epoch = checkpoint_data.get("epoch", "N/A")

        param_count = sum(p.numel() for p in model.parameters())

        test_metrics = evaluate_model(
            model, test_loader, criterion, device,
            desc=f"Testing {entry['name']}"
        )

        print(f"  Test Accuracy:   {test_metrics['accuracy']*100:.2f}%")
        print(f"  Test Macro-F1:   {test_metrics['f1_macro']:.4f}")
        print(f"  Test Precision:  {test_metrics['precision_macro']:.4f}")
        print(f"  Test Recall:     {test_metrics['recall_macro']:.4f}")
        print(f"  Parameters:      {param_count:,}")

        results.append({
            "model": entry["name"],
            "best_epoch": best_epoch,
            "parameters": param_count,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "test_loss": test_metrics["loss"],
        })

    if not results:
        print("\n❌ No model checkpoints found. Train your models first!")
        return

    # Sort by Test Macro-F1
    results.sort(key=lambda x: x["test_f1_macro"], reverse=True)

    print("\n" + "=" * 70)
    print("FINAL LEADERBOARD (Ranked by Test Macro-F1)")
    print("=" * 70)

    print(f"{'Rank':<5} | {'Model':<28} | {'Params':>10} | {'Acc (%)':<10} | {'F1 Macro':<10} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 85)

    for rank, r in enumerate(results, start=1):
        print(
            f"{rank:<5} | {r['model']:<28} | {r['parameters']:>10,} | "
            f"{r['test_accuracy']*100:<10.2f} | {r['test_f1_macro']:<10.4f} | "
            f"{r['test_precision_macro']:<10.4f} | {r['test_recall_macro']:<8.4f}"
        )

    print("=" * 85)
    winner = results[0]
    print(f"\n🏆 Best Model: {winner['model']}")
    print(f"   Test Accuracy: {winner['test_accuracy']*100:.2f}%  |  Test Macro-F1: {winner['test_f1_macro']:.4f}")

    output_path = REPORTS_DIR / "model_comparison.json"

    with open(output_path, "w") as f:
        json.dump({"winner": winner["model"], "leaderboard": results}, f, indent=2)
    print(f"\n📁 Comparison report saved to: {output_path}\n")


if __name__ == "__main__":
    compare_all_models(batch_size=64)

        











"""
src/split_data.py — Stratified Dataset Splitter for FloraGuard AI.

Creates reproducible, leakage-safe train/validation/test splits:
- 70% Training (~38,012 images)
- 15% Validation (~8,146 images)
- 15% Test (~8,146 images)
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_from_disk

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def create_stratified_splits(train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, random_seed: int = 42):
    """
    Loads raw PlantVillage data, creates stratified splits, and saves train.csv, validation.csv, and test.csv.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    print(f"Loading raw dataset from: {RAW_DATA_DIR} ...")
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_DIR}")

    dataset = load_from_disk(str(RAW_DATA_DIR))
    total_images = len(dataset)
    print(f"Total raw images loaded: {total_images:,}")

    # Extract metadata columns (excluding image bytes for speed)
    df = pd.DataFrame({
        "image_index": range(len(dataset)),
        "class_idx": dataset["class_idx"],
        "class_label": dataset["class_label"],
        "host": dataset["host"],
        "disease": dataset["disease"]
    })

    num_classes = df["class_idx"].nunique()
    print(f"Detected {num_classes} distinct disease classes across {df['host'].nunique()} crops.")

    # 1. First Split: Train (70%) vs Temp (30%)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["class_idx"],
        random_state=random_seed
    )

    # 2. Second Split: Validation (15%) vs Test (15%)
    val_share_of_temp = val_ratio / temp_ratio  # 0.15 / 0.30 = 0.50
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        stratify=temp_df["class_idx"],
        random_state=random_seed
    )

    # Ensure output directory exists
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # Save to CSV
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "validation.csv"
    test_path = SPLITS_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"• Training Set:   {len(train_df):>6,} images ({len(train_df)/total_images*100:>5.1f}%) -> {train_path.name}")
    print(f"• Validation Set: {len(val_df):>6,} images ({len(val_df)/total_images*100:>5.1f}%) -> {val_path.name}")
    print(f"• Test Set:       {len(test_df):>6,} images ({len(test_df)/total_images*100:>5.1f}%) -> {test_path.name}")
    print(f"• Total Images:   {total_images:>6,} images")
    print("=" * 60)
    print("Stratified splits generated successfully without data leakage!\n")


if __name__ == "__main__":
    create_stratified_splits()

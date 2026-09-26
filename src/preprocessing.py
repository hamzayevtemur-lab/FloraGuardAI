"""
src/preprocessing.py — PyTorch Dataset & DataLoader Pipeline for FloraGuard AI
Handles:
1. Custom PyTorch Dataset reading from Hugging Face disk dataset + CSV splits.
2. Domain-tailored data augmentation (rotations, flips, color jitter, ImageNet normalization).
3. Optimized DataLoader creation for train, validation, and test splits.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_from_disk


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


## ImageNet normalization statistics standard across PyTorch vision models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224)-> transforms.Compose:
    """
    Data augmentation pipeline for training to combat overfitting on leaf patterns.
    """

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int=224)->transforms.Compose:
    """
    Deterministic preprocessing pipeline for validation, testing, and inference.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class PlantVillageDataset(Dataset):
    """
    PyTorch Dataset wrapping the PlantVillage disk dataset indexed via stratified CSV splits.
    """
    def __init__(
        self, split_df: pd.DataFrame,
        raw_dataset,
        transform:Optional[transforms.Compose]=None,
    ):
        self.df=split_df.reset_index(drop=True)
        self.raw_dataset=raw_dataset
        self.transform=transform

    def __len__(self)->int:
        return len(self.df)

    def __getitem__(self, idx: int)-> Tuple[torch.Tensor, int]:
        row=self.df.iloc[idx]
        raw_idx=int(row["image_index"])
        class_idx=int(row["class_idx"])

        # Fetch image from raw disk dataset
        img=self.raw_dataset[raw_idx]["image"]
        if not isinstance(img,Image.Image):
            img=Image.fromarray(img)

        # Convert to RGB if grayscale or RGBA
        if img.mode!="RGB":
            img=img.convert("RGB")

        # Apply transformation pipeline
        if self.transform is not None:
            image_tensor=self.transform(img)

        else:
            image_tensor=transforms.ToTensor()(img)

        return image_tensor, class_idx

def get_dataloaders(
    batch_size: int=64,
    num_workers: int=2,
    image_size: int=224,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:

    """
    Factory function to instantiate Train, Validation, and Test DataLoaders.
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # 1. Load raw dataset pnce into memory
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_DIR}")
    raw_dataset=load_from_disk(str(RAW_DATA_DIR))

    # 2. Load split metadata CSVs
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    #3. Extract ordered list of 38 class names
    class_map=(
        train_df[["class_idx", "class_label"]]
        .drop_duplicates()
        .sort_values(by="class_idx")
    )
    class_names=class_map["class_label"].tolist()

    #4. Create Dataset with respective transform pipelines
    train_dataset=PlantVillageDataset(
        split_df=train_df,
        raw_dataset=raw_dataset,
        transform=get_train_transforms(image_size=image_size)
    )

    val_dataset=PlantVillageDataset(
        split_df=val_df,
        raw_dataset=raw_dataset,
        transform=get_eval_transforms(image_size=image_size),
    )

    test_dataset=PlantVillageDataset(
        split_df=test_df,
        raw_dataset=raw_dataset,
        transform=get_eval_transforms(image_size=image_size),
    )

    ## 5. Create PyTorch Dataloaders
    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader=DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_names


if __name__=="__main__":
    print("=" * 65)
    print("TESTING PREPROCESSING & DATALOADER PIPELINE")
    print("=" * 65)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=32, num_workers=2
    )

    print(f"• Number of classes:        {len(class_names)}")
    print(f"• Training batches:         {len(train_loader):,} (Total: {len(train_loader.dataset):,} images)")
    print(f"• Validation batches:       {len(val_loader):,} (Total: {len(val_loader.dataset):,} images)")
    print(f"• Test batches:             {len(test_loader):,} (Total: {len(test_loader.dataset):,} images)")
    
    # Fetch 1 sample batch to verify tensor shapes and values
    images, labels = next(iter(train_loader))
    print(f"\nSample Training Batch Verification:")
    print(f"  - Image Batch Tensor Shape: {images.shape} (Expected: [32, 3, 224, 224])")
    print(f"  - Label Batch Tensor Shape: {labels.shape} (Expected: [32])")
    print(f"  - Tensor Dtype:             {images.dtype}")
    print(f"  - Pixel Value Range:        min={images.min():.2f}, max={images.max():.2f}")
    print("\n✅ Preprocessing & DataLoader pipeline verified successfully!\n")

    
    

    

    

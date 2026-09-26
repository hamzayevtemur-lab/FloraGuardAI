"""
src/models/custom_cnn.py — Custom CNN Architecture for Plant Disease Classification

Architecture Design:
  Feature Extractor (4 Conv Blocks):
    Block 1: Conv2d(3 -> 32, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2,2)   [112 x 112]
    Block 2: Conv2d(32 -> 64, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2,2)  [56 x 56]
    Block 3: Conv2d(64 -> 128, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2,2) [28 x 28]
    Block 4: Conv2d(128 -> 256, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2,2)[14 x 14]

  Classification Head:
    Global Average Pooling: AdaptiveAvgPool2d((1, 1)) -> Flatten (256)
    -> Linear(256 -> 512) -> BatchNorm1d -> ReLU -> Dropout(p)
    -> Linear(512 -> num_classes)
"""

from typing import Dict
import torch
import torch.nn as nn


class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes: int = 38, dropout_rate: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 -> 128 channels
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 128 -> 256 channels
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling (1x1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CUSTOM CNN ARCHITECTURE")
    print("=" * 60)

    model = PlantDiseaseCNN(num_classes=38, dropout_rate=0.4)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"• Input Shape:       {dummy_input.shape}")
    print(f"• Output Shape:      {output.shape} (Expected: [2, 38])")
    print(f"• Total Parameters:  {total_params:,}")
    print(f"• Trainable Params:  {trainable_params:,}")
    print("=" * 60)
    print("✅ Custom CNN verified successfully!\n")

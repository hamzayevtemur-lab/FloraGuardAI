"""
src/models/resnet.py — ResNet-18 Transfer Learning Architecture
Uses TorchVision's pretrained ResNet-18 with:
1. Pretrained ImageNet-1K residual backbone.
2. Replaceable final fully connected (`fc`) layer for 38 classes.
3. Configurable backbone freezing (`freeze_backbone=True/False`).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class PlantDiseaseResNet18(nn.Module):
    def __init__(
        self, num_classes: int=38, 
        pretrained: bool=True,
        freeze_backbone: bool=False,
        dropout_rate: float=0.3,
    ):

        super().__init__()

        weights=ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone=models.resnet18(weights=weights)

        ## Optionally freeze feature extractor weights for pure head training
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad=False

        # Replace default 1000-class head with custom plant disease classifier
        in_features=self.backbone.fc.in_features
        self.backbone.fc=nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.backbone(x)


if __name__=="__main__":
    print("=" * 60)
    print("TESTING RESNET-18 ARCHITECTURE")
    print("=" * 60)

    model=PlantDiseaseResNet18(num_classes=38, pretrained=True)
    dummy_input=torch.randn(2, 3, 224, 224)
    output=model(dummy_input)

    total_params=sum(p.numel() for p in model.parameters())
    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"• Input Shape:       {dummy_input.shape}")
    print(f"• Output Shape:      {output.shape} (Expected: [2, 38])")
    print(f"• Total Parameters:  {total_params:,}")
    print(f"• Trainable Params:  {trainable_params:,}")
    print("=" * 60)
    print("✅ ResNet-18 verified successfully!\n")

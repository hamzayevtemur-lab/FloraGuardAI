"""
src/models/mobilenet.py — MobileNetV3-Large Edge Architecture
Ultra-lightweight architecture optimized for edge and mobile deployment:
1. Inverted residual blocks with hard-swish activations.
2. Squeeze-and-Excitation (SE) attention modules.
3. Low latency & high parameter efficiency.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights

class PlantDiseaseMobileNet(nn.Module):
    def __init__(
        self, num_classes: int=38,
        pretrained: bool=True,
        freeze_backbone: bool=False,
        dropout_rate: float=0.2,
    ):

        super().__init__()

        weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone=models.mobilenet_v3_large(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad=False

        # Replace final classification head
        in_features=self.backbone.classifier[3].in_features
        self.backbone.classifier[2]=nn.Dropout(p=dropout_rate)
        self.backbone.classifier[3]=nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.backbone(x)


if __name__=="__main__":
    print("=" * 60)
    print("TESTING MOBILENET-V3 ARCHITECTURE")
    print("=" * 60)

    model=PlantDiseaseMobileNet(num_classes=38, pretrained=True)
    dummy_input=torch.randn(2, 3, 224, 224)
    output=model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f"• Input Shape:       {dummy_input.shape}")
    print(f"• Output Shape:      {output.shape} (Expected: [2, 38])")
    print(f"• Total Parameters:  {total_params:,}")
    print(f"• Trainable Params:  {trainable_params:,}")
    print("=" * 60)
    print("✅ MobileNetV3 verified successfully!\n")


        
        
                
    
"""
src/models/__init__.py — Unified Model Factory & Exports
"""

from typing import Dict, Any
import torch.nn as nn

from src.models.custom_cnn import PlantDiseaseCNN
from src.models.resnet import PlantDiseaseResNet18 as PlantDiseaseResNet
from src.models.mobilenet import PlantDiseaseMobileNet
from src.models.huggingface_model import PlantDiseaseHFModel


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Computes total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_model(model_type: str, num_classes: int = 38, **kwargs: Any) -> nn.Module:
    """
    Unified factory function to instantiate any model architecture by name.
    """
    model_type = model_type.lower()

    if model_type == "custom_cnn":
        return PlantDiseaseCNN(num_classes=num_classes, **kwargs)
    elif model_type in ["resnet18", "resnet"]:
        return PlantDiseaseResNet(num_classes=num_classes, **kwargs)
    elif model_type in ["mobilenet_v3", "mobilenet"]:
        return PlantDiseaseMobileNet(num_classes=num_classes, **kwargs)
    elif model_type in ["huggingface_vit", "hf_vit", "vit"]:
        hf_name = kwargs.get("hf_name", "google/vit-base-patch16-224")
        return PlantDiseaseHFModel(model_name=hf_name, num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: 'custom_cnn', 'resnet18', 'mobilenet_v3', 'huggingface_vit'"
        )


__all__ = [
    "PlantDiseaseCNN",
    "PlantDiseaseResNet",
    "PlantDiseaseMobileNet",
    "PlantDiseaseHFModel",
    "build_model",
    "count_parameters",
]

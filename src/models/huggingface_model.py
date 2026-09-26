"""
src/models/huggingface_model.py — Hugging Face Vision Transformer / ConvNeXt Wrapper

Wraps Hugging Face's `AutoModelForImageClassification` into a standard PyTorch
`nn.Module` forward interface so that it seamlessly fits our training engine.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification


class PlantDiseaseHFModel(nn.Module):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 38,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name

        # Load Hugging Face vision backbone with custom classification head
        self.hf_model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hugging Face models return an ImageClassifierOutput object; extract logits tensor
        outputs = self.hf_model(pixel_values=x)
        return outputs.logits


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING HUGGING FACE VISION MODEL")
    print("=" * 60)

    hf_checkpoint = "google/vit-base-patch16-224"
    print(f"Loading checkpoint: {hf_checkpoint} ...")
    model = PlantDiseaseHFModel(model_name=hf_checkpoint, num_classes=38)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"• Input Shape:       {dummy_input.shape}")
    print(f"• Output Shape:      {output.shape} (Expected: [2, 38])")
    print(f"• Total Parameters:  {total_params:,}")
    print("=" * 60)
    print("✅ Hugging Face Vision Model verified successfully!\n")

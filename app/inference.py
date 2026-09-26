"""
app/inference.py — Real-Time Model Inference Engine
Handles:
1. Dynamic discovery and loading of trained PyTorch checkpoints (.pt).
2. Hardware accelerator selection (MPS / CUDA / CPU).
3. Image preprocessing pipeline (RGB, 224x224, ImageNet normalization).
4. Top-K confidence breakdown and disease treatment lookup.
"""

import io
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from src.models import build_model, count_parameters
from src.dl_utils import get_device
from app.disease_info import get_disease_info

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Standard computer vision inference transform
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Default PlantVillage 38 class list fallback
DEFAULT_CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


import numpy as np


def is_crop_leaf_image(img: Image.Image) -> Tuple[bool, str]:
    """
    Bio-optical filter to verify if an uploaded image contains genuine plant foliage
    rather than screenshots, UI, documents, or unrelated indoor objects.
    """
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Calculate saturation
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c
    sat = np.where(max_c > 0.05, delta / (max_c + 1e-6), 0)

    # 1. Grayscale / monochrome dominance (typical of screenshots, code editors, text, windows)
    grayscale_ratio = float(np.mean(sat < 0.12))
    if grayscale_ratio > 0.85:
        return False, f"Image appears to be non-organic (monochrome ratio: {grayscale_ratio*100:.1f}%). Please upload a real plant leaf photo."

    # 2. Plant foliage color mask:
    # Chlorophyll green or chlorotic/necrotic yellow/brown foliage
    is_green = (g > r * 0.95) & (g > b * 1.02) & (sat > 0.12)
    is_plant_brown_yellow = (r > b * 1.1) & (g > b * 0.88) & (sat > 0.14) & (g > 0.12) & (r < 0.96)

    plant_mask = is_green | is_plant_brown_yellow
    plant_pixel_ratio = float(np.mean(plant_mask))

    if plant_pixel_ratio < 0.08:
        return False, f"No crop foliage detected ({plant_pixel_ratio*100:.1f}% plant pixels). Please ensure the leaf fills a significant portion of the photo."

    return True, "Valid plant foliage"


class ModelManager:
    """Manages active deep learning model in memory and provides real-time inference."""

    def __init__(self):
        self.device = get_device()
        self.active_model: Optional[nn.Module] = None
        self.active_model_name: Optional[str] = None
        self.active_metadata: Dict[str, Any] = {}
        self.class_names: List[str] = DEFAULT_CLASS_NAMES

    def list_available_models(self) -> List[Dict[str, Any]]:
        """Scans the models/ directory for all available trained .pt files."""
        available = []
        if not MODELS_DIR.exists():
            return available

        for pt_file in sorted(MODELS_DIR.glob("*.pt")):
            size_mb = pt_file.stat().st_size / (1024 * 1024)
            arch = self._infer_architecture(pt_file.name)
            is_active = (self.active_model_name == pt_file.name)

            model_info = {
                "filename": pt_file.name,
                "filepath": str(pt_file),
                "architecture": arch,
                "size_mb": round(size_mb, 2),
                "is_active": is_active,
            }

            # Inspect checkpoint metadata safely
            try:
                chk = torch.load(pt_file, map_location="cpu")
                model_info["epoch"] = chk.get("epoch", "N/A")
                model_info["val_accuracy"] = round(chk.get("val_accuracy", 0.0) * 100, 2) if "val_accuracy" in chk else None
                model_info["val_f1_macro"] = round(chk.get("val_f1_macro", 0.0), 4) if "val_f1_macro" in chk else None
            except Exception:
                pass

            available.append(model_info)

        return available

    def _infer_architecture(self, filename: str) -> str:
        """Determines model architecture string from filename."""
        name = filename.lower()
        if "resnet" in name:
            return "resnet18"
        elif "mobilenet" in name:
            return "mobilenet_v3"
        elif "vit" in name or "huggingface" in name:
            return "huggingface_vit"
        elif "cnn" in name or "custom" in name:
            return "custom_cnn"
        return "custom_cnn"

    def load_model(self, filename: Optional[str] = None) -> bool:
        """Loads a checkpoint into memory. If filename is None, loads the best available model."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        target_file: Optional[Path] = None
        if filename:
            target_file = MODELS_DIR / filename
        else:
            # Preferred priority order
            priority = ["custom_cnn_best.pt", "resnet18_best.pt", "mobilenet_best.pt", "huggingface_vit_best.pt"]
            for p in priority:
                if (MODELS_DIR / p).exists():
                    target_file = MODELS_DIR / p
                    break

            if target_file is None:
                # Any .pt file in directory
                all_pts = list(MODELS_DIR.glob("*.pt"))
                if all_pts:
                    target_file = all_pts[0]

        if target_file is None or not target_file.exists():
            print(f"⚠️ No model checkpoint found in {MODELS_DIR}")
            self.active_model = None
            self.active_model_name = None
            return False

        print(f"🔄 Loading model checkpoint: {target_file.name} onto {self.device}...")
        try:
            checkpoint = torch.load(target_file, map_location=self.device)
            arch = self._infer_architecture(target_file.name)

            if "class_names" in checkpoint:
                self.class_names = checkpoint["class_names"]

            num_classes = len(self.class_names)
            model = build_model(model_type=arch, num_classes=num_classes)

            # Load weights
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.active_model = model
            self.active_model_name = target_file.name
            params = count_parameters(model)

            self.active_metadata = {
                "name": target_file.name,
                "architecture": arch,
                "epoch": checkpoint.get("epoch", "N/A"),
                "val_accuracy": round(checkpoint.get("val_accuracy", 0.0) * 100, 2) if "val_accuracy" in checkpoint else None,
                "val_f1_macro": round(checkpoint.get("val_f1_macro", 0.0), 4) if "val_f1_macro" in checkpoint else None,
                "parameters": params["total"],
                "trainable_parameters": params["trainable"],
                "device": str(self.device).upper(),
                "num_classes": num_classes,
            }

            print(f"✅ Active Model: {target_file.name} ({arch}) loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Failed to load model {target_file.name}: {e}")
            return False

    def predict(self, image_bytes: bytes, top_k: int = 5) -> Dict[str, Any]:
        """
        Runs inference on uploaded image bytes with Out-of-Distribution (OOD) protection.
        """
        if self.active_model is None:
            # Attempt to auto-load
            if not self.load_model():
                raise RuntimeError(
                    "No trained model is currently loaded. Place your .pt checkpoint in the models/ folder."
                )

        start_time = time.perf_counter()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # --- TIER 1: Biological Foliage Guardrail ---
        is_plant, reason = is_crop_leaf_image(image)
        if not is_plant:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "is_valid_leaf": False,
                "rejection_reason": reason,
                "prediction": None,
                "top_k": [],
                "inference_time_ms": round(latency_ms, 2),
                "model_used": {
                    "name": self.active_model_name,
                    "architecture": self.active_metadata.get("architecture"),
                    "device": str(self.device).upper(),
                }
            }

        # --- TIER 2: Deep Learning Inference ---
        tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.active_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract Top-K predictions
        top_k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probs, k=top_k)

        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.class_names[idx.item()]
            info = get_disease_info(class_name)
            top_predictions.append({
                "class_name": class_name,
                "crop": info["crop"],
                "condition": info["condition"],
                "is_healthy": info["is_healthy"],
                "probability": float(prob.item()),
                "confidence_percent": round(float(prob.item()) * 100, 2),
            })

        best = top_predictions[0]

        # --- TIER 3: Confidence Threshold Filter (< 50% = Unknown/Uncertain) ---
        if best["probability"] < 0.50:
            return {
                "is_valid_leaf": False,
                "rejection_reason": f"Low diagnostic confidence ({best['confidence_percent']}%). The image does not clearly match any of the 38 recognized crop diseases.",
                "prediction": None,
                "top_k": top_predictions,
                "inference_time_ms": round(latency_ms, 2),
                "model_used": {
                    "name": self.active_model_name,
                    "architecture": self.active_metadata.get("architecture"),
                    "device": str(self.device).upper(),
                }
            }

        full_info = get_disease_info(best["class_name"])

        return {
            "is_valid_leaf": True,
            "prediction": {
                "class_name": best["class_name"],
                "crop": best["crop"],
                "condition": best["condition"],
                "is_healthy": best["is_healthy"],
                "confidence": best["probability"],
                "confidence_percent": best["confidence_percent"],
                "pathogen": full_info["pathogen"],
                "symptoms": full_info["symptoms"],
                "treatment": full_info["treatment"],
            },
            "top_k": top_predictions,
            "inference_time_ms": round(latency_ms, 2),
            "model_used": {
                "name": self.active_model_name,
                "architecture": self.active_metadata.get("architecture"),
                "device": str(self.device).upper(),
                "accuracy": self.active_metadata.get("val_accuracy"),
            }
        }


# Singleton manager instance
model_manager = ModelManager()

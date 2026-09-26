"""
app/main.py — Production FastAPI Diagnostic Engine for FloraGuard AI
Serves:
1. RESTful API for model discovery, loading, and real-time disease inference.
2. Static web dashboard for browser-based interactive leaf scanning.
"""

import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.inference import model_manager

# Initialize FastAPI application
app = FastAPI(
    title="FloraGuard AI — Plant Disease Diagnostic API",
    description="High-precision deep learning diagnostic engine for agricultural crop diseases.",
    version="1.0.0",
)

# Enable Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Attempt to load the best available model on server launch."""
    print("🌿 FloraGuard AI is booting up...")
    model_manager.load_model()


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Returns engine health, compute hardware, and active model state."""
    available = model_manager.list_available_models()
    return {
        "status": "healthy",
        "device": str(model_manager.device).upper(),
        "active_model": model_manager.active_model_name,
        "active_metadata": model_manager.active_metadata,
        "available_models_count": len(available),
    }


@app.get("/api/models")
async def get_models() -> Dict[str, Any]:
    """Returns list of all model checkpoints found in models/ directory."""
    models = model_manager.list_available_models()
    return {
        "active_model": model_manager.active_model_name,
        "active_metadata": model_manager.active_metadata,
        "models": models,
    }


@app.post("/api/models/load")
async def switch_model(filename: str = Query(..., description="Checkpoint filename in models/")) -> Dict[str, Any]:
    """Switches active model checkpoint in memory."""
    success = model_manager.load_model(filename)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Could not load checkpoint '{filename}'. Ensure the file exists in the models/ folder."
        )
    return {
        "success": True,
        "message": f"Successfully activated {filename}",
        "active_metadata": model_manager.active_metadata,
    }


from pydantic import BaseModel
import requests
from PIL import Image
import io


class URLPredictionRequest(BaseModel):
    image_url: str
    top_k: int = 5


@app.post("/api/predict")
async def predict_plant_disease(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=10)
) -> Dict[str, Any]:
    """
    Accepts an uploaded crop leaf image and returns real-time diagnostic predictions,
    confidence metrics, symptom profiles, and targeted agronomic treatments.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPEG/PNG/WebP).")

    try:
        contents = await file.read()
        results = model_manager.predict(image_bytes=contents, top_k=top_k)
        return results
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {str(e)}")


@app.post("/api/predict-url")
async def predict_plant_disease_from_url(payload: URLPredictionRequest) -> Dict[str, Any]:
    """
    Downloads an image from a public URL and runs deep learning diagnostic inference.
    """
    url = payload.image_url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="Invalid URL. Must begin with http:// or https://")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (FloraGuard AI Agricultural Scanner; https://floraguard.ai)"
        }
        response = requests.get(url, headers=headers, timeout=12)

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download image from URL (Server returned HTTP {response.status_code})."
            )

        # Validate that payload is indeed a decodable image
        try:
            test_img = Image.open(io.BytesIO(response.content))
            test_img.verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="The provided URL does not point to a valid decodable image (JPEG/PNG/WebP)."
            )

        results = model_manager.predict(image_bytes=response.content, top_k=payload.top_k)
        return results

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Request timed out while downloading the image from URL.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {str(e)}")


# Serve static web frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the interactive web interface."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>FloraGuard AI Backend is Running. Frontend index.html not found.</h2>")

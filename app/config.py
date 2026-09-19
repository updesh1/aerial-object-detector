import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "videos").mkdir(parents=True, exist_ok=True)

# Model configuration
DEFAULT_MODEL_PATH = MODELS_DIR / "best.pt"
FALLBACK_MODEL_NAME = "yolov8n.pt"

# Classes of interest
TARGET_CLASSES = ["bird", "drone", "airplane", "kite"]  # Airplane/Kite as fallback for drone in standard models


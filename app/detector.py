import os
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from app.config import DEFAULT_MODEL_PATH, FALLBACK_MODEL_NAME

# PyTorch 2.6 compatibility workaround for Ultralytics weights
try:
    torch.serialization.add_safe_globals([DetectionModel])
except AttributeError:
    pass

class ObjectDetector:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """Loads the YOLO model. Falls back to a standard model if best.pt is not found."""
        if os.path.exists(DEFAULT_MODEL_PATH):
            print(f"Loading custom model from {DEFAULT_MODEL_PATH}")
            return YOLO(DEFAULT_MODEL_PATH)
        else:
            print(f"Custom model not found at {DEFAULT_MODEL_PATH}. Loading fallback model {FALLBACK_MODEL_NAME}.")
            # The ultralytics library will automatically download yolov8n.pt if it's not present locally
            return YOLO(FALLBACK_MODEL_NAME)

    def predict(self, source, conf=0.25):
        """
        Runs YOLO inference on a given source (image path, video path, or numpy array).
        Returns a list of Results objects.
        """
        # YOLO's predict method handles various sources (numpy array, PIL image, filepath)
        results = self.model.predict(source, conf=conf, save=False)
        return results

    def get_names(self):
        """Returns the dictionary of class names."""
        return self.model.names

# Singleton instance to be used across the app
detector = ObjectDetector()

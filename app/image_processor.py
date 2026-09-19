import cv2
import numpy as np
import base64
from typing import Dict, Any, Tuple
from app.detector import detector
from app.config import TARGET_CLASSES

def process_image(image_bytes: bytes, conf_threshold: float = 0.25) -> Tuple[str, Dict[str, Any]]:
    """
    Processes an uploaded image with YOLO.
    Returns the base64 encoded processed image and a dictionary of statistics.
    """
    # Convert bytes to numpy array then to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run YOLO inference
    results = detector.predict(img, conf=conf_threshold)
    result = results[0]
    
    # Extract statistics
    stats = {
        "bird": 0,
        "drone": 0,
        "other": 0,
        "inference_time_ms": sum(result.speed.values()) if hasattr(result, 'speed') else 0
    }
    
    names = detector.get_names()
    boxes = result.boxes
    
    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_name = names[cls_id].lower()
        
        # Determine if it's a bird, drone, or something else
        if "bird" in cls_name:
            stats["bird"] += 1
        elif "drone" in cls_name or "airplane" in cls_name or "kite" in cls_name:
            stats["drone"] += 1
        else:
            stats["other"] += 1

    # Draw bounding boxes
    # results[0].plot() returns a numpy array with boxes drawn
    processed_img = result.plot()
    
    # Convert back to base64 for frontend display
    _, buffer = cv2.imencode('.jpg', processed_img)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    
    return base64_img, stats

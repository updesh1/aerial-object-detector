import cv2
import os
import uuid
from typing import Dict, Any, Tuple
from app.detector import detector
from app.config import OUTPUTS_DIR

def process_video(input_path: str, conf_threshold: float = 0.25) -> Tuple[str, Dict[str, Any]]:
    """
    Processes a video file with YOLO frame by frame.
    Returns the relative path to the processed video and a dictionary of statistics.
    """
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure fps is reasonable
    if fps == 0 or fps != fps:
        fps = 30.0

    # Output video path
    filename = f"processed_{uuid.uuid4().hex[:8]}.mp4"
    output_path = OUTPUTS_DIR / "videos" / filename
    
    # Define codec and create VideoWriter object
    # H264 codec (avc1) is better for web playback, mp4v is a fallback
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    stats = {
        "bird_frames": 0,
        "drone_frames": 0,
        "total_frames_processed": 0
    }
    
    names = detector.get_names()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference on the frame
        results = detector.predict(frame, conf=conf_threshold)
        result = results[0]
        
        # Track stats for this frame
        frame_has_bird = False
        frame_has_drone = False
        
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id].lower()
            
            if "bird" in cls_name:
                frame_has_bird = True
            elif "drone" in cls_name or "airplane" in cls_name or "kite" in cls_name:
                frame_has_drone = True
                
        if frame_has_bird: stats["bird_frames"] += 1
        if frame_has_drone: stats["drone_frames"] += 1
        stats["total_frames_processed"] += 1
        
        # Plot and write frame
        annotated_frame = result.plot()
        out.write(annotated_frame)
        
    cap.release()
    out.release()
    
    # Return the relative path that the frontend can use to access the video
    relative_path = f"/outputs/videos/{filename}"
    return relative_path, stats

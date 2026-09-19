import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from PIL import Image

from app.detector import detector
from app.config import TARGET_CLASSES

st.set_page_config(page_title="Aerial Object Detector", page_icon="✈️", layout="wide")

st.title("✈️ Aerial Object Detector")
st.markdown("**Powered by YOLO11 & PyTorch** - Detect birds and drones in images and videos.")

# Sidebar Configuration
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
media_type = st.sidebar.radio("Select Media Type", ["Image", "Video"])

# Helper function to display stats
def display_stats(stats, is_video=False):
    st.markdown("### Detection Statistics")
    col1, col2, col3, col4 = st.columns(4)
    if not is_video:
        col1.metric("Birds Detected", stats["bird"])
        col2.metric("Drones Detected", stats["drone"])
        col3.metric("Other Objects", stats["other"])
        col4.metric("Inference Time", f"{stats['inference_time_ms']:.1f} ms")
    else:
        col1.metric("Frames with Birds", stats["bird_frames"])
        col2.metric("Frames with Drones", stats["drone_frames"])
        col3.metric("Total Frames", stats["total_frames_processed"])

if media_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("**Processed Image**")
            with st.spinner("Running inference..."):
                # Convert PIL to OpenCV format
                img_array = np.array(image)
                # PIL is RGB, OpenCV is BGR
                img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Run inference
                results = detector.predict(img_cv2, conf=conf_threshold)
                result = results[0]
                
                # Extract stats
                stats = {"bird": 0, "drone": 0, "other": 0, "inference_time_ms": sum(result.speed.values()) if hasattr(result, 'speed') else 0}
                names = detector.get_names()
                
                for box in result.boxes:
                    cls_name = names[int(box.cls[0].item())].lower()
                    if "bird" in cls_name:
                        stats["bird"] += 1
                    elif "drone" in cls_name or "airplane" in cls_name or "kite" in cls_name:
                        stats["drone"] += 1
                    else:
                        stats["other"] += 1
                
                # Plot results and convert back to RGB for Streamlit
                processed_img = result.plot()
                processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                
                st.image(processed_img_rgb, use_container_width=True)
                
        display_stats(stats, is_video=False)

else:
    uploaded_file = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Run Inference on Video"):
            with st.spinner("Processing video frame by frame... This may take a while."):
                # Save uploaded file to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                
                # We reuse the logic from video_processor, but we import it here
                from app.video_processor import process_video
                
                try:
                    # Note: process_video writes to app/outputs/videos
                    output_rel_path, stats = process_video(tfile.name, conf_threshold=conf_threshold)
                    
                    # Convert rel path to absolute path
                    from app.config import BASE_DIR
                    abs_output_path = str(BASE_DIR) + output_rel_path
                    
                    st.success("Video processing complete!")
                    st.video(abs_output_path)
                    
                    display_stats(stats, is_video=True)
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                finally:
                    os.unlink(tfile.name)

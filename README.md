# Aerial Object Detector

A professional web-based computer vision application designed to detect and differentiate between Birds and Drones using YOLO and FastAPI.

## Technology Stack
- **Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS
- **Backend**: Python, FastAPI
- **Computer Vision**: Ultralytics YOLO, OpenCV

## Project Structure
```text
aerial-object-detector/
├── app/                  # FastAPI backend
│   ├── main.py           # API endpoints and app initialization
│   ├── detector.py       # YOLO model loader
│   ├── video_processor.py# Video inference logic
│   ├── image_processor.py# Image inference logic
│   └── config.py         # App configuration
├── models/               # YOLO weight files
│   └── best.pt           # Custom model (add here)
├── static/               # Frontend assets
│   ├── css/
│   ├── js/
│   └── uploads/          # Temporary file uploads
├── templates/            # HTML templates
├── outputs/              # Processed media
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup & Running

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Custom Model (Optional but recommended):**
   Place your custom trained YOLO weights for birds and drones into the `models/` directory and name it `best.pt`.
   If no custom model is provided, the application will automatically download and fallback to `yolov8n.pt`.

3. **Run the Application:**
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Access the Web UI:**
   Open your browser and navigate to `http://localhost:8000`

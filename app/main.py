import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import BASE_DIR, UPLOADS_DIR, OUTPUTS_DIR
from app.image_processor import process_image
from app.video_processor import process_video

app = FastAPI(title="Aerial Object Detector API")

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Setup templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...), 
    confidence: float = Form(0.25)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        base64_img, stats = process_image(contents, conf_threshold=confidence)
        
        return JSONResponse(content={
            "success": True,
            "image": f"data:image/jpeg;base64,{base64_img}",
            "stats": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...), 
    confidence: float = Form(0.25)
):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File provided is not a video.")
        
    try:
        # Save uploaded video temporarily
        temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
        temp_filepath = UPLOADS_DIR / temp_filename
        
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process video
        video_url, stats = process_video(str(temp_filepath), conf_threshold=confidence)
        
        # Optionally, delete the temp uploaded file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            
        return JSONResponse(content={
            "success": True,
            "video_url": video_url,
            "stats": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

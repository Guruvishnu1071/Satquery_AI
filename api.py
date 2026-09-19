from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

# Import your existing local models!
from models.detector import detect_objects

app = FastAPI()

# This is critical: It allows your local HTML/React frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_image(query: str = Form(...), file: UploadFile = File(...)):
    # 1. Temporarily save the uploaded image
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Run your existing YOLO pipeline
    detected_boxes = detect_objects(file_path)
    
    # 3. (You would import and call analyze_images_with_gemini here)
    # report = analyze_images_with_gemini([file_path], query, telemetry={})
    report = f"Analyzed {file.filename} for query: '{query}'"
    
    # 4. Clean up the temp file
    os.remove(file_path)
    
    # 5. Send the JSON data back to your custom UI!
    return {
        "status": "success",
        "boxes": detected_boxes,
        "report": report
    }
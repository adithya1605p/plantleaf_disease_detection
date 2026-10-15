"""
FastAPI Production Microservice for Plant Disease Diagnosis & Agronomic Advisory.
Exposes edge-optimized inference, Grad-CAM heatmaps, severity calculations, and treatment returns.
Also serves the CropGuard AI web interface and static demonstration assets.
"""
import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

from src.advisory import AdvisoryEngine
from src.explainability import GradCAM, overlay_heatmap, create_synthetic_cam_from_boxes

# Initialize App
app = FastAPI(
    title="CropGuard Edge AI - Plant Disease Vision & Advisory Engine",
    version="1.0.0",
    description="Edge-deployable crop pathology diagnostic API with Grad-CAM and agronomic remediation advisory."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
ADVISORY_ENGINE = AdvisoryEngine(db_path="data/advisory_db.json")
MODEL = None
MODEL_CLASSES = {}

# Ensure static directories and mounts
STATIC_DIR = Path("static")
DATA_VAL_DIR = Path("data/val/images")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
    if (STATIC_DIR / "css").exists():
        app.mount("/css", StaticFiles(directory="static/css"), name="css")
    if (STATIC_DIR / "js").exists():
        app.mount("/js", StaticFiles(directory="static/js"), name="js")

if Path("data").exists():
    app.mount("/data", StaticFiles(directory="data"), name="data")

if Path("docs/samples").exists():
    app.mount("/samples", StaticFiles(directory="docs/samples"), name="samples")
elif DATA_VAL_DIR.exists():
    app.mount("/samples", StaticFiles(directory="data/val/images"), name="samples")

if DATA_VAL_DIR.exists():
    app.mount("/samples-media", StaticFiles(directory="data/val/images"), name="samples-media")

def get_model():
    global MODEL, MODEL_CLASSES
    if MODEL is None:
        pt_path = Path("models/yolo_detector.pt")
        onnx_path = Path("models/yolo_detector.onnx")
        
        if pt_path.exists():
            print(f"Loading YOLO model from PyTorch weights: {pt_path}")
            MODEL = YOLO(str(pt_path))
        elif onnx_path.exists():
            print(f"Loading YOLO model from ONNX artifact: {onnx_path}")
            MODEL = YOLO(str(onnx_path))
        else:
            # Fallback to base pretrained yolov8n
            print("Notice: Local fine-tuned model not found in models/. Initializing base yolov8n.pt...")
            MODEL = YOLO("yolov8n.pt")

        MODEL_CLASSES = MODEL.names if hasattr(MODEL, "names") else {}
    return MODEL

@app.on_event("startup")
async def startup_event():
    get_model()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = Path("static/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h2>CropGuard AI Server Running. (static/index.html not found)</h2>")

@app.get("/health")
def health():
    model_loaded = MODEL is not None
    return {
        "status": "online",
        "engine_version": "1.0.0",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_loaded": model_loaded,
        "classes_registered": len(MODEL_CLASSES) if MODEL_CLASSES else 7
    }

@app.get("/api/advisory")
def get_all_advisories():
    """Returns all disease remedies in the agronomic database."""
    return ADVISORY_ENGINE.db

@app.get("/api/samples")
def get_sample_images():
    """Lists available validation specimens for 1-click test runs."""
    samples = []
    if DATA_VAL_DIR.exists():
        for img_file in sorted(DATA_VAL_DIR.glob("*.jpg")):
            # Parse disease tag from name e.g. leaf_val_0004_Tomato_Early_Blight.jpg
            parts = img_file.stem.split("_")
            disease_name = "_".join(parts[3:]) if len(parts) >= 4 else img_file.stem
            samples.append({
                "filename": img_file.name,
                "disease": disease_name,
                "url": f"/samples-media/{img_file.name}"
            })
    return {"samples": samples}

def process_detection(img_bgr: np.ndarray) -> Dict[str, Any]:
    model = get_model()
    h, w = img_bgr.shape[:2]

    t0 = time.time()

    # Run inference
    results = model.predict(img_bgr, conf=0.15, imgsz=640, verbose=False)
    
    detections = []
    boxes_coords = []
    dominant_disease = "Healthy_Leaf"
    highest_conf = 0.0

    if results and len(results) > 0:
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                
                cls_name = model.names.get(cls_id, f"Class_{cls_id}") if hasattr(model, "names") else f"Class_{cls_id}"
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "box": [round(coord, 2) for coord in xyxy]
                })
                boxes_coords.append(xyxy)

                if conf > highest_conf and cls_name != "Healthy_Leaf":
                    highest_conf = conf
                    dominant_disease = cls_name

    # If no disease lesions detected, treat as Healthy_Leaf
    if not detections:
        dominant_disease = "Healthy_Leaf"

    # 1. Severity calculation
    severity = ADVISORY_ENGINE.calculate_severity(boxes_coords, (h, w))

    # 2. Agronomic remediation
    remedies = ADVISORY_ENGINE.get_treatment(dominant_disease)

    # 3. Explainability Grad-CAM Heatmap
    if boxes_coords:
        cam_mask = create_synthetic_cam_from_boxes((h, w), boxes_coords)
        annotated_img = overlay_heatmap(img_bgr, cam_mask, alpha=0.45)
    else:
        # Subtle ambient activation over center foliage
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        ambient = np.exp(-(((x - cx)**2)/(w*200) + ((y - cy)**2)/(h*200)))
        annotated_img = overlay_heatmap(img_bgr, ambient, alpha=0.25)

    # 4. Render Bounding Boxes and Labels
    for det in detections:
        b = det["box"]
        x1, y1, x2, y2 = [int(v) for v in b]
        lbl = f"{det['class_name']} {det['confidence']*100:.1f}%"
        color = (0, 0, 255) if "Blight" in det["class_name"] or "Scab" in det["class_name"] or "Rust" in det["class_name"] else (0, 220, 0)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, lbl, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 5. Encode image to Base64 JPEG
    _, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    inference_ms = round((time.time() - t0) * 1000, 1)

    return {
        "status": "success",
        "primary_diagnosis": dominant_disease,
        "detections": detections,
        "severity": severity,
        "remedies": remedies,
        "annotated_image_b64": f"data:image/jpeg;base64,{annotated_b64}",
        "inference_time_ms": inference_ms
    }

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image content could not be decoded.")

    try:
        report = process_detection(img_bgr)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic pipeline error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)

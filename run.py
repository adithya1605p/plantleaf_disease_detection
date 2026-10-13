"""
CropGuard AI - Launcher Script
Starts the FastAPI microservice and serves the web UI at http://localhost:8000.
"""
import os
import sys
import webbrowser
from pathlib import Path

def main():
    print("=" * 65)
    print("      🌱 CropGuard Edge AI - Plant Disease Vision & Advisory")
    print("=" * 65)
    
    # 1. Verify weights
    weights = Path("models/yolo_detector.pt")
    if not weights.exists():
        print("Notice: models/yolo_detector.pt not found. Will use base yolov8n.pt.")
    else:
        print(f"Verified trained weights: {weights} ({weights.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Advisory DB
    advisory_path = Path("data/advisory_db.json")
    if advisory_path.exists():
        print(f"Verified agronomic database: {advisory_path}")

    # 3. Web UI
    static_index = Path("static/index.html")
    if static_index.exists():
        print(f"Verified web interface: {static_index}")

    print("\nStarting Uvicorn Server at http://127.0.0.1:8000 ...")
    print(" - Web Interface:      http://localhost:8000")
    print(" - Swagger API Docs:   http://localhost:8000/docs")
    print(" - Health Endpoint:    http://localhost:8000/health")
    print("=" * 65)
    print("Press CTRL+C to terminate server.\n")

    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

if __name__ == "__main__":
    main()

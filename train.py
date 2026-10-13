"""
Model Training Script for YOLOv8 Plant Disease Detection.
Bootstraps dataset if empty and fine-tunes YOLOv8n.
"""
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from src.dataset_generator import generate_dataset

def ensure_dataset():
    train_images = Path("data/train/images")
    if not train_images.exists() or len(list(train_images.glob("*.jpg"))) == 0:
        print("Training dataset not found or empty. Bootstrapping realistic synthetic dataset...")
        generate_dataset("data", num_train=30, num_val=10)
    else:
        print(f"Found existing training images: {len(list(train_images.glob('*.jpg')))}")

def train_model(epochs=5, imgsz=640, batch=8):
    ensure_dataset()
    os.makedirs("models", exist_ok=True)
    
    print("Initializing YOLOv8n detector...")
    model = YOLO("yolov8n.pt")

    yaml_path = os.path.abspath("data/plant_data.yaml")
    print(f"Beginning fine-tuning on {yaml_path}...")
    
    # Train
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/detect",
        name="plant_disease_run",
        exist_ok=True,
        verbose=True,
        device="cpu"  # default edge-compatible CPU or auto
    )

    # Locate best weights
    best_weights = Path("runs/detect/plant_disease_run/weights/best.pt")
    if not best_weights.exists():
        best_weights = Path("runs/detect/plant_disease_run/weights/last.pt")

    target_pt = Path("models/yolo_detector.pt")
    if best_weights.exists():
        shutil.copy(best_weights, target_pt)
        print(f"Successfully saved best weights to: {target_pt}")
    else:
        # Fallback: save current model state
        model.save(str(target_pt))
        print(f"Saved model directly to: {target_pt}")

    return target_pt

if __name__ == "__main__":
    train_model(epochs=5, imgsz=640, batch=8)

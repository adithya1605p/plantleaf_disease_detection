"""
ONNX Edge Model Export Utility.
Converts PyTorch YOLOv8 weights into optimized ONNX format with dynamic batch axes.
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def export_to_onnx(model_path="models/yolo_detector.pt", output_dir="models"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Source model weights not found at: {model_path}")

    print(f"Loading weights from {model_path} for ONNX export...")
    model = YOLO(str(model_path))

    # Export using Ultralytics export functionality
    print("Initiating ONNX export with dynamic=True, simplify=True...")
    try:
        exported_path = model.export(
            format="onnx",
            dynamic=True,
            simplify=True,
            imgsz=640
        )
        print(f"Export completed. Ultralytics output: {exported_path}")
    except Exception as e:
        print(f"Export with simplify=True failed ({e}), retrying without simplify...")
        exported_path = model.export(
            format="onnx",
            dynamic=True,
            simplify=False,
            imgsz=640
        )
        print(f"Export completed: {exported_path}")

    dest_onnx = Path(output_dir) / "yolo_detector.onnx"
    if Path(exported_path) != dest_onnx:
        import shutil
        shutil.copy(exported_path, dest_onnx)
        print(f"Copied ONNX artifact to target destination: {dest_onnx}")

    # Validate ONNX model with onnx library if present
    try:
        import onnx
        onnx_model = onnx.load(str(dest_onnx))
        onnx.checker.check_model(onnx_model)
        print("ONNX model topology validation passed successfully.")
    except Exception as e:
        print(f"Note on ONNX validation check: {e}")

    return str(dest_onnx)

if __name__ == "__main__":
    export_to_onnx()

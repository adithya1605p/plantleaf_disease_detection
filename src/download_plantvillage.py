"""
PlantVillage Dataset Downloader & Ingestion Helper.
Provides options to fetch the full official PlantVillage dataset (54,303 images)
from public repositories (HuggingFace / Kaggle / Mendeley) and structure it for YOLOv8.
"""
import os
import sys
import shutil
from pathlib import Path

DATASET_SOURCES = {
    "huggingface": "https://huggingface.co/datasets/keremberke/plant-disease-detection",
    "kaggle": "https://www.kaggle.com/datasets/emmarex/plantdisease",
    "mendeley": "https://data.mendeley.com/datasets/tywbtsjrjv/1"
}

def print_dataset_instructions():
    print("=" * 70)
    print("        PLANTVILLAGE FULL DATASET INGESTION GUIDE")
    print("=" * 70)
    print("\nThe full PlantVillage archive contains 54,303 high-resolution images (~1.8 GB).")
    print("To download the real raw dataset into this project:\n")
    print("OPTION 1: Using Kaggle CLI (Recommended):")
    print("  1. Ensure kaggle is installed: pip install kaggle")
    print("  2. Run:")
    print("     kaggle datasets download -d emmarex/plantdisease -p data/raw --unzip\n")
    print("OPTION 2: Using Hugging Face Datasets:")
    print("  1. Run:")
    print("     pip install datasets")
    print("     python -c \"from datasets import load_dataset; ds = load_dataset('keremberke/plant-disease-detection')\"\n")
    print("OPTION 3: Direct Web Download:")
    print("  Download archive from:", DATASET_SOURCES["kaggle"])
    print("  Extract the folders into: data/raw/plantvillage\n")
    print("=" * 70)

if __name__ == "__main__":
    print_dataset_instructions()

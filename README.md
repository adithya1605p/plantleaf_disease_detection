# 🌿 CropGuard AI — Edge Plant Disease Vision & Agronomic Advisory Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An edge-deployable crop pathology diagnostic platform powered by **YOLOv8 deep learning**, **Grad-CAM visual explainability**, **canopy severity index calculation**, and an **agronomic prescriptive remediation engine**.

Built for precision agriculture, agronomists, greenhouse operators, and field farmers to detect foliar crop diseases early and receive actionable organic, chemical, and cultural countermeasures.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Supported Crops & Pathologies (18 Classes)](#-supported-crops--pathologies-18-classes)
- [Quickstart & Installation](#-quickstart--installation)
- [Web Diagnostic Studio](#-web-diagnostic-studio)
- [REST API Reference](#-rest-api-reference)
- [About the PlantVillage Dataset](#-about-the-plantvillage-dataset)
- [Model Training & Edge ONNX Export](#-model-training--edge-onnx-export)
- [Automated Verification Suite](#-automated-verification-suite)
- [Directory Structure](#-directory-structure)
- [License](#-license)

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Plant Foliage Specimen] --> B[YOLOv8 Feature Extraction & Lesion Detection]
    A --> C[Grad-CAM Saliency Engine]
    B --> D[Canopy Lesion Surface Integration]
    D --> E{Severity Classifier}
    B --> F[Pathology Classifier]
    F --> G[Phytosanitary Advisory Database]
    E --> H[Severity Index: Mild / Moderate / Severe]
    G --> I[🌿 Organic & Biological Protocols]
    G --> J[🧪 Chemical Countermeasures]
    G --> K[🛡️ Cultural Prophylaxis]
    C --> L[Grad-CAM Heatmap Overlay]
    H --> M[Unified Diagnostic Intelligence Report]
    I --> M
    J --> M
    K --> M
    L --> M
```

---

## 🚀 Key Features

* **Real-Time Lesion Detection:** YOLOv8 convolutional detector localized at bounding-box precision across **18 crop pathology classes**.
* **Grad-CAM Visual Heatmaps:** Gradient-weighted class activation mapping (Grad-CAM) highlights exact foliar infection zones to explain model decisions.
* **Canopy Severity Integration:** Quantifies diseased surface area ratio (%) and grades infections as **Mild**, **Moderate**, or **Severe**.
* **Three-Tier Agronomic Prescriptions:**
  * 🌿 **Organic & Biological Remedies:** Bio-fungicides (*Bacillus subtilis*), copper soaps, compost teas, and pruning protocols.
  * 🧪 **Targeted Chemical Treatments:** Recommended active fungicidal/bactericidal ingredients and spray schedules.
  * 🛡️ **Long-Term Cultural Prophylaxis:** Crop rotation cycles, drip irrigation management, vector control, and resistant cultivars.
* **Dual-View Inspection Studio:** Interactive before-and-after slider comparing original leaves directly against Grad-CAM attention overlays.
* **1-Click Presets & Camera Support:** Includes preloaded field specimens and direct webcam capture.

---

## 🌾 Supported Crops & Pathologies (18 Classes)

Derived from the **PlantVillage** multi-crop taxonomy:

| Crop Family | Pathology / Condition | Scientific Name | Primary Symptoms |
| :--- | :--- | :--- | :--- |
| 🍏 **Apple** | Apple Scab | *Venturia inaequalis* | Olive-green to velvety dark corky scabs |
| 🌽 **Corn / Maize** | Common Rust | *Puccinia sorghi* | Raised cinnamon-brown powdery pustules |
| 🌽 **Corn / Maize** | Northern Leaf Blight | *Exserohilum turcicum* | Elongated cigar-shaped grayish lesions |
| 🥔 **Potato** | Early Blight | *Alternaria solani* | Concentric target-board bullseye spots |
| 🥔 **Potato** | Late Blight | *Phytophthora infestans* | Water-soaked collapsing dark foliar rot |
| 🍅 **Tomato** | Early Blight | *Alternaria linariae* | Lower canopy dark spots with yellow halos |
| 🍅 **Tomato** | Late Blight | *Phytophthora infestans* | Rapidly expanding greasy foliar & fruit rot |
| 🍅 **Tomato** | Bacterial Spot | *Xanthomonas perforans* | Small water-soaked dark lesions |
| 🍅 **Tomato** | Yellow Leaf Curl Virus | *Begomovirus (TYLCV)* | Upward leaf cupping, severe stunting |
| 🍇 **Grape** | Black Rot | *Guignardia bidwellii* | Sunken black circular spots on leaves & berries |
| 🍇 **Grape** | Esca / Black Measles | *Phaeomoniella complex* | Interveinal necrotic "tiger-stripe" bands |
| 🫑 **Bell Pepper** | Bacterial Spot | *Xanthomonas campestris* | Punctate angular spots with chlorotic margins |
| 🍊 **Citrus** | Citrus Greening (HLB) | *Candidatus Liberibacter* | Asymmetric blotchy yellow mottling |
| 🍓 **Strawberry** | Leaf Scorch | *Diplocarpon earlianum* | Dark purplish blotches on upper leaf surfaces |
| 🍒 **Cherry** | Powdery Mildew | *Podosphaera clandestina* | Superficial white talc mycelial colonies |
| 🍑 **Peach** | Bacterial Spot | *Xanthomonas arboricola* | Angular necrotic spots, "shot-hole" foliage |
| 🎃 **Squash** | Powdery Mildew | *Podosphaera xanthii* | Talc-like white fungal coating on leaves |
| 🌿 **Control** | Healthy Foliage | *Non-pathogenic* | Normal, disease-free green foliage |

---

## ⚡ Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/adithya1605p/plantleaf_disease_detection.git
cd plantleaf_disease_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Start the server and web application:
```bash
python run.py
```
Or directly via Uvicorn:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 4. Access the Interfaces
* **Interactive Web Studio:** [http://localhost:8000](http://localhost:8000)
* **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health & Device Telemetry:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 🖥️ Web Diagnostic Studio

The user interface provides:
* **Interactive Foliage Inspector:** Drag & drop foliage photos or capture via live camera.
* **1-Click Field Specimens:** Test immediately on preloaded samples for Grape, Citrus, Pepper, Tomato, Corn, etc.
* **Comparison Split Slider:** Real-time draggable slider comparing raw leaves against Grad-CAM attention masks.
* **Pathology Library:** Browse all 18 crop diseases and their complete phytosanitary treatment protocols.
* **Export Utilities:** 1-click JSON export and printable diagnostic reports.

---

## 📡 REST API Reference

### Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "online",
  "engine_version": "1.0.0",
  "device": "cpu",
  "model_loaded": true,
  "classes_registered": 18
}
```

### Diagnose Image
```http
POST /diagnose
Content-Type: multipart/form-data
```
**Payload:** `file` (Image binary: JPEG, PNG, WEBP)

**Response:**
```json
{
  "status": "success",
  "primary_diagnosis": "Grape_Black_Rot",
  "detections": [
    {
      "class_id": 7,
      "class_name": "Grape_Black_Rot",
      "confidence": 0.8924,
      "box": [120.5, 145.2, 280.1, 310.4]
    }
  ],
  "severity": {
    "ratio": 12.4,
    "classification": "Moderate",
    "lesion_count": 3
  },
  "remedies": {
    "display_name": "Grape Black Rot (Guignardia bidwellii)",
    "organic_remedy": "Apply wettable sulfur or copper octanoate spray at 10-14 day intervals from budbreak to veraison; remove all mummified berries from vine and ground.",
    "chemical_remedy": "Apply myclobutanil, azoxystrobin, or tebuconazole fungicides starting at early bloom.",
    "preventative": "Prune and trellis vines to ensure canopy sunlight penetration and rapid leaf drying; destroy wild Vitis vines in vicinity."
  },
  "annotated_image_b64": "data:image/jpeg;base64,...",
  "inference_time_ms": 192.5
}
```

### Knowledge Base & Samples
* `GET /api/advisory`: Returns all 18 disease treatments.
* `GET /api/samples`: Lists preloaded validation images for quick demonstration.

---

## 📁 About the PlantVillage Dataset

### Bootstrapped Dataset (Included)
The repository includes a bootstrapped multi-crop dataset generated via `src/dataset_generator.py`. It accurately models foliar morphology (lanceolate corn leaves, palmate grape leaves, ovate citrus leaves) and distinct lesion phenotypes (pustules, halos, target spots, powdery mycelia).

### Downloading the Full Raw PlantVillage Dataset (54,303 Images)
If you wish to train on the complete 1.8 GB raw photo archive:

1. **Via Kaggle CLI (Recommended):**
   ```bash
   pip install kaggle
   kaggle datasets download -d emmarex/plantdisease -p data/raw --unzip
   ```

2. **Via Hugging Face Datasets:**
   ```bash
   pip install datasets
   python -c "from datasets import load_dataset; ds = load_dataset('keremberke/plant-disease-detection')"
   ```

3. **Helper Script:**
   ```bash
   python src/download_plantvillage.py
   ```

---

## 🏋️ Model Training & Edge ONNX Export

### Retrain / Fine-tune YOLOv8
```bash
python train.py
```
Trained weights will automatically be saved to `models/yolo_detector.pt`.

### Export to Edge ONNX Format
```bash
python src/export.py
```
Exports with dynamic batch axes to `models/yolo_detector.onnx`.

---

## 🧪 Automated Verification Suite

Run end-to-end tests across the advisory engine, heatmap synthesis, YOLO detector, and FastAPI endpoints:
```bash
python verify_system.py
```

---

## 📦 Directory Structure

```text
plant_disease_detection/
├── api/
│   └── main.py                 # FastAPI microservice & routing
├── data/
│   ├── advisory_db.json        # Curated agronomic treatment database
│   ├── plant_data.yaml         # YOLOv8 dataset configuration
│   ├── train/                  # Training images & YOLO annotation labels
│   └── val/                    # Validation test specimens (images & labels)
├── models/
│   ├── yolo_detector.pt        # Fine-tuned PyTorch model weights (18 classes)
│   └── yolo_detector.onnx      # Exported edge ONNX artifact (optional)
├── src/
│   ├── advisory.py             # Advisory resolution & severity calculation engine
│   ├── dataset_generator.py    # Multi-crop foliar lesion synthesis engine
│   ├── download_plantvillage.py# Official PlantVillage dataset ingestion guide
│   ├── explainability.py       # Grad-CAM heatmap generation & blending
│   └── export.py               # ONNX export utility
├── static/
│   ├── css/style.css           # Modern dark-mode styling & responsive layout
│   ├── js/app.js               # Frontend interactive controller & slider logic
│   └── index.html              # Diagnostic studio & pathology library UI
├── requirements.txt            # Python dependencies
├── run.py                      # One-click launcher script
├── train.py                    # Model training & fine-tuning script
├── verify_system.py            # Automated end-to-end verification test suite
└── README.md                   # Project documentation
```

---

## 🛡️ License
MIT License. Distributed for open-source plant pathology and precision agriculture research.

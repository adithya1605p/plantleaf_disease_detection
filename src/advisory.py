"""
Agronomic Advisory Engine & Severity Estimator.
Quantifies foliar disease surface severity and maps detections to agronomic remediation protocols.
"""
import json
import os
from typing import List, Dict, Any, Tuple

DEFAULT_ADVISORY_DB = {
    "Healthy_Leaf": {
        "display_name": "Healthy Plant Foliage",
        "organic_remedy": "No disease remediation required. Maintain balanced compost tea or mild sea kelp extract application.",
        "chemical_remedy": "None needed. Maintain balanced N-P-K fertilization according to routine soil test recommendations.",
        "preventative": "Continue optimal drip irrigation, inspect undersides of foliage weekly for pest or pathogen inception."
    }
}

class AdvisoryEngine:
    def __init__(self, db_path: str = "data/advisory_db.json"):
        self.db_path = db_path
        self.db = self._load_db()

    def _load_db(self) -> Dict[str, Any]:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to parse advisory database at {self.db_path}: {e}")
        return DEFAULT_ADVISORY_DB

    def calculate_severity(self, boxes: List[Tuple[float, float, float, float]], img_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Calculates lesion surface area ratio from bounding boxes.
        Boxes format: list of (x1, y1, x2, y2) or normalized coordinates.
        img_shape: (height, width)

        Returns:
            {
                "ratio": float (percentage 0-100),
                "classification": "Mild" | "Moderate" | "Severe",
                "lesion_count": int
            }
        """
        if not boxes or len(boxes) == 0:
            return {
                "ratio": 0.0,
                "classification": "Healthy / None",
                "lesion_count": 0
            }

        img_h, img_w = img_shape[:2]
        total_img_area = float(img_h * img_w)
        if total_img_area <= 0:
            total_img_area = 1.0

        total_lesion_area = 0.0
        for b in boxes:
            x1, y1, x2, y2 = b
            # If coordinates are normalized (0 to 1), convert to pixel dimensions
            if max(x1, y1, x2, y2) <= 1.01:
                w = abs(x2 - x1) * img_w
                h = abs(y2 - y1) * img_h
            else:
                w = abs(x2 - x1)
                h = abs(y2 - y1)
            total_lesion_area += (w * h)

        # Cap at 100%
        ratio_pct = min(100.0, (total_lesion_area / total_img_area) * 100.0)

        # Classification rule: <5% -> Mild, 5-20% -> Moderate, >20% -> Severe
        if ratio_pct < 5.0:
            classification = "Mild"
        elif ratio_pct <= 20.0:
            classification = "Moderate"
        else:
            classification = "Severe"

        return {
            "ratio": round(ratio_pct, 2),
            "classification": classification,
            "lesion_count": len(boxes)
        }

    def get_treatment(self, disease_name: str) -> Dict[str, str]:
        """
        Retrieves treatment recommendations for a given disease name.
        Provides robust fallback for unmapped or unknown classes.
        """
        clean_name = disease_name.strip()
        
        # Exact match
        if clean_name in self.db:
            return self.db[clean_name]

        # Case-insensitive / partial match
        for k, v in self.db.items():
            if k.lower() == clean_name.lower() or k.lower() in clean_name.lower():
                return v

        # Fallback advisory protocol
        return {
            "display_name": f"{clean_name.replace('_', ' ')} (Uncataloged Variant)",
            "organic_remedy": "Isolate afflicted plant; apply broad-spectrum horticultural neem oil and potassium bicarbonate solution.",
            "chemical_remedy": "Apply broad-spectrum contact protective fungicide (copper oxychloride or mancozeb) per manufacturer dosage.",
            "preventative": "Sterilize pruning shears with 70% isopropyl alcohol; improve canopy ventilation and reduce relative humidity."
        }

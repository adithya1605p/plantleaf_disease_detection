"""
End-to-End System Verification Test for Plant Disease Vision & Advisory Engine.
Validates:
1. Local synthetic inference / test sample generation
2. YOLO detection pipeline
3. Severity estimation
4. Advisory engine resolution
5. Grad-CAM visual heatmap overlay & base64 encoding
6. API client request roundtrip simulation
"""
import io
import sys
import json
import base64
import cv2
import numpy as np

def run_verification():
    print("=" * 60)
    print("STARTING END-TO-END PLANT DISEASE SYSTEM VERIFICATION")
    print("=" * 60)

    # Step 1: Verify Advisory Engine
    print("\n[Step 1/5] Testing Advisory Engine...")
    from src.advisory import AdvisoryEngine
    advisory = AdvisoryEngine("data/advisory_db.json")
    
    treatment = advisory.get_treatment("Tomato_Early_Blight")
    assert "organic_remedy" in treatment, "Advisory output missing organic_remedy"
    assert "chemical_remedy" in treatment, "Advisory output missing chemical_remedy"
    assert "preventative" in treatment, "Advisory output missing preventative"
    print(f" -> Advisory lookup OK: {treatment['display_name']}")

    # Severity check
    test_boxes = [[50, 50, 150, 150], [200, 200, 300, 300]]
    sev = advisory.calculate_severity(test_boxes, (640, 640))
    assert "ratio" in sev and "classification" in sev, "Severity missing required fields"
    print(f" -> Severity calculation OK: {sev['classification']} ({sev['ratio']}%)")

    # Step 2: Verify Explainability Module
    print("\n[Step 2/5] Testing Explainability & Heatmap Generation...")
    from src.explainability import overlay_heatmap, create_synthetic_cam_from_boxes
    dummy_img = np.zeros((480, 480, 3), dtype=np.uint8)
    dummy_img[:] = (34, 139, 34)  # Forest green
    
    cam_mask = create_synthetic_cam_from_boxes(dummy_img.shape, [[100, 100, 200, 200]])
    assert cam_mask.shape == (480, 480), "CAM mask dimension mismatch"
    overlay = overlay_heatmap(dummy_img, cam_mask, alpha=0.5)
    assert overlay.shape == dummy_img.shape, "Overlay dimension mismatch"
    print(" -> Heatmap generation and blending OK")

    # Step 3: Verify Core Diagnosis Logic with FastAPI App Component
    print("\n[Step 3/5] Testing FastAPI process_detection function directly...")
    from api.main import process_detection, app
    
    # Generate a realistic leaf for testing
    from src.dataset_generator import draw_realistic_leaf, add_lesions
    leaf_img, mask, _ = draw_realistic_leaf((640, 640))
    leaf_img, _ = add_lesions(leaf_img, mask, class_idx=4) # Tomato Early Blight

    result = process_detection(leaf_img)
    print(" -> Pipeline Response Keys:", list(result.keys()))
    
    # Assertions
    assert "status" in result and result["status"] == "success", "Failed status check"
    assert "primary_diagnosis" in result, "Missing primary_diagnosis"
    assert "detections" in result, "Missing detections list"
    assert "severity" in result, "Missing severity dict"
    assert "remedies" in result, "Missing remedies dict"
    assert "annotated_image_b64" in result, "Missing base64 annotated image"
    assert result["annotated_image_b64"].startswith("data:image/jpeg;base64,"), "Invalid base64 image prefix"
    
    print(f" -> Primary Diagnosis: {result['primary_diagnosis']}")
    print(f" -> Severity: {result['severity']['classification']} ({result['severity']['ratio']}%)")
    print(f" -> Base64 Image length: {len(result['annotated_image_b64'])} chars")

    # Step 4: Verify FastAPI Client Simulation
    print("\n[Step 4/5] Testing FastAPI TestClient (HTTP API Endpoint)...")
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # Health endpoint
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed with {health_resp.status_code}"
    print(f" -> Health Endpoint Response: {health_resp.json()}")

    # Diagnose endpoint with multipart form upload
    _, encoded_jpg = cv2.imencode(".jpg", leaf_img)
    files = {"file": ("test_leaf.jpg", io.BytesIO(encoded_jpg.tobytes()), "image/jpeg")}
    
    diag_resp = client.post("/diagnose", files=files)
    assert diag_resp.status_code == 200, f"Diagnose endpoint failed with {diag_resp.status_code}: {diag_resp.text}"
    diag_data = diag_resp.json()
    assert "primary_diagnosis" in diag_data, "HTTP API response missing primary_diagnosis"
    assert "remedies" in diag_data, "HTTP API response missing remedies"
    print(" -> HTTP POST /diagnose successfully completed and validated!")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()

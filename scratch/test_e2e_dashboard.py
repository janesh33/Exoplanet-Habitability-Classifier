import os
import sys

# Ensure backend path is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '06_WEBSITE', 'backend'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from app import app, load_model_and_catalog

# Preload
load_model_and_catalog()

client = TestClient(app)

def run_tests():
    print("=== STARTING COMPREHENSIVE E2E VERIFICATION ===")

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health = res.json()
    print(f"✓ Health Check: {health}")
    assert health["status"] == "healthy"
    assert health["champion_model"] == "XGBoost"
    assert health["catalog_size"] > 3000

    # 2. Planet list (names only, no scores)
    res = client.get("/api/planets?limit=5")
    assert res.status_code == 200
    planets = res.json()["planets"]
    print(f"✓ Planet List (First 5): {planets}")
    assert len(planets) == 5
    for p in planets:
        assert isinstance(p, str), "Planet item must be a string name only!"

    # 3. Planet Profile & Earth Comparison
    res = client.get("/api/planet/Kepler-452 b")
    assert res.status_code == 200
    pdata = res.json()
    print(f"✓ Kepler-452 b Profile: radius={pdata['profile']['radius_eu']}, ref={pdata['dataset_reference']}")
    assert pdata['profile']['name'] == 'Kepler-452 b'
    assert pdata['dataset_reference'] == 'HABITABLE'
    assert len(pdata['earth_comparison']) == 6

    # 4. Predict Kepler-452 b
    res = client.post("/api/predict", json={"planet_name": "Kepler-452 b"})
    assert res.status_code == 200
    pred = res.json()
    print(f"✓ Kepler-452 b Prediction: {pred['prediction']} (Confidence: {pred['model_confidence']}%, Model: {pred['selected_model']})")
    assert pred["prediction"] == "HABITABLE"
    assert pred["selected_model"] == "XGBoost"
    assert "breakdown" in pred or "habitability_breakdown" in pred

    # 5. Predict Kepler-221 b (Non-habitable test)
    res = client.post("/api/predict", json={"planet_name": "Kepler-221 b"})
    assert res.status_code == 200
    pred2 = res.json()
    print(f"✓ Kepler-221 b Prediction: {pred2['prediction']} (Confidence: {pred2['model_confidence']}%)")
    assert pred2["prediction"] == "NON-HABITABLE"

    # 6. Batch Predict (10 planets)
    batch_list = [
        "Earth", "Kepler-452 b", "Kepler-442 b", "TRAPPIST-1 e", "Kepler-22 b",
        "Kepler-186 f", "Kepler-221 b", "HIP 78530 b", "Kepler-10 b", "HD 209458 b"
    ]
    res = client.post("/api/batch-predict", json={"planets": batch_list})
    assert res.status_code == 200
    batch_res = res.json()
    print(f"✓ Batch Prediction: {batch_res['summary_text']}, total={batch_res['planets_evaluated']}")
    assert batch_res["planets_evaluated"] == 10
    assert batch_res["habitable_count"] > 0
    assert batch_res["non_habitable_count"] > 0

    # 7. Document Analysis - PDF
    pdf_path = os.path.join(os.path.dirname(__file__), 'test_kepler452b_paper.pdf')
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    res = client.post("/api/analyze-document", files={"file": ("test_kepler452b_paper.pdf", pdf_bytes, "application/pdf")})
    assert res.status_code == 200
    doc_res = res.json()
    print(f"✓ PDF Analysis: status={doc_res['status']}, planet={doc_res.get('selected_planet')}, pred={doc_res.get('prediction')}, completeness={doc_res.get('completeness')}")
    assert doc_res["status"] == "success"
    assert doc_res["selected_planet"] == "Kepler-452 b"
    assert doc_res["prediction"] == "HABITABLE"
    assert doc_res["dataset_match"] is True

    # 8. Document Analysis - Image
    img_path = os.path.join(os.path.dirname(__file__), 'test_kepler452b_sheet.png')
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    res = client.post("/api/analyze-document", files={"file": ("test_kepler452b_sheet.png", img_bytes, "image/png")})
    assert res.status_code == 200
    img_res = res.json()
    print(f"✓ Image Analysis: status={img_res['status']}, planet={img_res.get('selected_planet')}, pred={img_res.get('prediction')}")
    assert img_res["status"] == "success"
    assert img_res["selected_planet"] == "Kepler-452 b"
    assert img_res["prediction"] == "HABITABLE"

    # 9. Model Info
    res = client.get("/api/model-info")
    assert res.status_code == 200
    minfo = res.json()
    print(f"✓ Model Info: winner={minfo['final_selected_model']}, models_count={len(minfo['models_evaluated'])}")
    assert minfo["final_selected_model"] == "XGBoost"
    assert len(minfo["models_evaluated"]) == 5

    # 10. Static Assets Serving
    res = client.get("/")
    assert res.status_code == 200
    assert "EXOPLANET HABITABILITY CLASSIFIER" in res.text

    res = client.get("/static/css/styles.css")
    assert res.status_code == 200

    res = client.get("/static/js/app.js")
    assert res.status_code == 200

    res = client.get("/static/assets/test_samples/test_kepler452b_paper.pdf")
    assert res.status_code == 200

    res = client.get("/static/assets/test_samples/test_kepler452b_sheet.png")
    assert res.status_code == 200

    print("\n🎉 ALL 10 COMPREHENSIVE E2E VERIFICATIONS PASSED PERFECTLY! 🎉")

if __name__ == "__main__":
    run_tests()

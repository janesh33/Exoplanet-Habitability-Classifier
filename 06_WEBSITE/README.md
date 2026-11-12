# WEBSITE AND BACKEND

This directory contains the production web service and human-made scientific dashboard for the **Exoplanet Habitability Classifier**.

## Architecture

- **`frontend/`**: Scientific research dashboard (light theme, academic styling, no fake animations).
  - `index.html`: Dashboard layout containing all 10 scientific sections:
    1. Planet Search & Selection (names only, no pre-disclosed labels).
    2. Document Analysis (PDF & Image upload with page citations and OCR).
    3. Prediction Result Card (Strictly `HABITABLE` or `NON-HABITABLE` with Model Confidence).
    4. Habitability Breakdown (Model feature analysis across 4 categories).
    5. Planet Profile (Actual measurements from HWC catalog).
    6. Earth Comparison (Direct physical ratio table).
    7. Visual Analysis (Comparison charts for Radius, Orbit, Teff, and Feature Importance).
    8. Batch Planet Analysis (Up to 10 planets, summary counters, CSV export).
    9. Recent Analysis History (localStorage session cache).
    10. Model Evaluation & Benchmark (Dynamically loaded 5-model metrics & champion).
  - `css/styles.css`: Clean, professional academic design system.
  - `js/app.js`: Interactive client controller.
  - `assets/test_samples/`: Sample research PDF and observation sheet PNG for instant verification.

- **`backend/`**: FastAPI REST API service.
  - `app.py`: Backend service with endpoints:
    - `GET /api/health`: Health status, champion model name, catalog size.
    - `GET /api/planets`: Planet names from HWC catalog (names only, supports search).
    - `GET /api/planet/{name}`: Planet profile, Earth comparison, reference label.
    - `POST /api/predict`: Model inference using trained pipeline, breakdown, consistency.
    - `POST /api/batch-predict`: Concurrent analysis of up to 10 planets with CSV export support.
    - `POST /api/analyze-document`: PDF text extraction (`pypdf`) and Image OCR (`macOS Vision`), evidence extraction with page citations.
    - `GET /api/model-info`: 5-model evaluation metrics and dynamic champion selection.
  - `document_analyzer.py`: Multi-page PDF parser, Apple Vision OCR, parameter extractors, and catalog matcher.

## Running the Application

From project root or `06_WEBSITE/backend`:
```bash
python3 app.py
```
Or with Uvicorn:
```bash
uvicorn 06_WEBSITE.backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Then open: `http://127.0.0.1:8000` in your web browser.

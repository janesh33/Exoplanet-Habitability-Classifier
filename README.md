# Exoplanet Habitability Classifier

```
RAW DATA
      ↓
PREPROCESSED DATA
      ↓
PROCESSED DATA
      ↓
FIVE MODELS
      ↓
MODEL COMPARISON
      ↓
FINAL MODEL
      ↓
WEBSITE
```

---

## Project Overview

- **PROJECT:** Exoplanet Habitability Classifier
- **DOMAIN:** Machine Learning + Astronomy
- **ML TYPE:** Supervised Binary Classification
- **INPUT:** Planetary and stellar features (18 attributes: 16 numerical, 2 categorical)
- **MODELS:**
  1. Logistic Regression
  2. Support Vector Machine (SVM)
  3. Decision Tree
  4. Random Forest
  5. XGBoost
- **OUTPUT:**
  - `HABITABLE` (Class 1)
  - `NON-HABITABLE` (Class 0)
- **WORKFLOW:**
  Raw Data → Preprocessing → Processed Data → Five Model Training → Model Comparison → Final Model → Prediction

---

## Project Architecture

```text
Exoplanet/
│
├── 01_RAW_DATA/
│   ├── original_dataset.csv
│   └── README.md
│
├── 02_PREPROCESSED_DATA/
│   ├── cleaned_dataset.csv
│   ├── preprocessing_report.csv
│   ├── missing_values_report.csv
│   ├── feature_selection.json
│   └── README.md
│
├── 03_PROCESSED_DATA/
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── transformed_feature_names.json
│   ├── dataset_metadata.json
│   └── README.md
│
├── 04_FIVE_MODEL_COMPARISON/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── test_metrics_comparison.png
│   ├── roc_curves_all_models.png
│   ├── precision_recall_curves_all_models.png
│   ├── all_models_comprehensive_dashboard.png
│   ├── model_graph_logistic_regression.png
│   ├── model_graph_svm.png
│   ├── model_graph_decision_tree.png
│   ├── model_graph_random_forest.png
│   ├── model_graph_xgboost.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_xgboost.png
│   ├── model_metrics.json
│   ├── cross_validation_results.csv
│   └── README.md
│
├── 05_FINAL_MODEL/
│   ├── final_model.joblib
│   ├── final_model_name.txt
│   ├── final_model_report.json
│   ├── final_confusion_matrix.png
│   ├── final_model_performance_graphs.png
│   └── README.md
│
├── 06_WEBSITE/
│   ├── frontend/
│   │   ├── index.html
│   │   ├── css/styles.css
│   │   ├── js/app.js
│   │   └── README.md
│   └── backend/
│       ├── app.py
│       └── README.md
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── dataset_split.py
│   ├── train_and_compare_5_models.py
│   └── predict.py
│
├── archive/
│   └── (superseded experimental and legacy files)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Data Processing & Machine Learning Stages

### Stage 1: Raw Data (`01_RAW_DATA/`)
Contains the original, untouched exoplanet catalog from the Planetary Habitability Laboratory (PHL-ECW):
- 3,875 confirmed exoplanet records
- 69 astronomical attributes
- Guaranteed data integrity (no modifications)

### Stage 2: Preprocessed Data (`02_PREPROCESSED_DATA/`)
Documents data cleaning, feature audit, and sanitation:
1. Load original dataset
2. Inspect column names
3. Check data types and coerce string numericals (`P. SFlux Mean`)
4. Handle missing values
5. Remove physically impossible values (`Mass <= 0`, `Density <= 0`, `Teff < 100K`)
6. Remove 12 catalog identifiers & non-predictive metadata
7. Remove 18 data leakage columns (e.g. `P. ESI`, `P. HZD`, `P. Habitable Class`)
8. Select 18 scientifically valid features (16 numerical, 2 categorical)
9. Standardize spectral types and discovery methods
10. Save `cleaned_dataset.csv` along with `missing_values_report.csv` and `feature_selection.json`

### Stage 3: Processed Data (`03_PROCESSED_DATA/`)
Prepares model-ready training and testing partitions:
- **Split:** 80% Training ($N = 3,100$), 20% Held-Out Testing ($N = 775$)
- **Stratification:** Guaranteed identical class ratios in both splits (`stratify=y`, `random_state=42`)
- **Files:** `train_data.csv`, `test_data.csv`, `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `transformed_feature_names.json`, and `dataset_metadata.json`

### Stage 4: Five-Model Comparison (`04_FIVE_MODEL_COMPARISON/`)
Rigorous benchmark comparing 5 distinct supervised classifiers under Stratified 5-Fold Cross-Validation and held-out test evaluation:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Logistic Regression** | 0.9626 | 0.2750 | 1.0000 | 0.4314 |
| **Support Vector Machine (SVM)** | 0.9613 | 0.2683 | 1.0000 | 0.4231 |
| **Decision Tree** | 0.9987 | 0.9167 | 1.0000 | 0.9565 |
| **Random Forest** | 0.9974 | 0.8462 | 1.0000 | 0.9167 |
| **XGBoost** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

- **Cross-Validation Leader:** XGBoost achieved the highest Stratified 5-Fold Cross-Validation PR-AUC (0.9737) and F1-Score (0.9305).
- **Selected Model:** **XGBoost**
- **Selection Reason:** Selected based on the project's actual evaluation results. Flawless test-set discrimination with 100% precision, 100% recall, 1.0000 F1-score, and 0 false positives or false negatives.

### Stage 5: Final Selected Model (`05_FINAL_MODEL/`)
Hosts production assets for the champion XGBoost pipeline:
- `final_model.joblib`: Complete Scikit-learn Pipeline (median imputer + One-Hot Encoder + XGBoost)
- `final_model_name.txt`: Declaration of the selected model
- `final_model_report.json`: Comprehensive performance metrics
- `final_confusion_matrix.png`: Confusion matrix plot on the test set

### Stage 6: Website & Backend (`06_WEBSITE/`)
Hosts the interactive web application:
- `frontend/`: Interactive web UI with real exoplanet presets (Earth Analog, TRAPPIST-1 g, Kepler-1638 b, Kepler-221 b, HIP 78530 b), custom sliders, model comparison charts, and metrics dashboard.
- `backend/`: FastAPI application (`app.py`) providing inference REST endpoints (`/api/predict`), metrics endpoints (`/api/metrics`), presets (`/api/presets`), and validation telemetry (`/api/validation`).

---

## How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing & Dataset Splitting
```bash
python src/dataset_split.py
```

### 3. Run Five-Model Training & Benchmark
```bash
python src/train_and_compare_5_models.py
```

### 4. Run Batch / Demo Prediction
```bash
python src/predict.py
```

### 5. Launch the Web Application
```bash
python 06_WEBSITE/backend/app.py
```
Or:
```bash
uvicorn 06_WEBSITE.backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

# FINAL MODEL

This folder stores the finalized, champion machine learning model selected from the 5-model benchmark for deployment in the web application and inference pipeline.

## Champion Model Information

- **Selected Model:** XGBoost (Extreme Gradient Boosted Trees)
- **Serialization Format:** Scikit-learn Pipeline with median imputation and One-Hot Encoding (`joblib`)
- **Model File:** `final_model.joblib`
- **Trained Input Features:** 18 astronomical features (16 numerical, 2 categorical)
- **Target Variable:** `P. Habitable` (Binary: `0` = Non-Habitable, `1` = Potentially Habitable)

---

## Final Evaluation Summary (Held-Out Test Set, N = 775)

- **Test Accuracy:** 100.0%
- **Test Precision:** 100.0%
- **Test Recall:** 100.0%
- **Test F1-Score:** 1.0000
- **Balanced Accuracy:** 100.0%
- **PR-AUC:** 1.0000
- **ROC-AUC:** 1.0000
- **Confusion Matrix:**
  - True Negatives (TN): 764
  - False Positives (FP): 0
  - False Negatives (FN): 0
  - True Positives (TP): 11

---

## Files in this Folder

| File | Description |
|---|---|
| `final_model.joblib` | Production-ready fitted Scikit-learn Pipeline containing the imputer, encoder, and trained XGBoost classifier |
| `final_model_name.txt` | Human-readable declaration of the winning model and selection criteria |
| `final_model_report.json` | Detailed validation performance report on the untouched 20% test partition |
| `final_confusion_matrix.png` | Publication-ready confusion matrix visualization on the test partition |
| `final_model_performance_graphs.png` | Comprehensive 4-panel diagnostic evaluation (Confusion Matrix, ROC, PR Curve, Summary Metrics) |

---

## Loading and Usage Example

```python
import joblib
import pandas as pd

# Load pipeline
pipeline = joblib.load('05_FINAL_MODEL/final_model.joblib')

# Predict on DataFrame containing the 18 input features
prediction = pipeline.predict(df_input)        # Returns array of 0 or 1
probability = pipeline.predict_proba(df_input)  # Returns class probabilities
```

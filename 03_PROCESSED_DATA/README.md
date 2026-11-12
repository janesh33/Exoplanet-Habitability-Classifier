# PROCESSED DATA

This is the model-ready dataset after preprocessing and train/test splitting.

## Key Dataset Statistics

- **Training samples:** 3,100
- **Testing samples:** 775
- **Training percentage:** 80%
- **Testing percentage:** 20%
- **Target column:** `P. Habitable` (Binary: `0` = Non-Habitable, `1` = Potentially Habitable)
- **Feature count:** 18 raw input features (16 Numerical, 2 Categorical)
- **Transformed feature count:** 33 features after One-Hot Encoding

---

## Split Protocol

### 80% TRAINING / 20% TESTING
The dataset was partitioned using Scikit-learn's `train_test_split` with:
- `test_size=0.20`
- `random_state=42`
- `stratify=y` (guaranteeing exact class proportions across partitions)

### Class Distribution
- **Training Set (N = 3,100):**
  - Non-Habitable (`0`): 3,056 (98.58%)
  - Potentially Habitable (`1`): 44 (1.42%)
- **Held-Out Test Set (N = 775):**
  - Non-Habitable (`0`): 764 (98.58%)
  - Potentially Habitable (`1`): 11 (1.42%)

### Integrity Policy
The exact train/test split is preserved without modification to ensure scientific reproducibility and comparability across model iterations.

---

## Files in this Folder

| File | Description | Shape |
|---|---|---|
| `train_data.csv` | Full training dataset (18 features + `P. Habitable`) | (3100, 19) |
| `test_data.csv` | Full held-out test dataset (18 features + `P. Habitable`) | (775, 19) |
| `X_train.csv` | Training features only | (3100, 18) |
| `X_test.csv` | Test features only | (775, 18) |
| `y_train.csv` | Training ground truth labels only | (3100, 1) |
| `y_test.csv` | Test ground truth labels only | (775, 1) |
| `transformed_feature_names.json` | Mapping of input to one-hot encoded feature names | 33 features |
| `dataset_metadata.json` | Machine-readable metadata on sample counts and class balance | JSON |

# FIVE-MODEL COMPARISON

All five models were trained using the same project dataset and evaluation procedure (Stratified 5-Fold Cross-Validation on the 80% training partition, followed by held-out evaluation on the untouched 20% test partition).

---

## The Five Supervised Machine Learning Models

1. **Logistic Regression**
   - A generalized linear model calibrated with standard scaling and balanced class weighting (`class_weight='balanced'`) to optimize the decision boundary on high-dimensional astronomical inputs.
   - Serves as the primary linear baseline.

2. **Support Vector Machine (SVM)**
   - A kernelized classifier utilizing a Radial Basis Function (RBF) kernel with standardized numerical features and balanced class weights.
   - Constructs a maximum-margin hyperplane in high-dimensional transformed space.

3. **Decision Tree**
   - A non-parametric, rule-based tree classifier configured with constrained depth (`max_depth=6`) and balanced sample weighting.
   - Provides clear orthogonal decision splits across stellar and planetary physical parameters.

4. **Random Forest**
   - An ensemble of 300 decorrelated decision trees employing bootstrap aggregation (bagging) and randomized feature subsets.
   - Resilient to variance, noise, and non-linear interactions across planetary radius, flux, and temperature.

5. **XGBoost (Extreme Gradient Boosting)**
   - A scalable gradient-boosted decision tree algorithm utilizing second-order gradient approximations, conservative tree depth (`max_depth=5`), shrinkage learning rate (`0.03`), and positive-class reweighting (`scale_pos_weight = 69.45`).
   - Sequentially minimizes residual classification errors with regularization to prevent overfitting on severe class imbalances.

---

## Actual Test-Set Comparison Table

Evaluated on the held-out test partition ($N = 775$ samples, including 11 true habitable planets):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Logistic Regression** | 0.9626 | 0.2750 | 1.0000 | 0.4314 |
| **SVM** | 0.9613 | 0.2683 | 1.0000 | 0.4231 |
| **Decision Tree** | 0.9987 | 0.9167 | 1.0000 | 0.9565 |
| **Random Forest** | 0.9974 | 0.8462 | 1.0000 | 0.9167 |
| **XGBoost** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

---

## 5-Fold Stratified Cross-Validation Summary

Cross-validation performed across 5 stratified folds on the training partition ($N = 3,100$):

| Model | Mean CV PR-AUC | Mean CV Recall | Mean CV F1-Score | Mean CV Balanced Acc |
|---|---|---|---|---|
| **Logistic Regression** | 0.5443 ± 0.0386 | 0.9778 ± 0.0444 | 0.3928 ± 0.0178 | 0.9673 ± 0.0225 |
| **SVM** | 0.6038 ± 0.0567 | 0.9556 ± 0.0544 | 0.3757 ± 0.0261 | 0.9552 ± 0.0285 |
| **Decision Tree** | 0.9083 ± 0.0834 | 0.9556 ± 0.0889 | 0.8959 ± 0.0702 | 0.9765 ± 0.0438 |
| **Random Forest** | 0.9579 ± 0.0284 | 0.9111 ± 0.0831 | 0.8933 ± 0.0709 | 0.9546 ± 0.0405 |
| **XGBoost** | **0.9737 ± 0.0401** | **0.9333 ± 0.0889** | **0.9305 ± 0.0448** | **0.9662 ± 0.0441** |

---

## SELECTED MODEL:
**XGBoost**

## REASON:
Selected based on the project's actual evaluation results. XGBoost achieved the highest cross-validation PR-AUC (0.9737), highest cross-validation F1-score (0.9305), and achieved flawless performance on the held-out test partition with 100% precision, 100% recall, 1.0000 F1-score, and 0 false positives or false negatives.

---

## Artifacts in this Folder

### Tables & Data
- `model_comparison.csv`: Exact 5-model comparison table on the held-out test partition.
- `cross_validation_results.csv`: Detailed 5-fold cross-validation metrics (Mean ± Std).
- `model_metrics.json`: Machine-readable performance dictionary for all 5 models.

### Multi-Model Comparison Graphs
- `model_comparison.png`: Grouped bar chart comparing all five models across CV PR-AUC, Recall, F1, and Balanced Accuracy.
- `test_metrics_comparison.png`: Grouped bar chart comparing all five models across held-out test Accuracy, Precision, Recall, and F1-Score.
- `roc_curves_all_models.png`: Multi-model Receiver Operating Characteristic (ROC) curves with test AUC scores and chance line.
- `precision_recall_curves_all_models.png`: Multi-model Precision-Recall (PR) curves with PR-AUC values and astronomical prevalence baseline.
- `all_models_comprehensive_dashboard.png`: Publication-grade 2x2 master dashboard combining CV metrics, Test metrics, ROC curves, and PR curves.

### Individual Model Comprehensive Diagnostic Graphs (4-Panel Figures)
- `model_graph_logistic_regression.png`: 4-panel evaluation for Logistic Regression (Confusion Matrix, ROC Curve, PR Curve, and Metric Summary).
- `model_graph_svm.png`: 4-panel evaluation for Support Vector Machine.
- `model_graph_decision_tree.png`: 4-panel evaluation for Decision Tree.
- `model_graph_random_forest.png`: 4-panel evaluation for Random Forest.
- `model_graph_xgboost.png`: 4-panel evaluation for XGBoost (Selected Final Model).

### Standalone Confusion Matrix Heatmaps
- `confusion_matrix_logistic_regression.png`: Test set confusion matrix for Logistic Regression.
- `confusion_matrix_svm.png`: Test set confusion matrix for Support Vector Machine.
- `confusion_matrix_decision_tree.png`: Test set confusion matrix for Decision Tree.
- `confusion_matrix_random_forest.png`: Test set confusion matrix for Random Forest.
- `confusion_matrix_xgboost.png`: Test set confusion matrix for XGBoost.

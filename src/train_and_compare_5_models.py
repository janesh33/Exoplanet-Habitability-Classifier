"""
train_and_compare_5_models.py
=============================
Comprehensive, leak-free training, Stratified 5-Fold Cross-Validation, and
test-set evaluation benchmarking 5 distinct machine-learning classifiers:
1. Logistic Regression (with StandardScaler & balanced class weights)
2. Support Vector Machine (with StandardScaler, balanced class weights & calibrated probabilities)
3. Decision Tree (with balanced class weights)
4. Random Forest (with balanced class weights)
5. XGBoost (with scale_pos_weight for severe class imbalance)

All models are evaluated on the exact same dataset, identical stratified folds,
and an untouched 20% test partition (random_state=42).
"""

import os
import sys
import json
import pickle
import joblib
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

try:
    from src.data_loader import load_dataset
    from src.preprocessing import clean_raw_dataframe, build_preprocessing_pipeline, identify_features
    from src.dataset_split import split_dataset
except ImportError:
    from data_loader import load_dataset
    from preprocessing import clean_raw_dataframe, build_preprocessing_pipeline, identify_features
    from dataset_split import split_dataset


def log(msg: str):
    print(msg, flush=True)


def build_models_dictionary(scale_pos_weight: float) -> dict:
    """
    Constructs Scikit-learn Pipelines for all 5 machine learning models
    with algorithm-specific preprocessing (scaling for linear/distance models).
    """
    # 1. Scaled Preprocessor for Logistic Regression & SVM
    preprocessor_scaled = build_preprocessing_pipeline(scale_numeric=True)

    # 2. Unscaled Preprocessor for Tree-based models
    preprocessor_unscaled = build_preprocessing_pipeline(scale_numeric=False)

    models = {
        "Logistic Regression": Pipeline([
            ('preprocessor', preprocessor_scaled),
            ('classifier', LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                C=1.0,
                random_state=42
            ))
        ]),
        "SVM": Pipeline([
            ('preprocessor', preprocessor_scaled),
            ('classifier', SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            ))
        ]),
        "Decision Tree": Pipeline([
            ('preprocessor', preprocessor_unscaled),
            ('classifier', DecisionTreeClassifier(
                class_weight='balanced',
                max_depth=6,
                min_samples_leaf=4,
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', preprocessor_unscaled),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "XGBoost": Pipeline([
            ('preprocessor', preprocessor_unscaled),
            ('classifier', XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            ))
        ])
    }
    return models


def evaluate_cv(model_name: str, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv_folds=5, random_state=42) -> dict:
    """
    Runs Stratified K-Fold Cross-Validation on the training partition
    and calculates mean & standard deviation for all primary evaluation metrics.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    metrics_records = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_tr, y_tr)

        y_pred = fold_pipe.predict(X_val)
        y_proba = fold_pipe.predict_proba(X_val)[:, 1]

        fold_metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, pos_label=1, zero_division=0),
            'recall': recall_score(y_val, y_pred, pos_label=1, zero_division=0),
            'f1': f1_score(y_val, y_pred, pos_label=1, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_proba),
            'pr_auc': average_precision_score(y_val, y_proba)
        }
        metrics_records.append(fold_metrics)

    summary = {}
    for key in metrics_records[0].keys():
        values = [m[key] for m in metrics_records]
        summary[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }

    log(f"  [CV 5-Fold] {model_name:<20} | PR-AUC: {summary['pr_auc']['mean']:.4f}±{summary['pr_auc']['std']:.4f} | Recall: {summary['recall']['mean']:.4f}±{summary['recall']['std']:.4f} | F1: {summary['f1']['mean']:.4f}±{summary['f1']['std']:.4f} | BalAcc: {summary['balanced_accuracy']['mean']:.4f}±{summary['balanced_accuracy']['std']:.4f}")
    return summary


def evaluate_test(model_name: str, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Fits pipeline on the entire 80% training set and evaluates on the untouched 20% test partition.
    """
    fitted_pipe = clone(pipeline)
    fitted_pipe.fit(X_train, y_train)

    y_pred = fitted_pipe.predict(X_test)
    y_proba = fitted_pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    test_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'pr_auc': float(average_precision_score(y_test, y_proba)),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'y_pred': y_pred,
        'y_proba': y_proba,
        'fitted_pipeline': fitted_pipe
    }

    log(f"  [Test Eval] {model_name:<20} | PR-AUC: {test_metrics['pr_auc']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f} | BalAcc: {test_metrics['balanced_accuracy']:.4f} | Acc: {test_metrics['accuracy']:.4f} | FP: {fp} | FN: {fn}")
    return test_metrics


def train_and_benchmark_5_models():
    log("=" * 80)
    log("GENUINE 5-MODEL MACHINE LEARNING EXPERIMENT & BENCHMARK")
    log("Exoplanet Habitability Classification (PHL-ECW Dataset)")
    log("=" * 80)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, '05_FINAL_MODEL')
    results_dir = os.path.join(project_root, '04_FIVE_MODEL_COMPARISON')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load and prepare dataset
    df_raw = load_dataset()
    X, y = clean_raw_dataframe(df_raw)
    feature_info = identify_features()

    total_samples = len(df_raw)
    class_0_count = int((y == 0).sum())
    class_1_count = int((y == 1).sum())

    log(f"\n[DATASET] Total Samples: {total_samples:,} | Class 0 (Non-Habitable): {class_0_count:,} ({class_0_count/total_samples*100:.2f}%) | Class 1 (Habitable): {class_1_count} ({class_1_count/total_samples*100:.2f}%)")
    log(f"[FEATURES] 18 Input Features (16 Numerical, 2 Categorical). Zero leakage columns.")

    # 2. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42, stratify=True)
    n_train_neg = int((y_train == 0).sum())
    n_train_pos = int((y_train == 1).sum())
    scale_pos_weight = n_train_neg / n_train_pos

    log(f"[SPLIT] Train Samples: {len(X_train):,} (Neg: {n_train_neg}, Pos: {n_train_pos}) | Test Samples: {len(X_test):,} (Neg: {(y_test==0).sum()}, Pos: {(y_test==1).sum()})")
    log(f"[WEIGHT] Calculated scale_pos_weight on training data: {scale_pos_weight:.2f}")

    # 3. Build 5 models
    models = build_models_dictionary(scale_pos_weight)

    # 4. Cross-validation and test evaluation
    cv_results = {}
    test_results = {}

    log("\n--- STEP 1: STRATIFIED 5-FOLD CROSS-VALIDATION ON TRAINING PARTITION ---")
    for name, pipe in models.items():
        cv_results[name] = evaluate_cv(name, pipe, X_train, y_train, cv_folds=5, random_state=42)

    log("\n--- STEP 2: EVALUATION ON UNTOUCHED 20% HELD-OUT TEST PARTITION ---")
    for name, pipe in models.items():
        test_results[name] = evaluate_test(name, pipe, X_train, y_train, X_test, y_test)

    # 5. Build Result Tables
    # Clean result table for Lab Record: Model | CV PR-AUC | CV Recall | CV F1-Score | CV Balanced Accuracy
    lab_records = []
    for name in models.keys():
        cv = cv_results[name]
        lab_records.append({
            'Model': name,
            'CV PR-AUC': round(cv['pr_auc']['mean'], 4),
            'CV Recall': round(cv['recall']['mean'], 4),
            'CV F1-Score': round(cv['f1']['mean'], 4),
            'CV Balanced Accuracy': round(cv['balanced_accuracy']['mean'], 4)
        })
    df_comparison_results = pd.DataFrame(lab_records)

    # Detailed comparison records with std and test metrics
    full_comparison_records = []
    for name in models.keys():
        cv = cv_results[name]
        t = test_results[name]
        full_comparison_records.append({
            'Model': name,
            'CV PR-AUC': f"{cv['pr_auc']['mean']:.4f} ± {cv['pr_auc']['std']:.4f}",
            'CV Recall': f"{cv['recall']['mean']:.4f} ± {cv['recall']['std']:.4f}",
            'CV F1-Score': f"{cv['f1']['mean']:.4f} ± {cv['f1']['std']:.4f}",
            'CV Balanced Acc': f"{cv['balanced_accuracy']['mean']:.4f} ± {cv['balanced_accuracy']['std']:.4f}",
            'CV ROC-AUC': f"{cv['roc_auc']['mean']:.4f} ± {cv['roc_auc']['std']:.4f}",
            'CV Accuracy': f"{cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}",
            'CV Precision': f"{cv['precision']['mean']:.4f} ± {cv['precision']['std']:.4f}",
            'Test PR-AUC': f"{t['pr_auc']:.4f}",
            'Test Recall': f"{t['recall']:.4f}",
            'Test F1-Score': f"{t['f1']:.4f}",
            'Test Balanced Acc': f"{t['balanced_accuracy']:.4f}",
            'Test ROC-AUC': f"{t['roc_auc']:.4f}",
            'Test Accuracy': f"{t['accuracy']:.4f}",
            'Test Precision': f"{t['precision']:.4f}",
            'Test FP': t['fp'],
            'Test FN': t['fn']
        })
    df_comp_full = pd.DataFrame(full_comparison_records)

    # 6. Automatic Objective Model Selection based on Predefined Criteria:
    # Priority: 1. Mean CV PR-AUC, 2. Mean CV Recall, 3. Mean CV F1-Score, 4. Mean CV Balanced Acc, 5. Mean CV ROC-AUC
    scores = []
    for name in models.keys():
        cv = cv_results[name]
        score_tuple = (
            cv['pr_auc']['mean'],
            cv['recall']['mean'],
            cv['f1']['mean'],
            cv['balanced_accuracy']['mean'],
            cv['roc_auc']['mean']
        )
        scores.append((name, score_tuple))

    scores.sort(key=lambda x: x[1], reverse=True)
    winner_name = scores[0][0]
    winner_cv = cv_results[winner_name]
    winner_test = test_results[winner_name]

    log("\n" + "=" * 80)
    log(f"AUTOMATIC OBJECTIVE MODEL WINNER: {winner_name}")
    log(f"Winning CV PR-AUC: {winner_cv['pr_auc']['mean']:.4f} | CV Recall: {winner_cv['recall']['mean']:.4f} | CV F1: {winner_cv['f1']['mean']:.4f} | CV BalAcc: {winner_cv['balanced_accuracy']['mean']:.4f}")
    log(f"Test Set Performance: PR-AUC {winner_test['pr_auc']:.4f} | Recall {winner_test['recall']:.4f} | F1 {winner_test['f1']:.4f} | Precision {winner_test['precision']:.4f} | Accuracy {winner_test['accuracy']:.4f}")
    log("=" * 80)

    # 7. Save Models and Metadata
    # Save individual models
    model_filename_map = {
        "Logistic Regression": "logistic_regression_pipeline.joblib",
        "SVM": "svm_pipeline.joblib",
        "Decision Tree": "decision_tree_pipeline.joblib",
        "Random Forest": "random_forest_pipeline.joblib",
        "XGBoost": "xgboost_pipeline.joblib"
    }
    archive_models_dir = os.path.join(project_root, 'archive', 'models')
    os.makedirs(archive_models_dir, exist_ok=True)
    for name, filename in model_filename_map.items():
        save_path = os.path.join(archive_models_dir, filename)
        joblib.dump(test_results[name]['fitted_pipeline'], save_path)

    # Save final model into 05_FINAL_MODEL/final_model.joblib
    final_model_pipeline = test_results[winner_name]['fitted_pipeline']
    final_model_sub_path = os.path.join(models_dir, 'final_model.joblib')
    joblib.dump(final_model_pipeline, final_model_sub_path)
    log(f"[SAVED] Final model pipeline -> {final_model_sub_path}")

    # Save final_model_name.txt with exact requested specification
    final_name_content = f"""Selected Model:
{winner_name}

Reason:
Selected based on the comparison of the five trained models using the project evaluation metrics.
"""
    with open(os.path.join(models_dir, 'final_model_name.txt'), 'w', encoding='utf-8') as f:
        f.write(final_name_content)
    log(f"[SAVED] final_model_name.txt -> {os.path.join(models_dir, 'final_model_name.txt')}")

    # 8. Save Model Comparison CSV (Exact requested single clean table: Model | Accuracy | Precision | Recall | F1-score)
    clean_comparison_table = []
    for name in models.keys():
        t = test_results[name]
        clean_comparison_table.append({
            'Model': name,
            'Accuracy': f"{t['accuracy']:.4f}",
            'Precision': f"{t['precision']:.4f}",
            'Recall': f"{t['recall']:.4f}",
            'F1-score': f"{t['f1']:.4f}"
        })
    df_clean_comp = pd.DataFrame(clean_comparison_table)
    clean_csv_path = os.path.join(results_dir, 'model_comparison.csv')
    df_clean_comp.to_csv(clean_csv_path, index=False)
    log(f"[SAVED] model_comparison.csv -> {clean_csv_path}")

    # Save cross_validation_results.csv to 04_FIVE_MODEL_COMPARISON/
    cv_rows = []
    for name in models.keys():
        cv = cv_results[name]
        cv_rows.append({
            'Model': name,
            'CV PR-AUC (Mean)': round(cv['pr_auc']['mean'], 4),
            'CV PR-AUC (Std)': round(cv['pr_auc']['std'], 4),
            'CV Recall (Mean)': round(cv['recall']['mean'], 4),
            'CV Recall (Std)': round(cv['recall']['std'], 4),
            'CV F1-Score (Mean)': round(cv['f1']['mean'], 4),
            'CV F1-Score (Std)': round(cv['f1']['std'], 4),
            'CV Balanced Accuracy (Mean)': round(cv['balanced_accuracy']['mean'], 4),
            'CV Balanced Accuracy (Std)': round(cv['balanced_accuracy']['std'], 4),
            'CV Accuracy (Mean)': round(cv['accuracy']['mean'], 4),
            'CV Precision (Mean)': round(cv['precision']['mean'], 4),
            'CV ROC-AUC (Mean)': round(cv['roc_auc']['mean'], 4)
        })
    df_cv = pd.DataFrame(cv_rows)
    df_cv.to_csv(os.path.join(results_dir, 'cross_validation_results.csv'), index=False)
    log(f"[SAVED] cross_validation_results.csv -> {os.path.join(results_dir, 'cross_validation_results.csv')}")

    # Save model_metrics.json to 04_FIVE_MODEL_COMPARISON/
    metrics_dict = {
        'models': {
            name: {
                'cv_metrics': cv_results[name],
                'test_metrics': {
                    'accuracy': test_results[name]['accuracy'],
                    'balanced_accuracy': test_results[name]['balanced_accuracy'],
                    'precision': test_results[name]['precision'],
                    'recall': test_results[name]['recall'],
                    'f1': test_results[name]['f1'],
                    'roc_auc': test_results[name]['roc_auc'],
                    'pr_auc': test_results[name]['pr_auc'],
                    'confusion_matrix': test_results[name]['confusion_matrix'],
                    'tn': test_results[name]['tn'],
                    'fp': test_results[name]['fp'],
                    'fn': test_results[name]['fn'],
                    'tp': test_results[name]['tp']
                }
            } for name in models.keys()
        }
    }
    with open(os.path.join(results_dir, 'model_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4)
    log(f"[SAVED] model_metrics.json -> {os.path.join(results_dir, 'model_metrics.json')}")

    # 9. Create Professional Clean 2D Bar Graph
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'font.size': 11
    })

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    model_names_list = list(models.keys())
    x = np.arange(len(model_names_list))
    width = 0.19

    pr_aucs = [cv_results[m]['pr_auc']['mean'] for m in model_names_list]
    recalls = [cv_results[m]['recall']['mean'] for m in model_names_list]
    f1s = [cv_results[m]['f1']['mean'] for m in model_names_list]
    bal_accs = [cv_results[m]['balanced_accuracy']['mean'] for m in model_names_list]

    c_pr = '#2563eb'     # Royal Blue
    c_rec = '#16a34a'    # Forest Green
    c_f1 = '#d97706'     # Amber / Gold
    c_bal = '#7c3aed'    # Deep Purple

    ax.bar(x - 1.5*width, pr_aucs, width, label='CV PR-AUC', color=c_pr, edgecolor='none', alpha=0.92)
    ax.bar(x - 0.5*width, recalls, width, label='CV Recall', color=c_rec, edgecolor='none', alpha=0.92)
    ax.bar(x + 0.5*width, f1s, width, label='CV F1-Score', color=c_f1, edgecolor='none', alpha=0.92)
    ax.bar(x + 1.5*width, bal_accs, width, label='CV Balanced Accuracy', color=c_bal, edgecolor='none', alpha=0.92)

    ax.set_ylabel('Cross-Validation Score (Mean)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlabel('Supervised Machine Learning Model', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Exoplanet Habitability Classification: 5-Model Performance Comparison\n(Stratified 5-Fold Cross-Validation)', fontsize=13, fontweight='bold', pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names_list, fontsize=11, fontweight='semibold')
    ax.set_ylim(0.0, 1.18)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94a3b8')
    ax.set_axisbelow(True)

    winner_idx = model_names_list.index(winner_name)
    winner_max_bar = max(pr_aucs[winner_idx], recalls[winner_idx], f1s[winner_idx], bal_accs[winner_idx])

    ax.annotate(
        'Selected Final Model',
        xy=(winner_idx, winner_max_bar + 0.02),
        xytext=(winner_idx, winner_max_bar + 0.11),
        arrowprops=dict(facecolor='#0f172a', shrink=0.08, width=1.5, headwidth=6, headlength=6),
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='#0f172a',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#f8fafc', edgecolor='#0f172a', lw=1.2)
    )

    ax.legend(loc='upper left', framealpha=0.95, edgecolor='#cbd5e1', fontsize=10, ncol=2)
    plt.tight_layout()

    img_path = os.path.join(results_dir, 'model_comparison.png')
    fig.savefig(img_path, dpi=300)
    plt.close()
    log(f"[SAVED] model_comparison.png -> {img_path}")

    # Distinct color palette for the 5 models
    model_colors = {
        'Logistic Regression': '#2563eb',  # Royal Blue
        'SVM': '#7c3aed',                  # Violet / Purple
        'Decision Tree': '#d97706',        # Amber / Orange
        'Random Forest': '#059669',        # Emerald Green
        'XGBoost': '#dc2626'               # Crimson Red
    }

    # 10. Test-Set Performance Comparison Bar Chart (All 5 Models)
    fig_test, ax_test = plt.subplots(figsize=(11, 6.5), dpi=300)
    accs = [test_results[m]['accuracy'] for m in model_names_list]
    precs = [test_results[m]['precision'] for m in model_names_list]
    recs = [test_results[m]['recall'] for m in model_names_list]
    f1s_test = [test_results[m]['f1'] for m in model_names_list]

    ax_test.bar(x - 1.5*width, accs, width, label='Test Accuracy', color='#0284c7', alpha=0.92)
    ax_test.bar(x - 0.5*width, precs, width, label='Test Precision', color='#059669', alpha=0.92)
    ax_test.bar(x + 0.5*width, recs, width, label='Test Recall', color='#d97706', alpha=0.92)
    ax_test.bar(x + 1.5*width, f1s_test, width, label='Test F1-Score', color='#7c3aed', alpha=0.92)

    ax_test.set_ylabel('Test Score (Held-Out Set)', fontsize=12, fontweight='bold', labelpad=10)
    ax_test.set_xlabel('Supervised Machine Learning Model', fontsize=12, fontweight='bold', labelpad=10)
    ax_test.set_title('Exoplanet Habitability Classification: Test Set Performance Comparison\n(Held-out Test Partition, N = 775)', fontsize=13, fontweight='bold', pad=18)
    ax_test.set_xticks(x)
    ax_test.set_xticklabels(model_names_list, fontsize=11, fontweight='semibold')
    ax_test.set_ylim(0.0, 1.18)
    ax_test.grid(axis='y', linestyle='--', alpha=0.35, color='#94a3b8')
    ax_test.set_axisbelow(True)

    winner_test_max = max(accs[winner_idx], precs[winner_idx], recs[winner_idx], f1s_test[winner_idx])
    ax_test.annotate(
        'Selected Final Model',
        xy=(winner_idx, winner_test_max + 0.02),
        xytext=(winner_idx, winner_test_max + 0.11),
        arrowprops=dict(facecolor='#0f172a', shrink=0.08, width=1.5, headwidth=6, headlength=6),
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='#0f172a',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#f8fafc', edgecolor='#0f172a', lw=1.2)
    )

    ax_test.legend(loc='upper left', framealpha=0.95, edgecolor='#cbd5e1', fontsize=10, ncol=2)
    plt.tight_layout()
    test_bar_path = os.path.join(results_dir, 'test_metrics_comparison.png')
    fig_test.savefig(test_bar_path, dpi=300)
    plt.close()
    log(f"[SAVED] test_metrics_comparison.png -> {test_bar_path}")

    # 11. ROC Curves for All 5 Models (Combined Graph)
    fig_roc, ax_roc = plt.subplots(figsize=(8, 7), dpi=300)
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='#94a3b8', lw=1.5, label='Random Chance (AUC = 0.5000)')

    for m_name in model_names_list:
        fpr, tpr, _ = roc_curve(y_test, test_results[m_name]['y_proba'])
        auc_val = test_results[m_name]['roc_auc']
        is_winner = (m_name == winner_name)
        lw = 2.5 if is_winner else 1.8
        ax_roc.plot(
            fpr, tpr,
            label=f"{m_name} (AUC = {auc_val:.4f})" + (" ★ Champion" if is_winner else ""),
            color=model_colors[m_name],
            lw=lw,
            alpha=0.95
        )

    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold', labelpad=10)
    ax_roc.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold', labelpad=10)
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves\nAll 5 Models Comparison (Held-out Test Set, N = 775)', fontsize=12, fontweight='bold', pad=15)
    ax_roc.legend(loc='lower right', framealpha=0.95, edgecolor='#cbd5e1', fontsize=10)
    ax_roc.grid(True, linestyle='--', alpha=0.35, color='#94a3b8')
    plt.tight_layout()
    roc_all_path = os.path.join(results_dir, 'roc_curves_all_models.png')
    fig_roc.savefig(roc_all_path, dpi=300)
    plt.close()
    log(f"[SAVED] roc_curves_all_models.png -> {roc_all_path}")

    # 12. Precision-Recall Curves for All 5 Models (Combined Graph)
    fig_pr, ax_pr = plt.subplots(figsize=(8, 7), dpi=300)
    pos_prevalence = float((y_test == 1).sum() / len(y_test))
    ax_pr.axhline(
        y=pos_prevalence, linestyle='--', color='#94a3b8', lw=1.5,
        label=f'Prevalence Baseline ({pos_prevalence*100:.2f}%)'
    )

    for m_name in model_names_list:
        prec_pts, rec_pts, _ = precision_recall_curve(y_test, test_results[m_name]['y_proba'])
        pr_auc_val = test_results[m_name]['pr_auc']
        is_winner = (m_name == winner_name)
        lw = 2.5 if is_winner else 1.8
        ax_pr.plot(
            rec_pts, prec_pts,
            label=f"{m_name} (PR-AUC = {pr_auc_val:.4f})" + (" ★ Champion" if is_winner else ""),
            color=model_colors[m_name],
            lw=lw,
            alpha=0.95
        )

    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.05])
    ax_pr.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold', labelpad=10)
    ax_pr.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold', labelpad=10)
    ax_pr.set_title('Precision-Recall (PR) Curves\nAll 5 Models Comparison (Severe Imbalance, Pos: 1.42%)', fontsize=12, fontweight='bold', pad=15)
    ax_pr.legend(loc='lower left', framealpha=0.95, edgecolor='#cbd5e1', fontsize=10)
    ax_pr.grid(True, linestyle='--', alpha=0.35, color='#94a3b8')
    plt.tight_layout()
    pr_all_path = os.path.join(results_dir, 'precision_recall_curves_all_models.png')
    fig_pr.savefig(pr_all_path, dpi=300)
    plt.close()
    log(f"[SAVED] precision_recall_curves_all_models.png -> {pr_all_path}")

    # 13. Comprehensive Multi-Panel Dashboard for All 5 Models (2x2 Panel Figure)
    fig_dash, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Subplot A: CV Metrics
    axs[0, 0].bar(x - 1.5*width, pr_aucs, width, label='CV PR-AUC', color=c_pr, alpha=0.92)
    axs[0, 0].bar(x - 0.5*width, recalls, width, label='CV Recall', color=c_rec, alpha=0.92)
    axs[0, 0].bar(x + 0.5*width, f1s, width, label='CV F1-Score', color=c_f1, alpha=0.92)
    axs[0, 0].bar(x + 1.5*width, bal_accs, width, label='CV Bal. Acc', color=c_bal, alpha=0.92)
    axs[0, 0].set_title('(A) Stratified 5-Fold Cross-Validation Scores', fontsize=12, fontweight='bold')
    axs[0, 0].set_ylabel('Score (Mean)', fontsize=11, fontweight='bold')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(model_names_list, fontsize=10, rotation=15, ha='right')
    axs[0, 0].set_ylim(0.0, 1.15)
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.35, color='#94a3b8')
    axs[0, 0].legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Subplot B: Test Set Metrics
    axs[0, 1].bar(x - 1.5*width, accs, width, label='Accuracy', color='#0284c7', alpha=0.92)
    axs[0, 1].bar(x - 0.5*width, precs, width, label='Precision', color='#059669', alpha=0.92)
    axs[0, 1].bar(x + 0.5*width, recs, width, label='Recall', color='#d97706', alpha=0.92)
    axs[0, 1].bar(x + 1.5*width, f1s_test, width, label='F1-Score', color='#7c3aed', alpha=0.92)
    axs[0, 1].set_title('(B) Held-Out Test Set Performance (N = 775)', fontsize=12, fontweight='bold')
    axs[0, 1].set_ylabel('Score', fontsize=11, fontweight='bold')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(model_names_list, fontsize=10, rotation=15, ha='right')
    axs[0, 1].set_ylim(0.0, 1.15)
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.35, color='#94a3b8')
    axs[0, 1].legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Subplot C: ROC Curves
    axs[1, 0].plot([0, 1], [0, 1], linestyle='--', color='#94a3b8', lw=1.2, label='Chance (0.50)')
    for m_name in model_names_list:
        fpr, tpr, _ = roc_curve(y_test, test_results[m_name]['y_proba'])
        auc_val = test_results[m_name]['roc_auc']
        axs[1, 0].plot(fpr, tpr, label=f"{m_name} ({auc_val:.3f})", color=model_colors[m_name], lw=1.8)
    axs[1, 0].set_title('(C) ROC Curves Comparison', fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    axs[1, 0].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    axs[1, 0].set_xlim([-0.02, 1.02])
    axs[1, 0].set_ylim([-0.02, 1.05])
    axs[1, 0].grid(True, linestyle='--', alpha=0.35, color='#94a3b8')
    axs[1, 0].legend(loc='lower right', fontsize=9, framealpha=0.9)

    # Subplot D: PR Curves
    axs[1, 1].axhline(y=pos_prevalence, linestyle='--', color='#94a3b8', lw=1.2, label=f'Base ({pos_prevalence*100:.1f}%)')
    for m_name in model_names_list:
        prec_pts, rec_pts, _ = precision_recall_curve(y_test, test_results[m_name]['y_proba'])
        pr_auc_val = test_results[m_name]['pr_auc']
        axs[1, 1].plot(rec_pts, prec_pts, label=f"{m_name} ({pr_auc_val:.3f})", color=model_colors[m_name], lw=1.8)
    axs[1, 1].set_title('(D) Precision-Recall Curves Comparison', fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel('Recall', fontsize=11, fontweight='bold')
    axs[1, 1].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axs[1, 1].set_xlim([-0.02, 1.02])
    axs[1, 1].set_ylim([-0.02, 1.05])
    axs[1, 1].grid(True, linestyle='--', alpha=0.35, color='#94a3b8')
    axs[1, 1].legend(loc='lower left', fontsize=9, framealpha=0.9)

    plt.suptitle('Exoplanet Habitability Classification — Comprehensive 5-Model Benchmark', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    dash_path = os.path.join(results_dir, 'all_models_comprehensive_dashboard.png')
    fig_dash.savefig(dash_path, dpi=300)
    plt.close()
    log(f"[SAVED] all_models_comprehensive_dashboard.png -> {dash_path}")

    # 14. Individual Diagnostic Graph for Each Model (2x2 Multi-Panel)
    model_slug_map = {
        'Logistic Regression': 'logistic_regression',
        'SVM': 'svm',
        'Decision Tree': 'decision_tree',
        'Random Forest': 'random_forest',
        'XGBoost': 'xgboost'
    }

    for m_name in model_names_list:
        m_slug = model_slug_map[m_name]
        t_res = test_results[m_name]
        cv_res = cv_results[m_name]

        fig_m, ax_m = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
        
        # 1. Confusion Matrix
        cm_arr = np.array(t_res['confusion_matrix'])
        cax = ax_m[0, 0].matshow(cm_arr, cmap='Blues')
        fig_m.colorbar(cax, ax=ax_m[0, 0], fraction=0.046, pad=0.04)
        ax_m[0, 0].set_xticks([0, 1])
        ax_m[0, 0].set_yticks([0, 1])
        ax_m[0, 0].set_xticklabels(['Non-Hab (0)', 'Hab (1)'], fontsize=10, fontweight='bold')
        ax_m[0, 0].set_yticklabels(['Non-Hab (0)', 'Hab (1)'], fontsize=10, fontweight='bold')
        ax_m[0, 0].set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
        ax_m[0, 0].set_ylabel('Ground Truth', fontsize=10, fontweight='bold')
        ax_m[0, 0].set_title(f'(A) Confusion Matrix (N = {len(X_test)})', fontsize=11, fontweight='bold', pad=12)
        for r in range(2):
            for c in range(2):
                v = cm_arr[r, c]
                txt_col = "white" if v > cm_arr.max() / 2 else "black"
                pct = (v / len(X_test)) * 100.0
                ax_m[0, 0].text(c, r, f"{v:,}\n({pct:.1f}%)", ha="center", va="center", color=txt_col, fontsize=11, fontweight='bold')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, t_res['y_proba'])
        ax_m[0, 1].plot([0, 1], [0, 1], linestyle='--', color='#94a3b8', lw=1.2, label='Chance (0.50)')
        ax_m[0, 1].plot(fpr, tpr, color=model_colors[m_name], lw=2.2, label=f"ROC (AUC = {t_res['roc_auc']:.4f})")
        ax_m[0, 1].set_title('(B) Receiver Operating Characteristic (ROC)', fontsize=11, fontweight='bold')
        ax_m[0, 1].set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
        ax_m[0, 1].set_ylabel('True Positive Rate', fontsize=10, fontweight='bold')
        ax_m[0, 1].set_xlim([-0.02, 1.02])
        ax_m[0, 1].set_ylim([-0.02, 1.05])
        ax_m[0, 1].grid(True, linestyle='--', alpha=0.35, color='#94a3b8')
        ax_m[0, 1].legend(loc='lower right', fontsize=10, framealpha=0.9)

        # 3. Precision-Recall Curve
        prec_pts, rec_pts, _ = precision_recall_curve(y_test, t_res['y_proba'])
        ax_m[1, 0].axhline(y=pos_prevalence, linestyle='--', color='#94a3b8', lw=1.2, label=f'Prevalence ({pos_prevalence*100:.2f}%)')
        ax_m[1, 0].plot(rec_pts, prec_pts, color=model_colors[m_name], lw=2.2, label=f"PR (AUC = {t_res['pr_auc']:.4f})")
        ax_m[1, 0].set_title('(C) Precision-Recall Curve', fontsize=11, fontweight='bold')
        ax_m[1, 0].set_xlabel('Recall', fontsize=10, fontweight='bold')
        ax_m[1, 0].set_ylabel('Precision', fontsize=10, fontweight='bold')
        ax_m[1, 0].set_xlim([-0.02, 1.02])
        ax_m[1, 0].set_ylim([-0.02, 1.05])
        ax_m[1, 0].grid(True, linestyle='--', alpha=0.35, color='#94a3b8')
        ax_m[1, 0].legend(loc='lower left', fontsize=10, framealpha=0.9)

        # 4. Metric Summary Bar Chart
        metric_labels = ['Accuracy', 'Balanced Acc', 'Precision', 'Recall', 'F1-Score', 'PR-AUC', 'ROC-AUC']
        metric_vals = [
            t_res['accuracy'],
            t_res['balanced_accuracy'],
            t_res['precision'],
            t_res['recall'],
            t_res['f1'],
            t_res['pr_auc'],
            t_res['roc_auc']
        ]
        y_pos = np.arange(len(metric_labels))
        bars = ax_m[1, 1].barh(y_pos, metric_vals, color=model_colors[m_name], alpha=0.85, height=0.6)
        ax_m[1, 1].set_yticks(y_pos)
        ax_m[1, 1].set_yticklabels(metric_labels, fontsize=10, fontweight='semibold')
        ax_m[1, 1].set_xlim([0.0, 1.15])
        ax_m[1, 1].set_xlabel('Score (0.0 - 1.0)', fontsize=10, fontweight='bold')
        ax_m[1, 1].set_title('(D) Test Performance Summary Metrics', fontsize=11, fontweight='bold')
        ax_m[1, 1].grid(axis='x', linestyle='--', alpha=0.35, color='#94a3b8')
        ax_m[1, 1].invert_yaxis()

        for b in bars:
            w = b.get_width()
            ax_m[1, 1].text(w + 0.02, b.get_y() + b.get_height()/2, f"{w:.4f}", va='center', fontsize=9, fontweight='bold')

        plt.suptitle(f'Model Diagnostic Evaluation: {m_name}' + (' [Selected Final Model]' if m_name == winner_name else ''), fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        indiv_path = os.path.join(results_dir, f'model_graph_{m_slug}.png')
        fig_m.savefig(indiv_path, dpi=300)
        plt.close()
        log(f"[SAVED] model_graph_{m_slug}.png -> {indiv_path}")

        # If winner, also save to 05_FINAL_MODEL
        if m_name == winner_name:
            final_graph_path = os.path.join(models_dir, 'final_model_performance_graphs.png')
            fig_m_w, _ = plt.subplots(figsize=(1, 1)) # dummy to not crash
            plt.close()
            import shutil
            shutil.copyfile(indiv_path, final_graph_path)
            log(f"[SAVED] final_model_performance_graphs.png -> {final_graph_path}")

    # 15. Save Standalone Confusion Matrix Plots
    def save_cm_plot(cm, title, out_path):
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.8), dpi=300)
        cm_arr = np.array(cm)
        cax = ax_cm.matshow(cm_arr, cmap='Blues')
        fig_cm.colorbar(cax)
        classes = ['Non-Habitable (0)', 'Habitable (1)']
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(classes, fontsize=10, fontweight='bold')
        ax_cm.set_yticklabels(classes, fontsize=10, fontweight='bold')
        ax_cm.set_xlabel('Predicted Label', fontsize=11, fontweight='bold', labelpad=10)
        ax_cm.set_ylabel('Actual Ground Truth', fontsize=11, fontweight='bold', labelpad=10)
        ax_cm.set_title(title, fontsize=11, fontweight='bold', pad=15)
        for i in range(2):
            for j in range(2):
                val = cm_arr[i, j]
                text_color = "white" if val > cm_arr.max() / 2 else "black"
                ax_cm.text(j, i, f"{val:,}", ha="center", va="center", color=text_color, fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_cm.savefig(out_path, dpi=300)
        plt.close()

    cm_map = {
        'Logistic Regression': 'confusion_matrix_logistic_regression.png',
        'SVM': 'confusion_matrix_svm.png',
        'Decision Tree': 'confusion_matrix_decision_tree.png',
        'Random Forest': 'confusion_matrix_random_forest.png',
        'XGBoost': 'confusion_matrix_xgboost.png'
    }
    for m_name, cm_fname in cm_map.items():
        save_cm_plot(
            test_results[m_name]['confusion_matrix'],
            f'Confusion Matrix: {m_name}\n(Held-out Test Set, N={len(X_test)})',
            os.path.join(results_dir, cm_fname)
        )
        log(f"[SAVED] {cm_fname} -> {os.path.join(results_dir, cm_fname)}")

    # Save final_confusion_matrix.png to 05_FINAL_MODEL/
    final_cm_path = os.path.join(models_dir, 'final_confusion_matrix.png')
    save_cm_plot(
        test_results[winner_name]['confusion_matrix'],
        f'Confusion Matrix: Selected Model ({winner_name})\n(Held-out Test Set, N={len(X_test)})',
        final_cm_path
    )
    log(f"[SAVED] final_confusion_matrix.png -> {final_cm_path}")

    # 16. Save Validation Report to 05_FINAL_MODEL/
    validation_report = {
        "dataset_name": "PHL-ECW Held-Out Test Set (20% partition)",
        "total_test_samples": len(X_test),
        "selected_model": winner_name,
        "correct_predictions": int((test_results[winner_name]['y_pred'] == y_test.values).sum()),
        "accuracy_percent": round(test_results[winner_name]['accuracy'] * 100.0, 1),
        "precision_percent": round(test_results[winner_name]['precision'] * 100.0, 1),
        "recall_percent": round(test_results[winner_name]['recall'] * 100.0, 1),
        "f1_score_percent": round(test_results[winner_name]['f1'] * 100.0, 1),
        "balanced_accuracy_percent": round(test_results[winner_name]['balanced_accuracy'] * 100.0, 1),
        "pr_auc": round(test_results[winner_name]['pr_auc'], 4),
        "roc_auc": round(test_results[winner_name]['roc_auc'], 4),
        "confusion_matrix": test_results[winner_name]['confusion_matrix'],
        "five_models_summary": {
            m: {
                "cv_pr_auc": round(cv_results[m]['pr_auc']['mean'], 4),
                "cv_recall": round(cv_results[m]['recall']['mean'], 4),
                "cv_f1": round(cv_results[m]['f1']['mean'], 4),
                "cv_balanced_acc": round(cv_results[m]['balanced_accuracy']['mean'], 4),
                "test_pr_auc": round(test_results[m]['pr_auc'], 4),
                "test_f1": round(test_results[m]['f1'], 4),
                "test_accuracy": round(test_results[m]['accuracy'], 4),
                "test_precision": round(test_results[m]['precision'], 4),
                "test_recall": round(test_results[m]['recall'], 4)
            } for m in models.keys()
        }
    }

    report_json_path = os.path.join(models_dir, 'final_model_report.json')
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=4)
    log(f"[SAVED] final_model_report.json -> {report_json_path}")

    log("\n[SUCCESS] 5-Model Benchmark Run Completed & All Requested Deliverables Saved!")
    return winner_name, df_clean_comp


if __name__ == '__main__':
    train_and_benchmark_5_models()

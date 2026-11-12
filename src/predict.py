"""
predict.py
==========
Inference module for the Exoplanet Habitability Classifier.
Loads the final selected model pipeline (preprocessor + model) and provides
prediction capabilities for exoplanet astronomical parameters.
"""

import os
import joblib
import pickle
import pandas as pd
import numpy as np
from typing import Union, Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FINAL_MODEL_DIR = os.path.join(MODELS_DIR, 'final_model')

_cached_pipeline = None


def load_final_model():
    """
    Loads and caches the final trained machine learning pipeline.
    Checks models/final_model/final_model.joblib first, with fallbacks.
    """
    global _cached_pipeline
    if _cached_pipeline is not None:
        return _cached_pipeline

    candidate_paths = [
        os.path.join(PROJECT_ROOT, '05_FINAL_MODEL', 'final_model.joblib'),
        os.path.join(FINAL_MODEL_DIR, 'final_model.joblib'),
        os.path.join(MODELS_DIR, 'final_model.joblib'),
        os.path.join(MODELS_DIR, 'champion_pipeline.joblib'),
        os.path.join(MODELS_DIR, 'final_model.pkl'),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.pkl'):
                    with open(path, 'rb') as f:
                        _cached_pipeline = pickle.load(f)
                else:
                    _cached_pipeline = joblib.load(path)
                return _cached_pipeline
            except Exception as e:
                continue

    raise FileNotFoundError(
        "Trained final model pipeline not found. "
        "Please run 'python src/train_and_compare_5_models.py' first."
    )


def predict(input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """
    Runs habitability inference on provided planetary and stellar features.

    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        Dictionary or DataFrame containing the 18 input features.

    Returns:
    --------
    dict
        Prediction dictionary containing classification label, code, probability, and confidence.
    """
    pipeline = load_final_model()

    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise ValueError("input_data must be a dictionary or pandas DataFrame.")

    pred_code = int(pipeline.predict(df_input)[0])
    probas = pipeline.predict_proba(df_input)[0]
    confidence = float(probas[pred_code] * 100.0)
    label = "HABITABLE" if pred_code == 1 else "NON-HABITABLE"

    return {
        "prediction": label,
        "prediction_code": pred_code,
        "probability_percent": round(confidence, 1),
        "confidence_percent": round(confidence, 1),
        "model_used": "XGBoost",
        "habitable_probability": round(float(probas[1] * 100.0), 1) if len(probas) > 1 else None
    }


def run_demo():
    """Demonstrates inference on verified test samples."""
    print("=" * 65)
    print("EXOPLANET HABITABILITY CLASSIFIER - INFERENCE DEMO")
    print("=" * 65)

    test_samples = [
        {
            "name": "Earth Analog",
            "features": {
                "P. Mass (EU)": 1.0, "P. Radius (EU)": 1.0, "P. Density (EU)": 1.0,
                "P. Gravity (EU)": 1.0, "P. Esc Vel (EU)": 1.0, "P. Surf Press (EU)": 1.0,
                "P. Teq Mean (K)": 255.0, "P. SFlux Mean (EU)": 1.0, "P. Period (days)": 365.25,
                "P. Sem Major Axis (AU)": 1.0, "P. Eccentricity": 0.0167, "S. Mass (SU)": 1.0,
                "S. Radius (SU)": 1.0, "S. Teff (K)": 5778.0, "S. Luminosity (SU)": 1.0,
                "S. No. Planets": 8, "P. Disc. Method": "Transit", "S. Type": "G"
            }
        },
        {
            "name": "TRAPPIST-1 g",
            "features": {
                "P. Mass (EU)": 1.34, "P. Radius (EU)": 1.13, "P. Density (EU)": 0.94,
                "P. Gravity (EU)": 1.06, "P. Esc Vel (EU)": 1.09, "P. Surf Press (EU)": 1.30,
                "P. Teq Mean (K)": 181.1, "P. SFlux Mean (EU)": 0.255, "P. Period (days)": 12.35,
                "P. Sem Major Axis (AU)": 0.0451, "P. Eccentricity": 0.0, "S. Mass (SU)": 0.08,
                "S. Radius (SU)": 0.12, "S. Teff (K)": 2550.0, "S. Luminosity (SU)": 0.000519,
                "S. No. Planets": 7, "P. Disc. Method": "Transit", "S. Type": "M"
            }
        },
        {
            "name": "Kepler-221 b (Hot sub-Neptune)",
            "features": {
                "P. Mass (EU)": 5.68, "P. Radius (EU)": 1.71, "P. Density (EU)": 1.13,
                "P. Gravity (EU)": 1.93, "P. Esc Vel (EU)": 1.82, "P. Surf Press (EU)": 6.40,
                "P. Teq Mean (K)": 1088.3, "P. SFlux Mean (EU)": 332.99, "P. Period (days)": 2.80,
                "P. Sem Major Axis (AU)": 0.037, "P. Eccentricity": 0.0, "S. Mass (SU)": 0.72,
                "S. Radius (SU)": 0.82, "S. Teff (K)": 5243.0, "S. Luminosity (SU)": 0.4558,
                "S. No. Planets": 4, "P. Disc. Method": "Transit", "S. Type": "K"
            }
        },
        {
            "name": "HIP 78530 b (Cold massive gas giant)",
            "features": {
                "P. Mass (EU)": 7325.34, "P. Radius (EU)": 10.74, "P. Density (EU)": 5.92,
                "P. Gravity (EU)": 63.53, "P. Esc Vel (EU)": 26.12, "P. Surf Press (EU)": 43335.5,
                "P. Teq Mean (K)": 43.2, "P. SFlux Mean (EU)": 0.0008, "P. Period (days)": 100000.0,
                "P. Sem Major Axis (AU)": 710.0, "P. Eccentricity": 0.0, "S. Mass (SU)": 2.50,
                "S. Radius (SU)": 6.19, "S. Teff (K)": 10500.0, "S. Luminosity (SU)": 415.08,
                "S. No. Planets": 1, "P. Disc. Method": "Imaging", "S. Type": "B"
            }
        }
    ]

    for sample in test_samples:
        res = predict(sample["features"])
        print(f"Exoplanet:   {sample['name']}")
        print(f"Prediction:  {res['prediction']}")
        print(f"Confidence:  {res['confidence_percent']}%")
        print("-" * 65)


if __name__ == '__main__':
    run_demo()

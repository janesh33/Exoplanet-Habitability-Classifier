"""
data_loader.py
==============
Module for loading the raw and processed Exoplanet Habitability datasets.
"""

import os
import pandas as pd
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Primary paths following the new project structure
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, '01_RAW_DATA', 'original_dataset.csv')
if not os.path.exists(RAW_DATA_PATH):
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'phl_hec_all_confirmed.csv')

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, '02_PREPROCESSED_DATA', 'cleaned_dataset.csv')
if not os.path.exists(PROCESSED_DATA_PATH):
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'exoplanets_cleaned.csv')


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the exoplanet dataset from a CSV file.

    Parameters:
    -----------
    file_path : Optional[str]
        Path to the CSV file. If None, defaults to the raw dataset path.

    Returns:
    --------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """
    target_path = file_path if file_path is not None else RAW_DATA_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Dataset file not found at: {target_path}")

    df = pd.read_csv(target_path)
    return df


if __name__ == '__main__':
    df = load_dataset()
    print(f"Dataset successfully loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")

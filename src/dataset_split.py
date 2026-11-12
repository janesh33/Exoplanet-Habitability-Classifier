"""
dataset_split.py
================
Module for stratified dataset splitting, test set isolation, and saving processed
artifacts into 02_PREPROCESSED_DATA/ and 03_PROCESSED_DATA/ for reproducible
model benchmarking.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional

try:
    from src.data_loader import load_dataset
    from src.preprocessing import clean_raw_dataframe, identify_features, build_preprocessing_pipeline
except ImportError:
    from data_loader import load_dataset
    from preprocessing import clean_raw_dataframe, identify_features, build_preprocessing_pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, '02_PREPROCESSED_DATA')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, '03_PROCESSED_DATA')


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train/test split to guarantee the rare positive class (Habitable)
    is distributed proportionally in both partitions.

    Parameters:
    -----------
    X : pd.DataFrame
        Cleaned input features.
    y : pd.Series
        Target binary labels.
    test_size : float
        Proportion of dataset reserved for unseen final evaluation (default 0.2 = 20%).
    random_state : int
        Reproducible random seed (42).
    stratify : bool
        Whether to perform stratified sampling based on y.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )
    return X_train, X_test, y_train, y_test


def prepare_and_save_data(
    raw_file_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    End-to-end dataset preparation:
    1. Loads raw dataset.
    2. Cleans features and separates target.
    3. Performs stratified train/test split.
    4. Saves cleaned_dataset.csv in 02_PREPROCESSED_DATA/.
    5. Saves X_train, X_test, y_train, y_test, train_data, test_data,
       transformed_feature_names.json, and dataset_metadata.json in 03_PROCESSED_DATA/.

    Returns:
    --------
    Dict
        Summary dictionary with split counts, class distributions, and saved paths.
    """
    target_processed_dir = output_dir if output_dir is not None else PROCESSED_DIR
    target_preprocessed_dir = PREPROCESSED_DIR

    os.makedirs(target_processed_dir, exist_ok=True)
    os.makedirs(target_preprocessed_dir, exist_ok=True)

    # 1. Load and Clean
    df_raw = load_dataset(raw_file_path)
    X, y = clean_raw_dataframe(df_raw)
    feature_info = identify_features()

    # 2. Split
    X_train, X_test, y_train, y_test = split_dataset(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=True
    )

    # 3. Save cleaned dataset into 02_PREPROCESSED_DATA/
    cleaned_df = X.copy()
    cleaned_df['P. Habitable'] = y
    cleaned_path = os.path.join(target_preprocessed_dir, 'cleaned_dataset.csv')
    cleaned_df.to_csv(cleaned_path, index=False)

    # 4. Save splits into 03_PROCESSED_DATA/
    train_df = X_train.copy()
    train_df['P. Habitable'] = y_train
    train_path = os.path.join(target_processed_dir, 'train_data.csv')
    train_df.to_csv(train_path, index=False)

    test_df = X_test.copy()
    test_df['P. Habitable'] = y_test
    test_path = os.path.join(target_processed_dir, 'test_data.csv')
    test_df.to_csv(test_path, index=False)

    # Separate feature and label files
    X_train.to_csv(os.path.join(target_processed_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(target_processed_dir, 'X_test.csv'), index=False)
    y_train.to_frame(name='P. Habitable').to_csv(os.path.join(target_processed_dir, 'y_train.csv'), index=False)
    y_test.to_frame(name='P. Habitable').to_csv(os.path.join(target_processed_dir, 'y_test.csv'), index=False)

    # 5. Extract and save transformed feature names
    preprocessor = build_preprocessing_pipeline(scale_numeric=True)
    preprocessor.fit(X_train)
    transformed_feature_names = list(preprocessor.get_feature_names_out())

    with open(os.path.join(target_processed_dir, 'transformed_feature_names.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'raw_input_features': list(X.columns),
            'numerical_features': feature_info['numerical_features'],
            'categorical_features': feature_info['categorical_features'],
            'transformed_feature_names': transformed_feature_names,
            'transformed_feature_count': len(transformed_feature_names)
        }, f, indent=4)

    # 6. Save metadata
    metadata = {
        'dataset_name': 'Planetary Habitability Laboratory - Habitable Exoplanets Catalog (PHL-ECW)',
        'source': 'Planetary Habitability Laboratory @ University of Puerto Rico at Arecibo',
        'total_raw_samples': len(df_raw),
        'total_raw_features': df_raw.shape[1],
        'clean_features_count': X.shape[1],
        'numerical_features_count': len(feature_info['numerical_features']),
        'categorical_features_count': len(feature_info['categorical_features']),
        'target_column': 'P. Habitable',
        'target_classes': {'0': 'Non-Habitable', '1': 'Potentially Habitable'},
        'total_class_distribution': {
            '0': int((y == 0).sum()),
            '1': int((y == 1).sum())
        },
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_percentage': f"{int((1-test_size)*100)}%",
        'test_percentage': f"{int(test_size*100)}%",
        'train_class_distribution': {
            '0': int((y_train == 0).sum()),
            '1': int((y_train == 1).sum())
        },
        'test_class_distribution': {
            '0': int((y_test == 0).sum()),
            '1': int((y_test == 1).sum())
        },
        'stratified': True,
        'random_state': random_state
    }

    metadata_path = os.path.join(target_processed_dir, 'dataset_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    return metadata


if __name__ == '__main__':
    meta = prepare_and_save_data()
    print("Dataset preparation and splitting complete!")
    print(json.dumps(meta, indent=2))

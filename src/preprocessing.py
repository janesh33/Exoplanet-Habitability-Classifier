"""
preprocessing.py
================
Module for cleaning raw astronomical measurements and building leak-free
Scikit-learn preprocessing pipelines for numerical and categorical features.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from typing import Tuple, List, Optional
def identify_features() -> dict:
    """
    Systematically categorizes all dataset columns into:
    - Target column
    - Data leakage columns (to exclude)
    - Identifiers & Metadata columns (to exclude)
    - High-missingness / Redundant columns (to exclude)
    - Valid Input Numerical Features
    - Valid Input Categorical Features
    """
    target = ['P. Habitable']

    data_leakage_columns = [
        'P. Habitable Class',   # Exact categorical string label of the target
        'P. Zone Class',        # Habitable zone classification directly derived from target boundaries
        'P. Atmosphere Class',  # Atmospheric classification derived from habitability status
        'P. Composition Class', # Empirical composition derived from density
        'P. Mass Class',        # Categorical mass classification
        'P. ESI',               # Earth Similarity Index (explicit mathematical formula for habitability)
        'P. Int ESI',           # Interior ESI metric
        'P. Surf ESI',          # Surface ESI metric
        'P. HZD',               # Habitable Zone Distance (explicit metric for habitability boundary)
        'P. HZC',               # Habitable Zone Composition metric
        'P. HZA',               # Habitable Zone Atmosphere metric
        'P. HZI',               # Habitable Zone Index
        'P. SPH',               # Standard Primary Habitability score
        'S. HabCat',            # Habitable Catalog inclusion binary flag
        'P. Hab Moon',          # Habitable Moon indicator
        'S. No. Planets HZ',    # Number of habitable zone planets in host system
        'S. Hab Zone Min (AU)', # Explicit Habitable Zone inner boundary
        'S. Hab Zone Max (AU)'  # Explicit Habitable Zone outer boundary
    ]

    identifier_columns = [
        'P. Name',
        'P. Name Kepler',
        'P. Name KOI',
        'S. Name',
        'S. Name HD',
        'S. Name HIP',
        'S. Constellation',
        'P. Disc. Year',
        'P. Confirmed',
        'Unnamed: 68',
        'S. RA (hrs)',
        'S. DEC (deg)'
    ]

    redundant_high_missing_columns = [
        'P. Min Mass (EU)',     # 70.4% missing; P. Mass contains the primary estimated value
        'P. Max Mass (EU)',     # 100.0% missing
        'P. Teq Min (K)',       # Redundant with P. Teq Mean
        'P. Teq Max (K)',       # Redundant with P. Teq Mean
        'P. Ts Min (K)',        # 54.9% missing; derived modeled surface temperature
        'P. Ts Mean (K)',       # 54.9% missing; derived modeled surface temperature
        'P. Ts Max (K)',        # 54.9% missing; derived modeled surface temperature
        'P. SFlux Min (EU)',    # Redundant with P. SFlux Mean
        'P. SFlux Max (EU)',    # Redundant with P. SFlux Mean
        'P. Mean Distance (AU)',# Redundant with P. Sem Major Axis (AU)
        'P. Mag',               # Apparent optical magnitude from Earth viewpoint
        'P. Appar Size (deg)',  # Apparent angular size from Earth viewpoint
        'P. Inclination (deg)', # 81.3% missing
        'P. Omega (deg)',       # Argument of periastron (orbital orientation angle)
        'S. Mag from Planet',   # Redundant with host star luminosity & distance
        'S. Size from Planet (deg)', # Redundant with host star radius & semi-major axis
        'S. [Fe/H]',            # 44.9% missing metallicity
        'S. Age (Gyrs)',        # 40.1% missing stellar age
        'S. Appar Mag',         # 24.0% missing apparent brightness
        'S. Distance (pc)'      # 30.6% missing distance from Earth
    ]

    numerical_features = [
        'P. Mass (EU)',
        'P. Radius (EU)',
        'P. Density (EU)',
        'P. Gravity (EU)',
        'P. Esc Vel (EU)',
        'P. Surf Press (EU)',
        'P. Teq Mean (K)',
        'P. SFlux Mean (EU)',
        'P. Period (days)',
        'P. Sem Major Axis (AU)',
        'P. Eccentricity',
        'S. Mass (SU)',
        'S. Radius (SU)',
        'S. Teff (K)',
        'S. Luminosity (SU)',
        'S. No. Planets'
    ]

    categorical_features = [
        'P. Disc. Method',
        'S. Type'
    ]

    return {
        'target': target,
        'data_leakage': data_leakage_columns,
        'identifiers': identifier_columns,
        'redundant_high_missing': redundant_high_missing_columns,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }


def clean_raw_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleans raw astronomical data:
    1. Extracts target y ('P. Habitable').
    2. Converts numerical fields stored as strings (e.g., 'P. SFlux Mean (EU)').
    3. Replaces physically impossible zeros (e.g., mass, density, stellar radius <= 0, stellar temp < 100K) with NaN.
    4. Cleans and standardizes categorical features (Discovery Method and Stellar Spectral Type).
    5. Drops data leakage, metadata, and redundant high-missing columns.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Cleaned feature matrix X and target series y.
    """
    feature_info = identify_features()
    df_clean = df.copy()

    # 1. Target
    if 'P. Habitable' not in df_clean.columns:
        raise ValueError("Target column 'P. Habitable' not found in dataframe.")
    y = df_clean['P. Habitable'].astype(int)

    # 2. String to Numeric conversion
    if 'P. SFlux Mean (EU)' in df_clean.columns:
        df_clean['P. SFlux Mean (EU)'] = pd.to_numeric(df_clean['P. SFlux Mean (EU)'], errors='coerce')

    # 3. Clean physically impossible values
    df_clean.loc[df_clean['P. Mass (EU)'] <= 0, 'P. Mass (EU)'] = np.nan
    df_clean.loc[df_clean['P. Density (EU)'] <= 0, 'P. Density (EU)'] = np.nan
    df_clean.loc[df_clean['S. Radius (SU)'] <= 0, 'S. Radius (SU)'] = np.nan
    df_clean.loc[df_clean['S. Teff (K)'] < 100, 'S. Teff (K)'] = np.nan

    # 4. Standardize Categorical Features
    # Standardize Spectral Type to primary spectral class (M, K, G, F, A, B, Unknown)
    def clean_spectral_type(val):
        if pd.isna(val):
            return 'Unknown'
        val_str = str(val).strip().upper()
        if len(val_str) > 0 and val_str[0] in ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T', 'Y', 'D']:
            return val_str[0]
        return 'Other'

    df_clean['S. Type'] = df_clean['S. Type'].apply(clean_spectral_type)

    # Standardize Discovery Method
    def clean_disc_method(val):
        if pd.isna(val):
            return 'Other'
        val_str = str(val).strip().title()
        if 'Transit' in val_str:
            return 'Transit'
        if 'Radial Velocity' in val_str:
            return 'Radial Velocity'
        if 'Imaging' in val_str:
            return 'Imaging'
        if 'Microlensing' in val_str:
            return 'Microlensing'
        return 'Other'

    df_clean['P. Disc. Method'] = df_clean['P. Disc. Method'].apply(clean_disc_method)

    # 5. Select only valid input features
    selected_cols = feature_info['numerical_features'] + feature_info['categorical_features']
    X = df_clean[selected_cols].copy()

    return X, y


def build_preprocessing_pipeline(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = False,
    scaler_type: str = 'standard'
) -> ColumnTransformer:
    """
    Builds a leak-free Scikit-learn ColumnTransformer for numerical and categorical features.

    Parameters:
    -----------
    numerical_features : Optional[List[str]]
        List of numerical feature names.
    categorical_features : Optional[List[str]]
        List of categorical feature names.
    scale_numeric : bool
        If True, applies scaling to numerical features (for SVM, Logistic Regression, MLP).
        If False, applies only imputation (for Random Forest, XGBoost).
    scaler_type : str
        'standard' for StandardScaler, 'robust' for RobustScaler.

    Returns:
    --------
    ColumnTransformer
        Configured preprocessing pipeline.
    """
    if numerical_features is None or categorical_features is None:
        info = identify_features()
        if numerical_features is None:
            numerical_features = info['numerical_features']
        if categorical_features is None:
            categorical_features = info['categorical_features']

    # Numerical Transformer
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        if scaler_type == 'robust':
            num_steps.append(('scaler', RobustScaler()))
        else:
            num_steps.append(('scaler', StandardScaler()))

    num_pipeline = Pipeline(steps=num_steps)

    # Categorical Transformer
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


if __name__ == '__main__':
    from data_loader import load_dataset
    df = load_dataset()
    X, y = clean_raw_dataframe(df)
    print(f"Cleaned Feature Matrix X Shape: {X.shape}, Target y Shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Test pipeline
    pipeline_scaled = build_preprocessing_pipeline(scale_numeric=True)
    X_trans = pipeline_scaled.fit_transform(X)
    print(f"Transformed X Shape (after OneHotEncoding): {X_trans.shape}")

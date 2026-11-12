# PREPROCESSED DATA

This folder documents the systematic cleaning, auditing, feature selection, and standardization applied to the raw exoplanet catalog.

## Files in this Folder

| File | Description |
|---|---|
| `cleaned_dataset.csv` | Cleaned 19-column dataset (18 features + 1 target, 3,875 rows) |
| `preprocessing_report.csv` | Comprehensive audit of all 69 columns, data types, missing percentages, and actions |
| `missing_values_report.csv` | Column-by-column missing value counts before vs. after cleaning |
| `feature_selection.json` | Explicit breakdown of 18 selected features and 50 removed features with reasons |

---

## PREPROCESSING STEPS

### 1. Load original dataset
The raw CSV (`01_RAW_DATA/original_dataset.csv`) is read into memory containing 3,875 records across 69 astronomical attributes.

### 2. Inspect column names
Column headers are audited for standardized naming conventions, whitespace anomalies, and indexing artifacts (e.g., stripping trailing commas, detecting `Unnamed: 68`).

### 3. Check data types
Data types across all fields are verified. Fields containing numerical values stored as formatted object/strings (specifically `P. SFlux Mean (EU)`) are coerced to standard IEEE floating-point numbers (`pd.to_numeric(errors='coerce')`).

### 4. Handle missing values
Missing values across valid predictive features are audited:
- Numerical features retain genuine missingness (`NaN`) so that an algorithm-specific, leak-free median imputer within the cross-validation pipeline fills them strictly inside training folds.
- Categorical features with missing values are labeled as `'Unknown'` / `'Other'`.

### 5. Remove invalid values
Physically impossible astronomical measurements that reflect placeholder errors or corrupted records are sanitized:
- Negative or zero planetary masses (`P. Mass <= 0`) are converted to `NaN`.
- Zero planetary densities (`P. Density <= 0`) are converted to `NaN`.
- Non-positive stellar radii (`S. Radius <= 0`) are converted to `NaN`.
- Unphysical stellar effective temperatures (`S. Teff < 100 K`) are converted to `NaN`.

### 6. Remove unnecessary identifiers
12 non-predictive identifiers, observational catalog names, and celestial coordinate flags are stripped to prevent overfitting to catalog metadata:
- `P. Name`, `P. Name Kepler`, `P. Name KOI`, `S. Name`, `S. Name HD`, `S. Name HIP`, `S. Constellation`, `P. Disc. Year`, `P. Confirmed`, `Unnamed: 68`, `S. RA (hrs)`, `S. DEC (deg)`.

### 7. Remove leakage-prone columns
18 columns that encode direct mathematical formulas of habitability, habitable zone boundaries, or target-derived classifications are eliminated to prevent severe data leakage:
- `P. Habitable Class`, `P. Zone Class`, `P. Atmosphere Class`, `P. Composition Class`, `P. Mass Class`, `P. ESI`, `P. Int ESI`, `P. Surf ESI`, `P. HZD`, `P. HZC`, `P. HZA`, `P. HZI`, `P. SPH`, `S. HabCat`, `P. Hab Moon`, `S. No. Planets HZ`, `S. Hab Zone Min (AU)`, `S. Hab Zone Max (AU)`.

### 8. Select useful features
20 redundant columns or columns with severe missingness (>30% to 100%, e.g., `P. Max Mass (EU)` at 100% missing, `P. Inclination (deg)` at 81.3% missing, `P. Min Mass (EU)` at 70.4% missing) are excluded. 18 genuine, scientifically grounded features are selected:
- **16 Numerical Features:** `P. Mass (EU)`, `P. Radius (EU)`, `P. Density (EU)`, `P. Gravity (EU)`, `P. Esc Vel (EU)`, `P. Surf Press (EU)`, `P. Teq Mean (K)`, `P. SFlux Mean (EU)`, `P. Period (days)`, `P. Sem Major Axis (AU)`, `P. Eccentricity`, `S. Mass (SU)`, `S. Radius (SU)`, `S. Teff (K)`, `S. Luminosity (SU)`, `S. No. Planets`.
- **2 Categorical Features:** `P. Disc. Method`, `S. Type`.

### 9. Standardize/clean categorical values
High-cardinality and noisy categorical text are normalized into robust astronomical bins:
- `S. Type` (Spectral Type): Extracted to primary Harvard spectral classes (`O, B, A, F, G, K, M, L, T, Y, D`) with rare/noisy types grouped into `'Other'` and unobserved types as `'Unknown'`.
- `P. Disc. Method`: Standardized to core discovery techniques (`Transit`, `Radial Velocity`, `Imaging`, `Microlensing`, and `'Other'`).

### 10. Save cleaned dataset
The resulting cleaned DataFrame (3,875 rows × 19 columns: 18 features + 1 target `P. Habitable`) is serialized to `02_PREPROCESSED_DATA/cleaned_dataset.csv`.

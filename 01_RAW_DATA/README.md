# RAW DATA

This is the original dataset before preprocessing.
No rows, columns, or values should be modified in this folder.

## Dataset Specifications

- **Dataset Name:** Planetary Habitability Laboratory - Habitable Exoplanets Catalog (PHL-ECW)
- **Source:** Planetary Habitability Laboratory (PHL) at the University of Puerto Rico at Arecibo
- **File Name:** `original_dataset.csv`
- **Number of Rows:** 3,875 confirmed exoplanet records
- **Number of Columns:** 69 astronomical, orbital, stellar, and planetary attributes
- **Target Variable:** `P. Habitable` (0 = Non-Habitable, 1 = Potentially Habitable)

## Integrity Policy

This folder contains the authoritative, untouched raw dataset. All downstream cleaning, transformations, and modeling must read from this source without modifying the raw data file.

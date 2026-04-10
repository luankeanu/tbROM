# method

## Scope of this file
This document records what has been implemented so far in the current overhaul.
At this stage it only documents `read.py`.

## `PySINDy/read.py` implemented so far

### Purpose
`read.py` is now the data-entry layer for the project.
It is responsible for loading all simulation files every run, preparing trajectory data, and exporting prepared tables used by downstream steps.

### Current structure
The file is written in sections:
1. schema and case container
2. filesystem and split helpers
3. import and validation
4. data treatment and feature engineering
5. per-case build and runtime debug output
6. combined exports and summaries
7. model-facing trajectory helpers
8. run configuration snapshot

### What it currently does
- Scans `PySINDy/Data` for `.out` files each run.
- Classifies cases using filename prefixes (`T#` as test, all others as train/development).
- Reads each file with the known six-column schema after header skipping.
- Validates row count, missing values, duplicate times, and monotonic time.
- Trims startup rows based on configuration.
- Builds pitch features from velocity components:
  - `pitch_rad`
  - `pitch_deg`
  - `pitch`
  - `pitch_rate`
  - `pitch_accel`
- Adds lag features for `cl` and `cd`.
- Builds one dataframe per case plus combined export tables.
- Exports validation, database, summary, and correlation CSV files.
- Prints runtime read/debug summaries when loading files.

### Data loading model currently used
- One simulation file corresponds to one time trajectory.
- In memory, the loader keeps a list of per-case dataframes.
- For reporting/inspection, the loader can also create one combined dataframe containing all cases with case metadata columns.

### Notes for next steps
- `analyse.py` reload prompt behavior is not implemented yet.
- Any further changes to read behavior should be appended here as this overhaul progresses.

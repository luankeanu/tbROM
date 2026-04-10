# Project Method

This document describes how each project file is structured and what role it serves in the finalized workflow.
It is not an update history.

## Overall workflow

The project is split into data preparation, model definition and search, run orchestration, and result analysis.
Data is reloaded from disk whenever the program runs so the read module naturally adapts to added or removed simulation files.

## `PySINDy/read.py`

Purpose:
- Read all simulation `.out` files from `PySINDy/Data`.
- Classify cases into train or test based on filename prefix (`T#` means test).
- Validate file integrity and numeric consistency.
- Treat the data by trimming startup rows and creating pitch-based features.
- Build combined tabular exports and model-ready trajectory arrays.

Main output concepts:
- One in-memory dataframe per simulation case.
- One combined dataframe for exports and global analysis.
- Validation and summary tables written to `PySINDy/outputs`.

## `PySINDy/model.py`

Purpose:
- Define the PySINDy model setup.
- Build optimizer and feature library combinations.
- Fit and simulate state trajectories (`Cl`, `Cd`) with pitch-derived controls.
- Run brute-force hyperparameter search.
- Save best model artifacts and prediction tables.

## `PySINDy/main.py`

Purpose:
- Execute an end-to-end run.
- Trigger data loading, exports, hyperparameter search, model save, and prediction generation.
- Print run progress and summary.

## `PySINDy/analyse.py`

Purpose:
- Save metric tables.
- Generate analysis plots and prediction diagnostics.
- Export visual artifacts to the outputs folder.

## `PySINDy/config.py`

Purpose:
- Centralize run configuration.
- Define paths, row trimming rules, test prefix logic, plotting behavior, and hyperparameter grid settings.

## Data loading model

The data is trajectory-based, not row-independent tabular data.
Each simulation file is treated as one time-ordered case:
- time axis from `flow_time`
- raw inputs include `vx`, `vy`
- computed controls include pitch variables from `atan2(vy, vx)`
- model state variables are `cl` and `cd`

At runtime, the loader creates:
- a list of per-case dataframes for trajectory-safe modeling
- a combined dataframe for exports and diagnostics

This keeps temporal structure intact while still producing a single consolidated file for inspection.

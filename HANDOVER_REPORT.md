# PySINDy Buffet Project Handover Report

## Purpose of this report
This document is a register of the work completed in `D:\Model\PySINDy` during the current implementation pass.

It is written for another engineer or another LLM that may continue the project on a different computer. The goal is to make the current state, design decisions, file roles, assumptions, and known limitations explicit so the next person does not need to infer them from the code alone.

## Requested project goal
The requested goal was to build a modular PySINDy-based workflow to analyse buffet simulation outputs and eventually predict `Cl(t)` and `Cd(t)` from a new pitch history derived from `Vx` and `Vy`.

The intended workflow was:

1. Read multiple CFD output files.
2. Validate and treat the data.
3. Compute pitch from velocity components.
4. Build a database from all available simulations.
5. Train one main model using PySINDy.
6. Support brute-force hyperparameter search.
7. Reserve `T*` files for final testing.
8. Export tables and plots into a dedicated output folder.
9. Keep the code modular, but simpler than the older coursework project.

## What was found in the workspace before implementation
At the start of implementation, the `PySINDy` folder only contained:

- `Data/`

There was no existing code in `PySINDy`.

The prior project used as a structural reference was:

- `D:\Model\DataAnalysis_CourseWork_final`

That coursework project contained:

- a modular layout with separate Python scripts
- a `README.txt`
- a pattern of exporting tables and plots into a specific folder
- a brute-force style hyperparameter workflow

The new project was intentionally designed to preserve that general style while reducing the number of modules.

## What was discovered about the data format
The input files in `D:\Model\PySINDy\Data` were inspected directly.

Observed current files:

- `B1_OUTPUT.out`
- `B2_OUTPUT.out`
- `B3_OUTPUT.out`
- `B4_OUTPUT.out`
- `C1_OUTPUT.out`
- `C2_OUTPUT.out`
- `C3_OUTPUT.out`
- `C4_OUTPUT.out`

Observed structure:

- 3 non-numeric header lines
- 8000 numeric rows per file
- actual numeric column order:
  - `time_step`
  - `flow_time`
  - `vy`
  - `vx`
  - `cl`
  - `cd`

Important detail:

- the printed `time_step` values begin at `501`
- therefore the 8000 numeric rows correspond to printed iterations `501` through `8500`

## Important trim rule clarification
There was a clarification after implementation planning:

- the user wants to remove the first 500 numeric rows
- since the printed iteration count begins at `501`, that means:
  - dropped printed iterations: `501` to `1000`
  - retained printed iterations: `1001` to `8500`

This leaves exactly 7500 retained rows.

### Current implemented behavior
The current code already does this correctly by row position.

In `pipeline.py`, trimming is done with:

```python
frame.iloc[config.ROWS_TO_SKIP :]
```

In `config.py`:

```python
ROWS_TO_SKIP = 500
EXPECTED_NUMERIC_ROWS = 8000
EXPECTED_RETAINED_ROWS = 7500
```

So the current behavior is:

- skip the first 500 numeric rows after the 3-line header
- keep the remaining 7500 numeric rows

The earlier confusion was wording only. The code matches the intended behavior.

## Conceptual modelling decision
One major conceptual point was clarified during planning and implementation:

### Each file is one trajectory, not one independent sample
This project is not structured like the coursework project where each row represented one static manufactured sample with several features.

Here:

- each `.out` file is one dynamic trajectory
- each row is one time point from that trajectory
- the airfoil is the same system across files
- the forcing changes between files through pitch history

Because of that, the correct way to combine the data is:

- keep each file as its own trajectory during preparation
- attach metadata such as `case_name` and `case_group`
- train a shared model across multiple trajectories
- validate by holding out complete files, not random rows

This is the central modelling assumption behind the current implementation.

## High-level design that was implemented
The project was simplified into five Python modules plus an outputs folder:

- `config.py`
- `pipeline.py`
- `model.py`
- `analyse.py`
- `main.py`
- `README.txt`
- `outputs/`

This was done intentionally because the user requested fewer modules than the original coursework structure.

## Files created
The following files were created in `D:\Model\PySINDy`:

### 1. `config.py`
Purpose:

- central place for user-edited settings
- paths
- trimming rules
- validation mode
- plot window
- hyperparameter ranges
- output filenames

Main contents:

- `BASE_DIR`, `DATA_DIR`, `OUTPUT_DIR`
- `TEST_PREFIXES = ("T",)`
- `HEADER_LINES = 3`
- `ROWS_TO_SKIP = 500`
- `EXPECTED_NUMERIC_ROWS = 8000`
- `EXPECTED_RETAINED_ROWS = 7500`
- pitch feature switches
- grouped validation settings
- brute-force hyperparameter grid for PySINDy

Why this file exists:

- to make future runs easy to modify without touching the logic
- to support manual narrowing of hyperparameter ranges between runs

### 2. `pipeline.py`
Purpose:

- all data reading and preparation logic

Main responsibilities:

- discover `.out` files
- ignore hidden `._...` files
- parse the data files
- validate file shape and numeric integrity
- trim the numeric rows
- compute pitch-derived features
- combine all cases into one database
- export validation, database, summary, correlation files
- split development files from future `T*` test files
- define grouped whole-file validation folds
- export run configuration snapshot

Important objects and functions:

- `CaseData`
  - a dataclass storing metadata and the prepared dataframe for one case

- `discover_data_files()`
  - finds input `.out` files in `Data`

- `read_case_file()`
  - reads the numeric block using pandas

- `validate_raw_frame()`
  - checks:
    - row count
    - missing values
    - duplicate times
    - monotonic time
    - basic `dt` statistics

- `trim_frame()`
  - removes the first 500 numeric rows

- `add_pitch_features()`
  - computes:
    - `pitch_rad`
    - `pitch_deg`
    - `pitch`
    - `pitch_rate`
    - `pitch_accel`
    - `cl_lag1`
    - `cd_lag1`

- `combined_database()`
  - merges all prepared cases into one dataframe while preserving case identity

- `validation_splits()`
  - returns grouped holdout splits by file

Why this file exists:

- to keep data treatment in one place
- to isolate the “how do we combine these simulations correctly?” logic from the modelling logic

### 3. `model.py`
Purpose:

- PySINDy model definition, brute-force hyperparameter search, simulation, metrics, and saved artifacts

Main responsibilities:

- check whether `pysindy` is installed
- build the optimizer
- build the feature library
- fit a SINDy-with-control model
- simulate `Cl/Cd` using the pitch-derived control inputs
- compute regression metrics
- evaluate grouped validation cases
- run brute-force hyperparameter search
- save the final model
- save summary tables
- export prediction tables

Important assumptions:

- model state is:
  - `[Cl, Cd]`

- control input starts as:
  - `pitch`
  - optionally `pitch_rate`
  - optionally `pitch_accel`

Current library/optimizer logic:

- optimizer options:
  - `STLSQ`
  - `SR3`

- library options:
  - polynomial library
  - optional Fourier library

Important correction made during implementation:

The original first pass risked exporting “validation” predictions from the final full-data refit, which would not be true held-out validation output.

This was corrected by adding:

- `prediction_frame()`

and returning true holdout predictions from the best cross-validation search result.

This means:

- `validation_predictions.csv` now corresponds to held-out file predictions
- `test_predictions.csv` corresponds to predictions from the final model on any `T*` cases

### 4. `analyse.py`
Purpose:

- save metric tables and produce plots

Main responsibilities:

- export `train_metrics.csv`
- export `validation_metrics.csv`
- export `hyperparameter_results.csv`
- create `feature_correlation_heatmap.html`
- create cross-case overlay plots over a chosen time interval
- create actual-vs-predicted plots per case
- create residual plots per case

Current plot outputs:

- feature correlation heatmap
- overlay of `pitch`, `Cl`, and `Cd` over the configured window
- per-case comparison plots for predicted vs actual
- per-case residual plots

### 5. `main.py`
Purpose:

- single entrypoint for the whole workflow

Main responsibilities:

1. create `outputs/`
2. write `run_config.json`
3. load and validate all cases
4. export prepared data tables
5. create basic analysis plots
6. split development vs test cases
7. run grouped hyperparameter search
8. save best-model summary and artifact
9. export held-out validation predictions
10. export test predictions for any `T*` files

This file is what should be run with:

```bash
python main.py
```

### 6. `README.txt`
Purpose:

- short usage and module guide in the same general style as the coursework README

This file was added because the user explicitly requested a README similar to the older project.

### 7. `outputs/`
Purpose:

- single location for all overwriteable run outputs

This folder was created even though the project could not be run end-to-end in the current environment, so the structure is already present.

## Files not created
A Jupyter notebook was not created.

Reason:

- the user explicitly said a notebook was not needed yet
- the notebook can be added later once the code workflow stabilizes

## Why the code is organized this way
The structure was chosen to balance three constraints:

1. the user wanted modular code
2. the user did not want too many tiny modules
3. the project still needs clear separation between data treatment, modelling, and analysis

So the final layout is intentionally more compact than the coursework project while still keeping the main responsibilities separate.

## How the files were created
The files were created directly in the workspace using patch-based edits.

No code generation framework or scaffolding tool was used.

The process was:

1. inspect existing workspace folders
2. inspect the coursework project files for style and structure
3. inspect the `.out` data files to verify actual column order and row count
4. plan the simplified module structure
5. create the new Python files
6. create the new README
7. create the outputs directory
8. re-read the created files to catch obvious design mistakes
9. patch the validation export logic after identifying a conceptual issue in the first pass

## Source inspection that informed the implementation
The following existing resources were used as references:

- `D:\Model\DataAnalysis_CourseWork_final\README.txt`
- `D:\Model\DataAnalysis_CourseWork_final\read.py`
- `D:\Model\DataAnalysis_CourseWork_final\modules.py`
- `D:\Model\DataAnalysis_CourseWork_final\hyperparameters.py`
- `D:\Model\DataAnalysis_CourseWork_final\analyse.py`
- `D:\Model\PySINDy\Data\*.out`

The coursework files were used mostly to preserve:

- modular style
- overwriteable exported outputs
- brute-force hyperparameter mentality
- README structure

The new project does not copy the coursework logic directly because the data and modelling problem are fundamentally different.

## Current data treatment logic
For each `.out` file:

1. skip the 3-line header
2. read 6 numeric columns
3. check that 8000 numeric rows exist
4. convert all columns to numeric
5. validate missing values and time monotonicity
6. remove the first 500 numeric rows
7. keep the remaining 7500 rows
8. compute pitch as:

```python
pitch_rad = np.arctan2(vy, vx)
pitch_deg = np.degrees(pitch_rad)
```

9. define the main working `pitch` column as degrees
10. compute `pitch_rate` with numerical differentiation over `flow_time`
11. compute `pitch_accel` similarly
12. compute one-step lag columns for `Cl` and `Cd`
13. store the case in a `CaseData` object

## Current modelling logic
The project currently assumes that PySINDy should be used as a controlled dynamical system model:

- state:
  - `x(t) = [Cl(t), Cd(t)]`

- control:
  - `u(t)` from pitch-derived inputs

The current implementation uses:

- `pitch` as the base control
- `pitch_rate` optionally included
- `pitch_accel` optionally included

The hyperparameter grid currently spans:

- optimizer type
- threshold
- alpha
- polynomial degree
- Fourier frequency count
- feature inclusion switches

Grouped validation is performed by holding out whole files.

## Current outputs expected after a successful run
If the project is run successfully on a machine with Python and PySINDy available, the outputs folder should contain:

- `file_validation.csv`
- `combined_database.csv`
- `case_summary.csv`
- `feature_correlation.csv`
- `feature_correlation_heatmap.html`
- `hyperparameter_results.csv`
- `train_metrics.csv`
- `validation_metrics.csv`
- `validation_predictions.csv`
- `test_predictions.csv`
- `best_model_summary.csv`
- `best_model.pkl`
- `run_config.json`
- case comparison plots
- case residual plots

## Current environment limitation during implementation
The project could not be executed end-to-end in the current environment.

Reason:

- the visible `python.exe` on this machine was the Windows app shim
- no working Python interpreter was available through the current tool execution path
- as a result, runtime verification of imports and model fitting could not be performed here

This means the implementation was completed through:

- repository inspection
- data inspection
- static reasoning
- static code review after writing the files

## Known limitations and likely next tasks
The next engineer or LLM should assume the following items still need runtime verification:

### 1. Verify Python environment on the destination machine
Required packages should include at least:

- `numpy`
- `pandas`
- `plotly`
- `pysindy`

Depending on the installed PySINDy version, minor API adjustments may be needed for:

- `SR3`
- `SINDy.fit(...)`
- `SINDy.simulate(...)`
- library constructors

### 2. Confirm simulation stability
Some SINDy models can become numerically unstable during simulation even if they fit derivatives acceptably.

That means the next step after environment setup should be:

- run `python main.py`
- inspect whether the simulated `Cl/Cd` histories remain stable
- inspect whether the best-ranked hyperparameters produce physically reasonable curves

### 3. Possibly refine feature engineering
The current feature set is a reasonable first pass, but may need refinement.

Potential future changes:

- turn `pitch_accel` on
- change smoothing window
- add delayed pitch terms
- include phase-shifted or filtered versions of pitch
- treat constant-pitch and varying-pitch cases differently during analysis

### 4. Possibly refine validation logic
Current validation is leave-one-file-out by default.

That is appropriate for the current use case, but later the project may need:

- grouped validation by case family
- separate reporting for `B`, `C`, `D`, `E`, `T`
- special treatment for ramp cases vs harmonic cases

### 5. Expand plots and reports
The current plots are functional but minimal.

A future pass may add:

- `pitch vs Cl`
- `pitch vs Cd`
- hysteresis loops
- side-by-side plots by case family
- more detailed residual analysis

## Important current assumptions
The current code assumes:

1. every valid file has exactly 8000 numeric rows
2. every file has the same six-column structure
3. files beginning with `T` are final test cases
4. all other files are development/training cases
5. pitch is computed from `atan2(Vy, Vx)` in degrees
6. the first 500 numeric rows should be discarded
7. grouped whole-file holdout is the correct default validation method

If any of those assumptions change, `config.py` and parts of `pipeline.py` may need to be updated.

## If another LLM continues this work
Recommended first actions for the next model:

1. Read:
   - `HANDOVER_REPORT.md`
   - `README.txt`
   - `config.py`
   - `pipeline.py`
   - `model.py`
   - `main.py`

2. Confirm the Python environment on the new computer.

3. Run:

```bash
python main.py
```

4. If imports fail, adjust the environment first before changing the code.

5. If PySINDy API differences appear, patch `model.py` only as much as needed to match the installed version.

6. Inspect:
   - `file_validation.csv`
   - `hyperparameter_results.csv`
   - `validation_predictions.csv`
   - generated plots

7. Only after that, decide whether the next step is:
   - bug fixing
   - model refinement
   - feature engineering
   - adding notebook/report material

## Summary of what was actually done
In plain terms, this implementation pass accomplished the following:

- created a new modular PySINDy project from scratch inside `D:\Model\PySINDy`
- based the structure loosely on the older coursework project
- inspected the actual `.out` files and corrected the assumed column order from the prompt
- implemented the trim rule that keeps the last 7500 numeric rows
- implemented pitch calculation from `Vy` and `Vx`
- implemented grouped whole-file validation
- implemented brute-force hyperparameter search scaffolding
- implemented output export logic and plotting logic
- added a coursework-style README
- created the `outputs` folder
- corrected the validation export logic so it uses true held-out predictions

## Final note
The current state should be treated as a strong first implementation pass, not as a runtime-verified final version. The design is coherent and aligned with the user requirements, but the next engineer or LLM should expect to spend the next pass on:

- environment setup
- runtime verification
- small API compatibility fixes if needed
- model-quality refinement once actual outputs can be inspected

# PySINDy_V2 Method Log

## 2026-04-11: Read-stage implementation baseline

### Objective of this increment
The first V2 implementation pass establishes the data-ingestion stage for the
PySINDy buffet workflow. The goal is to make the Fluent outputs usable in a
repeatable way before any model fitting or confirmation runs are introduced.

### Data assumptions fixed in this stage
The current CFD output files in `data/` are treated as time-resolved
trajectories rather than independent samples.

Each file is assumed to contain:
- 3 header lines
- 8000 numeric rows
- 6 numeric columns in the order:
  `time_step`, `flow_time`, `vy`, `vx`, `cl`, `cd`

Files whose names begin with `T` are treated as confirmation cases.
All other files are treated as training cases.

### Treatment applied in the read stage
The read stage applies the following sequence:
1. Discover all valid `.out` files in `data/`.
2. Ignore hidden macOS artefacts such as `._*`.
3. Read each file into a Pandas table with the fixed column order.
4. Coerce the data to numeric form.
5. Validate file length, missing values, duplicate times, and monotonic time.
6. Discard the first 500 numeric rows from every case.
7. Retain the remaining 7500 rows.
8. Compute the canonical `pitch(t)` signal from `atan2(vy, vx)`.
9. Store `pitch` in degrees.
10. Preserve case identity and dataset role on every prepared row.

### Explicit V2 design choice
The first V2 read stage does not create pitch derivatives.

The only engineered signal kept at this stage is `pitch(t)`, because the later
confirmation workflow is expected to use the pitch history itself alongside
`cl` and `cd`. This keeps the prepared dataset closer to the physical inputs
and avoids committing early to derivative-based features before the model stage
has been designed properly.

### Outputs created by the read stage
The stage is designed to support two forms of handoff:

1. In-memory Pandas objects for the rest of the Python workflow.
2. CSV exports for traceability, debugging, and reporting.

The exported files are:
- `prepared_train_cases.csv`
- `prepared_confirmation_cases.csv`
- `file_validation.csv`
- `case_summary.csv`

### Runtime file layout decision
The executable workflow keeps `main.py` at the project root and stores the
stage modules inside `assets/`.

At the current stage this means:
- `main.py` is the single script the user runs
- `assets/read.py` contains the read-stage implementation
- future stage files should follow the same layout inside `assets/`

### Main entry-point behaviour at this stage
`main.py` is now the single entry point and already presents four `y/n` prompts:
- read
- fit
- run
- analyse

Only the read stage is implemented in this increment.
The remaining prompts are kept as explicit placeholders so the top-level
workflow shape is already fixed while the later modules are developed.

## 2026-04-11: Model-definition baseline

### Objective of this increment
The second V2 implementation pass separates the model definition from fitting
and execution. This mirrors the methodology used in the coursework project,
where each model file defines the model itself and later scripts decide how to
train, compare, and evaluate it.

### Role of `assets/model.py`
`assets/model.py` is a definition-only module.

Its job is to:
- define the canonical PySINDy model used by the project
- expose the shared model metadata needed by later stages
- return an untrained model object

It does not:
- fit the model
- run simulations
- compute metrics
- perform brute-force tuning

Those responsibilities remain reserved for the later workflow stages.

### Canonical V2 signal definition
The current model is set up as a control-driven dynamical system.

The canonical signals are:
- state vector: `cl`, `cd`
- control input: `pitch`
- time column: `flow_time`

This matches the current project objective: later confirmation runs will supply
the pitch history from the `T#` cases and the workflow will predict `cl(t)` and
`cd(t)` for comparison against the reference Fluent outputs.

### Baseline PySINDy recipe
The baseline model definition uses:
- a polynomial feature library
- the STLSQ sparse optimizer
- finite-difference differentiation

This is the current default sparse-regression recipe for V2. It is intended as
the clean baseline definition to be imported by later scripts, while
`hyperparameters.py` remains the place for brute-force tuning and manual
selection of improved settings.

### Literature basis recorded for the model choice
The baseline sparse-regression direction is consistent with the project
literature already collected in:
`Documents/Uni/City, Univertisy of London/Year 3/EG3000_Personal Project/A2_litrev/Papers/Originals/`

The most directly relevant references identified during this implementation pass
are:
- Brunton, Proctor and Kutz (2016), *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*
- Loiseau and Brunton (2018), *Constrained sparse Galerkin regression*

These references support the choice to begin from an interpretable sparse
equation-discovery framework rather than a black-box predictor. No local test
record was found that singled out a more specific PySINDy recipe as the proven
best setup, so the V2 baseline remains polynomial + STLSQ until later tuning or
validation shows a stronger alternative.

## 2026-04-11: Fit-stage baseline

### Objective of this increment
The fit stage establishes the first training workflow for V2. Its purpose is to
take the already defined read-stage data and the canonical PySINDy model
definition, fit one model on all training cases, and save the fitted result for
later reuse by the run stage.

### Role of `assets/fit.py`
`assets/fit.py` is the training stage for the baseline workflow.

Its job is to:
- obtain valid prepared training data
- build the canonical untrained model
- fit the model on the full non-confirmation dataset
- save the fitted model artifacts
- export a short fit summary
- append the discovered equations to a persistent history file

It does not:
- tune hyperparameters
- run holdout validation
- generate confirmation predictions
- analyse fit quality in detail

Those responsibilities remain reserved for later stages.

### Prepared-data reuse rule
The fit stage is designed to save resources when possible.

It first checks whether the prepared output files from the read stage already
exist and whether they are still usable. Reuse is accepted only if:
- the prepared training CSV exists
- the validation CSV exists
- the expected schema is present
- the training table is not empty
- at least one training case is available
- the validation table reports valid input data

If any of these checks fail, the fit stage reruns the read stage automatically
instead of training from incomplete or suspicious cached outputs.

### Fit artifacts and history policy
The fit stage now creates two model-object outputs:
- one overwriteable latest fitted-model pickle
- one archived pickle for the specific fit run

This allows later stages to use a single canonical latest model while still
preserving a history of earlier fitted objects when needed.

The stage also creates one overwriteable fit summary file and one append-only
equations-history file.

### Append-only equations history
The equations-history file is intentionally different from the other fit-stage
outputs. It does not overwrite previous content.

Each fit appends a new section labelled:
- `Fit 1`
- `Fit 2`
- `Fit 3`
- and so on

Each section records:
- the fit label
- whether the prepared data was reused or regenerated
- the number of training cases and rows
- the core model settings
- the number of non-zero coefficients
- the discovered equations

This file is meant to preserve how the governing equations evolve before and
after tuning, which will be useful later when discussing methodology and model
development in the written report.

## 2026-04-11: Run-stage baseline

### Objective of this increment
The run stage establishes the first confirmation-prediction workflow for V2.
Its role is to take the latest fitted model, apply it to the `T#` confirmation
cases, and export both detailed prediction tables and compact case summaries.

### Role of `assets/run.py`
`assets/run.py` is the confirmation-prediction stage.

Its job is to:
- load the latest fitted model
- obtain valid prepared confirmation data
- simulate one trajectory for each `T#` case
- compare each prediction only against its matching confirmation case
- export latest and archived prediction outputs
- export latest and archived per-case run summaries

It does not:
- refit the model
- tune hyperparameters
- generate plots
- perform deeper interpretation of the errors

Those responsibilities remain reserved for other stages.

### Confirmation-case simulation contract
The run stage uses only the confirmation dataset.

For each `T#` case:
- the first retained true `cl` and `cd` values are used as the initial state
- the case `pitch(t)` history is used as the control input
- the case `flow_time` is used as the simulation timeline

This approach is appropriate for the current dataset because all simulations
start from the same retained time location after the first 500 rows have been
discarded, so the first retained point is a meaningful shared reference.

### Prepared-data reuse rule
Like the fit stage, the run stage is designed to save resources.

It first tries to reuse the prepared confirmation outputs already written by the
read stage. Reuse is accepted only if:
- the prepared confirmation CSV exists
- the validation CSV exists
- the expected schema is present
- the confirmation table is not empty
- at least one confirmation case is available
- the validation table reports valid source data

If any of these checks fail, the run stage reruns the read stage automatically.

### Run-stage outputs and archive policy
The run stage now exports:
- one overwriteable latest confirmation-predictions CSV
- one overwriteable latest confirmation-summary CSV
- one archived predictions CSV for each run
- one archived summary CSV for each run

Every exported row and summary entry retains the confirmation case identity, so
comparisons remain separated by `T#` case. This is important because `T1` must
be evaluated against `T1`, `T2` against `T2`, and so on.

### Summary information exported by the run stage
The per-case run summary currently records:
- RMSE and MAE for `cl` and `cd`
- mean residual for `cl` and `cd`
- true extrema for both variables
- predicted extrema for both variables

This gives a compact first view of prediction quality before the dedicated
analysis stage is introduced.

## 2026-04-11: Hyperparameter-tuning baseline

### Objective of this increment
The hyperparameter-tuning utility establishes a training-only search workflow
for improving the baseline PySINDy model without using the confirmation cases
for model selection.

### Role of `assets/utils/hyperparameters.py`
`assets/utils/hyperparameters.py` is a standalone tuning utility rather than a
normal workflow stage.

Its job is to:
- search over candidate model settings
- validate those settings using only training cases
- rank the candidates using a consistent score
- export a full results table and best-summary output
- append the best candidate equations to a tuning-history file
- optionally apply the chosen best values to `assets/model.py`

It does not:
- use `T#` cases for selection
- replace the fit or run stages
- automatically retrain the canonical model after tuning

### Search and validation strategy
The first tuning version uses:
- grid search
- leave-one-training-case-out validation
- a training-only split by whole case rather than by row

For each candidate combination:
1. fit on all training cases except one
2. predict the held-out training case
3. compute case-level error metrics
4. repeat for every training case
5. average the results to obtain one aggregate score

### Tuned parameters and ranking metric
The first search focuses on:
- STLSQ threshold
- STLSQ alpha
- polynomial degree

Candidates are ranked by the mean of `rmse_cl` and `rmse_cd` across the
leave-one-case-out validation loop.

This keeps the first search intentionally narrow and practical before adding
more parameters or more expensive search strategies.

### Output and apply policy
The utility writes:
- one overwriteable latest results table
- one overwriteable latest best-summary file
- one append-only tuning equations-history file

At the end of a tuning run, the utility prompts whether the best values should
be applied to `assets/model.py`.

If accepted, the live defaults in `model.py` are overwritten directly.
If declined, the utility remains recommendation-only and the project defaults
stay unchanged.

### Tuning-history purpose
The tuning equations-history file records the best candidate equations for each
search run. This complements the fit equations-history file by preserving how
the preferred model changes during the tuning process, which will be useful in
the written report when comparing the baseline and tuned formulations.

## 2026-04-12: Hyperparameter-tuning overhaul

### Revised search style
The tuning utility has been rewritten to follow a more explicit brute-force
style closer to the earlier coursework project.

The candidate ranges remain fixed at the top of the file, and the main search
now uses visible nested loops over:
- threshold
- alpha
- polynomial degree

Each candidate combination is printed before evaluation, and each validation
case is printed as it is tested.

### Reduced validation-case set
Instead of leaving out every available training case, the current tuning stage
now uses a reduced fixed validation subset:
- B1
- C1
- C4
- C5
- D1
- D4
- D5

For each candidate combination, the model is fit repeatedly while holding out
one of those selected cases at a time.

### Failure handling
Some candidate settings can produce numerically unstable models during the
simulation step.

The tuning utility now treats those combinations as failed candidates rather
than aborting the whole search. The failed holdout case and the failure message
are written into the tuning results table so unstable combinations can still be
reviewed afterwards.

### Final confirmation test
After the best candidate has been selected from the reduced training-case
validation set, the utility now performs one final confirmation test on the
current `T#` cases if they are available.

This confirmation pass is done only once for the winning parameter set, not for
every candidate combination. The resulting prediction table and summary table
are exported as separate tuning outputs.

## 2026-04-11: Analysis-stage baseline

### Objective of this increment
The analysis stage now provides an interactive Plotly-based view of both the
prepared dataset and the latest prediction outputs.

This first analysis increment is intended to answer two practical questions:
- what the available `cl`, `cd`, and `pitch` data look like across all cases
- how the latest confirmation predictions compare with the true trajectories

### Role of `assets/analyse.py`
`assets/analyse.py` is the fourth workflow stage and is now called directly by
`main.py`.

Its first version is intentionally view-first rather than export-heavy. The
stage reads the CSV outputs produced earlier in the workflow and opens the
interactive figures directly instead of writing a large set of saved plot files.

### Dataset-overview figures
The first plot family focuses on the available prepared data.

It includes:
- all-case overlays for `cl`, `cd`, and `pitch`
- grouped-by-category figures where cases are separated by the starting letter
- zoomed versions of both views over the first 0.25 seconds of each retained
  case window

For the zoomed views, the time axis is measured from the retained start of each
case so the first quarter-second can be compared consistently across cases.

### Prediction-assessment figures
The second plot family focuses on the latest confirmation predictions.

It includes:
- true-versus-predicted time-series overlays for `cl` and `cd`
- residual-versus-time plots for `cl` and `cd`
- parity plots for both predicted variables
- per-case RMSE and MAE bar charts based on the latest run summary

These figures are intended to provide a broad first inspection of model
behaviour before any deeper interpretation is added later.

### Deferred tuned comparison
Tuned-versus-untuned comparison plots are intentionally deferred.

The current analysis stage includes a clear placeholder note for that future
extension, but no tuned-comparison figures are implemented yet because the
tuned workflow and its output structure are not final yet.

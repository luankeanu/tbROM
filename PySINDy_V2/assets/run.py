from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import re

import numpy as np
import pandas as pd

from . import fit, model, read


"""
Section 1: Output paths, required schemas, and returned containers.
This section defines the run-stage output files and the minimum schema expected
before previously prepared confirmation outputs are trusted for reuse.
"""

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
RUN_ARCHIVE_DIR = OUTPUT_DIR / "run_archive"

LATEST_PREDICTIONS_FILENAME = "latest_confirmation_predictions.csv"
LATEST_SUMMARY_FILENAME = "latest_confirmation_run_summary.csv"

SIMULATION_INTEGRATOR_ATTEMPTS = (
    {"integrator": "solve_ivp", "integrator_kws": {"method": "LSODA", "rtol": 1e-12, "atol": 1e-12}},
    {"integrator": "solve_ivp", "integrator_kws": {"method": "BDF", "rtol": 1e-9, "atol": 1e-9}},
    {"integrator": "solve_ivp", "integrator_kws": {"method": "Radau", "rtol": 1e-9, "atol": 1e-9}},
    {"integrator": "solve_ivp", "integrator_kws": {"method": "RK45", "rtol": 1e-8, "atol": 1e-8}},
)

REQUIRED_CONFIRMATION_COLUMNS = {
    "source_file",
    "case_name",
    "case_group",
    "dataset_role",
    "is_confirmation",
    "time_step",
    "flow_time",
    "vy",
    "vx",
    "cl",
    "cd",
    "pitch",
}

REQUIRED_VALIDATION_COLUMNS = {
    "source_file",
    "case_name",
    "case_group",
    "dataset_role",
    "is_confirmation",
    "raw_rows",
    "retained_rows",
    "missing_values",
    "duplicate_time_rows",
    "monotonic_time",
    "is_valid",
    "validation_message",
}


"""
Container: PreparedConfirmationData.
Stores the confirmation table used by the run stage together with a note that
records whether the data came from cached prepared outputs or a fresh read pass.
"""


@dataclass(frozen=True)
class PreparedConfirmationData:
    confirmation_table: pd.DataFrame
    validation_table: pd.DataFrame
    source_mode: str


"""
Container: RunStageOutput.
Bundles the key outputs of the run stage so `main.py` can report what was
predicted and where the resulting files were written.
"""


@dataclass(frozen=True)
class RunStageOutput:
    run_index: int
    model_fit_index: int | None
    confirmation_case_count: int
    confirmation_row_count: int
    source_mode: str
    latest_predictions_path: Path
    archived_predictions_path: Path
    latest_summary_path: Path
    archived_summary_path: Path


"""
Section 2: Prepared confirmation-data reuse and validation.
These helpers let the run stage reuse the prepared confirmation CSV when it is
already available and trustworthy, avoiding an unnecessary rerun of `read.py`.
"""


"""
Function: _load_existing_confirmation_outputs.
Loads the prepared confirmation table and validation table from disk when both
files exist. If either file is missing, the run stage cannot reuse them.
"""


def _load_existing_confirmation_outputs() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    confirmation_path = read.OUTPUT_DIR / read.OUTPUT_FILENAMES["confirmation"]
    validation_path = read.OUTPUT_DIR / read.OUTPUT_FILENAMES["validation"]

    if not confirmation_path.exists() or not validation_path.exists():
        return None

    confirmation_table = pd.read_csv(confirmation_path)
    validation_table = pd.read_csv(validation_path)
    return confirmation_table, validation_table


"""
Function: _validate_reusable_confirmation_outputs.
Checks whether the cached confirmation data still matches the current run-stage
expectations before the stage commits to using it for prediction.
"""


def _validate_reusable_confirmation_outputs(
    confirmation_table: pd.DataFrame,
    validation_table: pd.DataFrame,
) -> tuple[bool, str]:
    missing_confirmation_columns = REQUIRED_CONFIRMATION_COLUMNS - set(confirmation_table.columns)
    if missing_confirmation_columns:
        return False, (
            "missing confirmation columns: "
            f"{sorted(missing_confirmation_columns)}"
        )

    missing_validation_columns = REQUIRED_VALIDATION_COLUMNS - set(validation_table.columns)
    if missing_validation_columns:
        return False, f"missing validation columns: {sorted(missing_validation_columns)}"

    if confirmation_table.empty:
        return False, "confirmation table is empty"

    if confirmation_table["case_name"].nunique() == 0:
        return False, "no confirmation cases were found in the prepared confirmation table"

    if not confirmation_table["dataset_role"].eq("confirmation").all():
        return False, "prepared confirmation table contains non-confirmation rows"

    confirmation_flags = confirmation_table["is_confirmation"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if not confirmation_flags.fillna(False).all():
        return False, "prepared confirmation table contains non-confirmation flags"

    if validation_table.empty:
        return False, "validation table is empty"

    validation_is_valid = validation_table["is_valid"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if not validation_is_valid.fillna(False).all():
        return False, "validation table reports invalid source files"

    confirmation_rows = validation_table.loc[
        validation_table["dataset_role"] == "confirmation"
    ]
    if confirmation_rows.empty:
        return False, "validation table contains no confirmation cases"

    return True, "prepared confirmation outputs passed reuse checks"


"""
Function: _obtain_prepared_confirmation_data.
Implements the run-stage reuse policy. It reuses cached prepared confirmation
outputs when they pass strict checks; otherwise it reruns the read stage.
"""


def _obtain_prepared_confirmation_data() -> PreparedConfirmationData:
    loaded_outputs = _load_existing_confirmation_outputs()

    if loaded_outputs is not None:
        confirmation_table, validation_table = loaded_outputs
        outputs_are_valid, message = _validate_reusable_confirmation_outputs(
            confirmation_table=confirmation_table,
            validation_table=validation_table,
        )
        if outputs_are_valid:
            print("[run] reusing prepared confirmation outputs from disk", flush=True)
            return PreparedConfirmationData(
                confirmation_table=confirmation_table,
                validation_table=validation_table,
                source_mode="reused_prepared_outputs",
            )
        print(f"[run] prepared confirmation outputs were not reusable: {message}", flush=True)

    print("[run] running read stage to refresh confirmation outputs", flush=True)
    read_output = read.run_read_stage()
    return PreparedConfirmationData(
        confirmation_table=read_output.confirmation_table.copy(),
        validation_table=read_output.validation_table.copy(),
        source_mode="refreshed_via_read_stage",
    )


"""
Section 3: Model loading and simulation helpers.
These helpers load the latest fitted model and run one confirmation-case
simulation at a time using the case pitch history as the control input.
"""


"""
Function: _load_latest_fitted_model.
Loads the canonical latest fitted-model pickle from disk and fails clearly if
the model does not exist or cannot be unpickled in the active environment.
"""


def _load_latest_fitted_model() -> tuple[object, int | None]:
    latest_model_path = fit.OUTPUT_DIR / fit.LATEST_MODEL_FILENAME
    if not latest_model_path.exists():
        raise FileNotFoundError(
            "Latest fitted model not found. Run the fit stage before the run stage."
        )

    try:
        with latest_model_path.open("rb") as handle:
            fitted_model = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(
            "The latest fitted model could not be loaded. "
            "Run the fit stage again before the run stage."
        ) from exc

    fit_summary_path = fit.OUTPUT_DIR / fit.LATEST_FIT_SUMMARY_FILENAME
    model_fit_index: int | None = None
    if fit_summary_path.exists():
        fit_summary_payload = json.loads(fit_summary_path.read_text(encoding="utf-8"))
        fit_index_value = fit_summary_payload.get("fit_index")
        if isinstance(fit_index_value, int):
            model_fit_index = fit_index_value

    return fitted_model, model_fit_index


"""
Function: _control_function.
Creates an interpolated control function for PySINDy simulation so the model
can query pitch values continuously over the confirmation-case timeline.
"""


def _control_function(time: np.ndarray, control: np.ndarray):
    time = np.asarray(time, dtype=float)
    control = np.asarray(control, dtype=float)

    def u_fun(query_time):
        query = np.asarray(query_time, dtype=float)
        query_1d = np.atleast_1d(query)
        clipped = np.clip(query_1d, time[0], time[-1])
        interpolated = np.column_stack(
            [np.interp(clipped, time, control[:, idx]) for idx in range(control.shape[1])]
        )
        if query.ndim == 0:
            return interpolated[0]
        return interpolated

    return u_fun


"""
Function: _format_case_size_message.
Builds a small pre-run message that reports how many retained simulation steps
exist in the current case before the solver starts its one-shot simulation.
"""


def _format_case_size_message(
    case_name: str,
    total_steps: int,
) -> str:
    return f"[run] {case_name}: {total_steps} retained solver steps to process"


"""
Function: _format_integrator_message.
Builds a compact status line for the current solver attempt so the user can see
which integration method is being tried on the active confirmation case.
"""


def _format_integrator_message(case_name: str, method_name: str) -> str:
    return f"[run] {case_name}: trying integrator {method_name}"


"""
Function: _simulate_with_fallback_integrators.
Runs the PySINDy simulation using a short ordered list of SciPy integrators.
This keeps the standard simulation path intact while allowing the run stage to
recover when one solver becomes numerically unstable on a difficult case.
"""


def _simulate_with_fallback_integrators(
    fitted_model: object,
    initial_state: np.ndarray,
    time_values: np.ndarray,
    control_values: np.ndarray,
    case_name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    control_function = _control_function(time_values, control_values)
    last_error: Exception | None = None

    for attempt in SIMULATION_INTEGRATOR_ATTEMPTS:
        method_name = str(attempt["integrator_kws"]["method"])
        print(_format_integrator_message(case_name, method_name), flush=True)
        try:
            prediction_array = np.asarray(
                fitted_model.simulate(
                    x0=initial_state,
                    t=time_values,
                    u=control_function,
                    integrator=attempt["integrator"],
                    integrator_kws=attempt["integrator_kws"],
                ),
                dtype=float,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"[run] {case_name}: integrator {method_name} failed",
                flush=True,
            )
            continue

        if not np.isfinite(prediction_array).all():
            last_error = ValueError(
                f"Integrator {method_name} produced non-finite prediction values."
            )
            print(
                f"[run] {case_name}: integrator {method_name} produced non-finite values",
                flush=True,
            )
            continue

        if prediction_array.shape != expected_shape:
            last_error = ValueError(
                f"Integrator {method_name} returned shape {prediction_array.shape} "
                f"instead of expected {expected_shape}."
            )
            print(
                f"[run] {case_name}: integrator {method_name} returned partial output",
                flush=True,
            )
            continue

        return prediction_array

    raise RuntimeError(
        f"All simulation integrators failed for case {case_name}. "
        "The current fitted model appears numerically unstable on this case."
    ) from last_error


"""
Function: _simulate_case.
Runs one confirmation-case simulation using the original PySINDy `simulate(...)`
path. This preserves the stable numerical behavior used earlier in the project.
"""


def _simulate_case(fitted_model: object, case_table: pd.DataFrame) -> np.ndarray:
    state_columns = model.get_state_columns()
    control_columns = model.get_control_columns()
    time_column = model.get_time_column()

    ordered_case_table = case_table.sort_values(by=time_column).reset_index(drop=True)
    case_name = str(ordered_case_table["case_name"].iloc[0])
    initial_state = ordered_case_table[state_columns].iloc[0].to_numpy(dtype=float)
    time_values = ordered_case_table[time_column].to_numpy(dtype=float)
    control_values = ordered_case_table[control_columns].to_numpy(dtype=float)

    total_steps = max(len(ordered_case_table) - 1, 0)
    print(
        _format_case_size_message(
            case_name=case_name,
            total_steps=total_steps,
        ),
        flush=True,
    )

    prediction_array = _simulate_with_fallback_integrators(
        fitted_model=fitted_model,
        initial_state=initial_state,
        time_values=time_values,
        control_values=control_values,
        case_name=case_name,
        expected_shape=(len(ordered_case_table), len(state_columns)),
    )

    if prediction_array.shape != (len(ordered_case_table), len(state_columns)):
        raise ValueError(
            f"Prediction shape {prediction_array.shape} does not match "
            f"expected shape {(len(ordered_case_table), len(state_columns))}"
        )

    return prediction_array


"""
Section 4: Prediction-table and summary builders.
These helpers convert the per-case simulation output into the latest and
archived files that the run stage will export.
"""


"""
Function: _format_progress_message.
Builds a compact textual progress bar for the confirmation-case loop so long
run-stage executions show how many cases are done and how many remain.
"""


def _format_progress_message(
    completed_cases: int,
    total_cases: int,
    case_name: str,
    bar_width: int = 12,
) -> str:
    filled_width = int(bar_width * completed_cases / total_cases) if total_cases else 0
    progress_bar = "#" * filled_width + "-" * (bar_width - filled_width)
    percentage = (100.0 * completed_cases / total_cases) if total_cases else 0.0
    return (
        f"[run] [{progress_bar}] "
        f"{completed_cases}/{total_cases} "
        f"({percentage:5.1f}%) "
        f"{case_name} done"
    )


"""
Function: _format_case_start_message.
Builds a small pre-run message for each confirmation case so the user sees
immediately which case has started even before the simulation finishes.
"""


def _format_case_start_message(
    case_number: int,
    total_cases: int,
    case_name: str,
) -> str:
    return f"[run] starting case {case_number}/{total_cases}: {case_name}"


"""
Function: _prediction_rows_for_case.
Builds the row-level prediction output for one confirmation case by attaching
predicted values and residuals to the prepared confirmation trajectory.
"""


def _prediction_rows_for_case(
    case_table: pd.DataFrame,
    prediction: np.ndarray,
    run_index: int,
    model_fit_index: int | None,
) -> pd.DataFrame:
    ordered_case_table = case_table.sort_values(by=model.get_time_column()).reset_index(drop=True)
    state_columns = model.get_state_columns()

    output_table = ordered_case_table.copy()
    output_table.insert(0, "run_index", run_index)
    output_table.insert(1, "model_fit_index", model_fit_index)

    for state_index, state_name in enumerate(state_columns):
        predicted_column = f"{state_name}_pred"
        residual_column = f"{state_name}_residual"
        output_table[predicted_column] = prediction[:, state_index]
        output_table[residual_column] = (
            output_table[state_name].to_numpy() - prediction[:, state_index]
        )

    return output_table


"""
Function: _summary_row_for_case.
Creates one compact metrics row for a single confirmation case. The summary is
kept intentionally simple because deeper interpretation belongs to `analyse.py`.
"""


def _summary_row_for_case(
    case_table: pd.DataFrame,
    prediction: np.ndarray,
    run_index: int,
    model_fit_index: int | None,
) -> dict[str, object]:
    ordered_case_table = case_table.sort_values(by=model.get_time_column()).reset_index(drop=True)
    cl_true = ordered_case_table["cl"].to_numpy()
    cd_true = ordered_case_table["cd"].to_numpy()
    cl_pred = prediction[:, 0]
    cd_pred = prediction[:, 1]

    cl_residual = cl_true - cl_pred
    cd_residual = cd_true - cd_pred

    return {
        "run_index": run_index,
        "model_fit_index": model_fit_index,
        "source_file": ordered_case_table["source_file"].iloc[0],
        "case_name": ordered_case_table["case_name"].iloc[0],
        "case_group": ordered_case_table["case_group"].iloc[0],
        "dataset_role": ordered_case_table["dataset_role"].iloc[0],
        "is_confirmation": ordered_case_table["is_confirmation"].iloc[0],
        "rows": int(len(ordered_case_table)),
        "rmse_cl": float(np.sqrt(np.mean(np.square(cl_residual)))),
        "rmse_cd": float(np.sqrt(np.mean(np.square(cd_residual)))),
        "mae_cl": float(np.mean(np.abs(cl_residual))),
        "mae_cd": float(np.mean(np.abs(cd_residual))),
        "mean_error_cl": float(np.mean(cl_residual)),
        "mean_error_cd": float(np.mean(cd_residual)),
        "cl_true_min": float(np.min(cl_true)),
        "cl_true_max": float(np.max(cl_true)),
        "cl_pred_min": float(np.min(cl_pred)),
        "cl_pred_max": float(np.max(cl_pred)),
        "cd_true_min": float(np.min(cd_true)),
        "cd_true_max": float(np.max(cd_true)),
        "cd_pred_min": float(np.min(cd_pred)),
        "cd_pred_max": float(np.max(cd_pred)),
    }


"""
Function: _build_run_outputs.
Runs prediction for every confirmation case and returns the row-level
predictions table plus the per-case summary table.
"""


def _build_run_outputs(
    fitted_model: object,
    confirmation_table: pd.DataFrame,
    run_index: int,
    model_fit_index: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    total_cases = int(confirmation_table["case_name"].nunique())

    for case_number, (case_name, case_table) in enumerate(
        confirmation_table.groupby("case_name", sort=True),
        start=1,
    ):
        print(
            _format_case_start_message(
                case_number=case_number,
                total_cases=total_cases,
                case_name=case_name,
            ),
            flush=True,
        )
        prediction = _simulate_case(fitted_model=fitted_model, case_table=case_table)
        prediction_tables.append(
            _prediction_rows_for_case(
                case_table=case_table,
                prediction=prediction,
                run_index=run_index,
                model_fit_index=model_fit_index,
            )
        )
        summary_rows.append(
            _summary_row_for_case(
                case_table=case_table,
                prediction=prediction,
                run_index=run_index,
                model_fit_index=model_fit_index,
            )
        )
        print(
            _format_progress_message(
                completed_cases=case_number,
                total_cases=total_cases,
                case_name=case_name,
            ),
            flush=True,
        )

    prediction_table = pd.concat(prediction_tables, ignore_index=True)
    summary_table = pd.DataFrame(summary_rows).sort_values(by="case_name").reset_index(drop=True)
    return prediction_table, summary_table


"""
Section 5: Output saving and run numbering.
These helpers create the run-archive directory, determine the next run number,
and save both the latest and archived prediction outputs.
"""


"""
Function: _ensure_run_output_directories.
Creates the directories needed for the run-stage latest files and archives.
"""


def _ensure_run_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


"""
Function: _next_run_index.
Reads the run-archive directory and returns the next sequential run number.
"""


def _next_run_index() -> int:
    _ensure_run_output_directories()
    archive_names = [path.name for path in RUN_ARCHIVE_DIR.glob("run_*_summary.csv")]
    matches = [
        int(match.group(1))
        for name in archive_names
        if (match := re.match(r"run_(\d+)_summary\.csv$", name))
    ]
    return max(matches) + 1 if matches else 1


"""
Function: _save_run_outputs.
Writes the latest prediction outputs and the archived copies for the current run.
"""


def _save_run_outputs(
    prediction_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    run_index: int,
) -> tuple[Path, Path, Path, Path]:
    _ensure_run_output_directories()

    latest_predictions_path = OUTPUT_DIR / LATEST_PREDICTIONS_FILENAME
    latest_summary_path = OUTPUT_DIR / LATEST_SUMMARY_FILENAME
    archived_predictions_path = RUN_ARCHIVE_DIR / f"run_{run_index:03d}_predictions.csv"
    archived_summary_path = RUN_ARCHIVE_DIR / f"run_{run_index:03d}_summary.csv"

    prediction_table.to_csv(latest_predictions_path, index=False)
    summary_table.to_csv(latest_summary_path, index=False)
    prediction_table.to_csv(archived_predictions_path, index=False)
    summary_table.to_csv(archived_summary_path, index=False)

    return (
        latest_predictions_path,
        archived_predictions_path,
        latest_summary_path,
        archived_summary_path,
    )


"""
Section 6: Public entry point for the run stage.
This final section coordinates model loading, confirmation-case simulation, and
the export of both latest and archived prediction outputs.
"""


"""
Function: run_prediction_stage.
Runs the V2 confirmation-prediction stage. The function is intentionally
limited to generating predictions and compact summaries; deeper interpretation
and plotting are left to the analysis stage.
"""


def run_prediction_stage() -> RunStageOutput:
    fitted_model, model_fit_index = _load_latest_fitted_model()
    prepared_confirmation_data = _obtain_prepared_confirmation_data()

    if prepared_confirmation_data.confirmation_table.empty:
        raise ValueError("No confirmation cases are available for the run stage.")

    run_index = _next_run_index()
    prediction_table, summary_table = _build_run_outputs(
        fitted_model=fitted_model,
        confirmation_table=prepared_confirmation_data.confirmation_table,
        run_index=run_index,
        model_fit_index=model_fit_index,
    )

    (
        latest_predictions_path,
        archived_predictions_path,
        latest_summary_path,
        archived_summary_path,
    ) = _save_run_outputs(
        prediction_table=prediction_table,
        summary_table=summary_table,
        run_index=run_index,
    )

    print(f"[run] completed run stage {run_index}", flush=True)
    print(
        f"[run] confirmation cases: {summary_table['case_name'].nunique()}",
        flush=True,
    )
    print(
        f"[run] confirmation rows: {len(prediction_table)}",
        flush=True,
    )
    print(f"[run] latest predictions saved to {latest_predictions_path}", flush=True)

    return RunStageOutput(
        run_index=run_index,
        model_fit_index=model_fit_index,
        confirmation_case_count=int(summary_table["case_name"].nunique()),
        confirmation_row_count=int(len(prediction_table)),
        source_mode=prepared_confirmation_data.source_mode,
        latest_predictions_path=latest_predictions_path,
        archived_predictions_path=archived_predictions_path,
        latest_summary_path=latest_summary_path,
        archived_summary_path=archived_summary_path,
    )


if __name__ == "__main__":
    run_prediction_stage()

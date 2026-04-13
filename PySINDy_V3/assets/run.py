from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import re
import threading
import time

import numpy as np
import pandas as pd

from . import fit, model, read


"""
Section 1: Output paths, required schemas, and returned containers.
The V3 run stage reuses the existing output format, but under the hood it loads
two separate one-output PySINDy models and joins their predictions.
"""

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
RUN_ARCHIVE_DIR = OUTPUT_DIR / "run_archive"

LATEST_PREDICTIONS_FILENAME = "latest_confirmation_predictions.csv"
LATEST_SUMMARY_FILENAME = "latest_confirmation_run_summary.csv"
SIMULATION_HEARTBEAT_SECONDS = 5.0

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
    "pitch_rate",
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


@dataclass(frozen=True)
class PreparedConfirmationData:
    confirmation_table: pd.DataFrame
    validation_table: pd.DataFrame
    source_mode: str


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
These helpers are unchanged in spirit from V2 and simply load the confirmation
tables used by the two-model simulation stage.
"""


def _load_existing_confirmation_outputs() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    confirmation_path = read.OUTPUT_DIR / read.OUTPUT_FILENAMES["confirmation"]
    validation_path = read.OUTPUT_DIR / read.OUTPUT_FILENAMES["validation"]
    if not confirmation_path.exists() or not validation_path.exists():
        return None
    return pd.read_csv(confirmation_path), pd.read_csv(validation_path)


def _validate_reusable_confirmation_outputs(
    confirmation_table: pd.DataFrame,
    validation_table: pd.DataFrame,
) -> tuple[bool, str]:
    missing_confirmation_columns = REQUIRED_CONFIRMATION_COLUMNS - set(confirmation_table.columns)
    if missing_confirmation_columns:
        return False, f"missing confirmation columns: {sorted(missing_confirmation_columns)}"

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

    validation_is_valid = validation_table["is_valid"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if not validation_is_valid.fillna(False).all():
        return False, "validation table reports invalid source files"

    confirmation_rows = validation_table.loc[validation_table["dataset_role"] == "confirmation"]
    if confirmation_rows.empty:
        return False, "validation table contains no confirmation cases"

    return True, "prepared confirmation outputs passed reuse checks"


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
These helpers load the bundled `cl`/`cd` models and simulate each target
separately before joining the outputs into one prediction table.
"""


def _load_latest_fitted_model() -> tuple[dict[str, object], int | None]:
    latest_model_path = fit.OUTPUT_DIR / fit.LATEST_MODEL_FILENAME
    if not latest_model_path.exists():
        raise FileNotFoundError(
            "Latest fitted model not found. Run the fit stage before the run stage."
        )

    try:
        with latest_model_path.open("rb") as handle:
            fitted_models = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(
            "The latest fitted model could not be loaded. Run the fit stage again before the run stage."
        ) from exc

    if not isinstance(fitted_models, dict) or set(fitted_models.keys()) != set(model.get_target_columns()):
        raise RuntimeError("The latest fitted model bundle does not contain the expected cl/cd models.")

    fit_summary_path = fit.OUTPUT_DIR / fit.LATEST_FIT_SUMMARY_FILENAME
    model_fit_index: int | None = None
    if fit_summary_path.exists():
        fit_summary_payload = json.loads(fit_summary_path.read_text(encoding="utf-8"))
        fit_index_value = fit_summary_payload.get("fit_index")
        if isinstance(fit_index_value, int):
            model_fit_index = fit_index_value

    return fitted_models, model_fit_index


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


def _format_case_size_message(case_name: str, total_steps: int) -> str:
    return f"[run] {case_name}: {total_steps} retained solver steps to process"


def _format_integrator_message(
    case_name: str,
    target_column: str,
    method_name: str,
    attempt_index: int,
    total_attempts: int,
    integrator_kws: dict[str, object],
) -> str:
    return (
        f"[run] {case_name}/{target_column}: trying integrator {method_name} "
        f"({attempt_index}/{total_attempts}) with {integrator_kws}"
    )


def _format_heartbeat_message(
    case_name: str,
    target_column: str,
    method_name: str,
    elapsed_seconds: float,
    total_steps: int,
) -> str:
    return (
        f"[run] {case_name}/{target_column}: {method_name} still running after "
        f"{elapsed_seconds:.0f}s on {total_steps} retained steps"
    )


def _format_integrator_failure_message(
    case_name: str,
    target_column: str,
    method_name: str,
    exc: Exception,
) -> str:
    return (
        f"[run] {case_name}/{target_column}: integrator {method_name} failed with "
        f"{type(exc).__name__}: {exc}"
    )


def _format_partial_output_message(
    case_name: str,
    target_column: str,
    method_name: str,
    returned_shape: tuple[int, ...],
    expected_shape: tuple[int, int],
) -> str:
    return (
        f"[run] {case_name}/{target_column}: integrator {method_name} returned partial output "
        f"{returned_shape}, expected {expected_shape}"
    )


@dataclass
class _SimulationHeartbeat:
    case_name: str
    target_column: str
    method_name: str
    total_steps: int
    interval_seconds: float = SIMULATION_HEARTBEAT_SECONDS
    _stop_event: threading.Event | None = None
    _thread: threading.Thread | None = None
    _start_time: float = 0.0

    def start(self) -> None:
        self._stop_event = threading.Event()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 1.0)

    def _run(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.wait(self.interval_seconds):
            print(
                _format_heartbeat_message(
                    case_name=self.case_name,
                    target_column=self.target_column,
                    method_name=self.method_name,
                    elapsed_seconds=time.monotonic() - self._start_time,
                    total_steps=self.total_steps,
                ),
                flush=True,
            )


def _reshape_target_prediction(
    prediction_array: np.ndarray,
    expected_rows: int,
    target_column: str,
) -> np.ndarray:
    reshaped_prediction = np.asarray(prediction_array, dtype=float)
    if reshaped_prediction.ndim == 1:
        reshaped_prediction = reshaped_prediction.reshape(-1, 1)
    if reshaped_prediction.shape != (expected_rows, 1):
        raise ValueError(
            f"Prediction shape {reshaped_prediction.shape} does not match expected shape "
            f"{(expected_rows, 1)} for target {target_column}"
        )
    return reshaped_prediction


def _simulate_target_with_fallback_integrators(
    fitted_model: object,
    initial_state: np.ndarray,
    time_values: np.ndarray,
    control_values: np.ndarray,
    case_name: str,
    target_column: str,
    total_steps: int,
) -> np.ndarray:
    control_function = _control_function(time_values, control_values)
    last_error: Exception | None = None

    for attempt_index, attempt in enumerate(SIMULATION_INTEGRATOR_ATTEMPTS, start=1):
        method_name = str(attempt["integrator_kws"]["method"])
        print(
            _format_integrator_message(
                case_name=case_name,
                target_column=target_column,
                method_name=method_name,
                attempt_index=attempt_index,
                total_attempts=len(SIMULATION_INTEGRATOR_ATTEMPTS),
                integrator_kws=attempt["integrator_kws"],
            ),
            flush=True,
        )

        heartbeat = _SimulationHeartbeat(
            case_name=case_name,
            target_column=target_column,
            method_name=method_name,
            total_steps=total_steps,
        )
        heartbeat.start()
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
            heartbeat.stop()
            last_error = exc
            print(
                _format_integrator_failure_message(
                    case_name=case_name,
                    target_column=target_column,
                    method_name=method_name,
                    exc=exc,
                ),
                flush=True,
            )
            continue
        heartbeat.stop()

        if not np.isfinite(prediction_array).all():
            last_error = ValueError(
                f"Integrator {method_name} produced non-finite prediction values."
            )
            print(
                f"[run] {case_name}/{target_column}: integrator {method_name} produced non-finite values",
                flush=True,
            )
            continue

        try:
            return _reshape_target_prediction(
                prediction_array=prediction_array,
                expected_rows=len(time_values),
                target_column=target_column,
            )
        except Exception as exc:
            last_error = exc
            print(
                _format_partial_output_message(
                    case_name=case_name,
                    target_column=target_column,
                    method_name=method_name,
                    returned_shape=prediction_array.shape,
                    expected_shape=(len(time_values), 1),
                ),
                flush=True,
            )
            continue

    raise RuntimeError(
        f"All simulation integrators failed for case {case_name}, target {target_column}."
    ) from last_error


def _simulate_case(
    fitted_models: dict[str, object],
    case_table: pd.DataFrame,
) -> np.ndarray:
    control_columns = model.get_control_columns()
    time_column = model.get_time_column()
    ordered_case_table = case_table.sort_values(by=time_column).reset_index(drop=True)
    case_name = str(ordered_case_table["case_name"].iloc[0])
    time_values = ordered_case_table[time_column].to_numpy(dtype=float)
    control_values = ordered_case_table[control_columns].to_numpy(dtype=float)
    total_steps = max(len(ordered_case_table) - 1, 0)

    print(_format_case_size_message(case_name, total_steps), flush=True)

    target_predictions: list[np.ndarray] = []
    for target_column in model.get_target_columns():
        initial_state = ordered_case_table[[target_column]].iloc[0].to_numpy(dtype=float)
        target_predictions.append(
            _simulate_target_with_fallback_integrators(
                fitted_model=fitted_models[target_column],
                initial_state=initial_state,
                time_values=time_values,
                control_values=control_values,
                case_name=case_name,
                target_column=target_column,
                total_steps=total_steps,
            )
        )

    return np.column_stack([prediction[:, 0] for prediction in target_predictions])


"""
Section 4: Prediction-table and summary builders.
These helpers stay close to V2 because the final exported format is still one
joined table containing both predicted targets.
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


def _format_case_start_message(case_number: int, total_cases: int, case_name: str) -> str:
    return f"[run] starting case {case_number}/{total_cases}: {case_name}"


def _prediction_rows_for_case(
    case_table: pd.DataFrame,
    prediction: np.ndarray,
    run_index: int,
    model_fit_index: int | None,
) -> pd.DataFrame:
    ordered_case_table = case_table.sort_values(by=model.get_time_column()).reset_index(drop=True)
    output_table = ordered_case_table.copy()
    output_table.insert(0, "run_index", run_index)
    output_table.insert(1, "model_fit_index", model_fit_index)

    for state_index, state_name in enumerate(model.get_state_columns()):
        output_table[f"{state_name}_pred"] = prediction[:, state_index]
        output_table[f"{state_name}_residual"] = (
            output_table[state_name].to_numpy() - prediction[:, state_index]
        )

    return output_table


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


def _build_run_outputs(
    fitted_models: dict[str, object],
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
        print(_format_case_start_message(case_number, total_cases, case_name), flush=True)
        prediction = _simulate_case(fitted_models=fitted_models, case_table=case_table)
        prediction_tables.append(
            _prediction_rows_for_case(case_table, prediction, run_index, model_fit_index)
        )
        summary_rows.append(
            _summary_row_for_case(case_table, prediction, run_index, model_fit_index)
        )
        print(
            _format_progress_message(case_number, total_cases, case_name),
            flush=True,
        )

    prediction_table = pd.concat(prediction_tables, ignore_index=True)
    summary_table = pd.DataFrame(summary_rows).sort_values(by="case_name").reset_index(drop=True)
    return prediction_table, summary_table


"""
Section 5: Output saving and public run entry point.
This section remains compatible with the V2 output naming so analysis can join
the two target-model predictions without extra post-processing.
"""


def _ensure_run_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _next_run_index() -> int:
    _ensure_run_output_directories()
    archive_names = [path.name for path in RUN_ARCHIVE_DIR.glob("run_*_summary.csv")]
    matches = [
        int(match.group(1))
        for name in archive_names
        if (match := re.match(r"run_(\d+)_summary\.csv$", name))
    ]
    return max(matches) + 1 if matches else 1


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


def run_prediction_stage() -> RunStageOutput:
    fitted_models, model_fit_index = _load_latest_fitted_model()
    prepared_confirmation_data = _obtain_prepared_confirmation_data()

    if prepared_confirmation_data.confirmation_table.empty:
        raise ValueError("No confirmation cases are available for the run stage.")

    run_index = _next_run_index()
    prediction_table, summary_table = _build_run_outputs(
        fitted_models=fitted_models,
        confirmation_table=prepared_confirmation_data.confirmation_table,
        run_index=run_index,
        model_fit_index=model_fit_index,
    )

    (
        latest_predictions_path,
        archived_predictions_path,
        latest_summary_path,
        archived_summary_path,
    ) = _save_run_outputs(prediction_table, summary_table, run_index)

    print(f"[run] completed run stage {run_index}", flush=True)
    print(f"[run] confirmation cases: {summary_table['case_name'].nunique()}", flush=True)
    print(f"[run] confirmation rows: {len(prediction_table)}", flush=True)
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import json
import pickle
import re

import numpy as np
import pandas as pd

from . import model, read


"""
Section 1: Output paths, required schemas, and returned containers.
This section defines the files produced by the fit stage and the minimum schema
required before previously prepared outputs are trusted for reuse.
"""

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_ARCHIVE_DIR = OUTPUT_DIR / "model_archive"

LATEST_MODEL_FILENAME = "latest_fitted_model.pkl"
LATEST_FIT_SUMMARY_FILENAME = "fit_summary_latest.json"
EQUATIONS_HISTORY_FILENAME = "equations_history.txt"

REQUIRED_TRAIN_COLUMNS = {
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
Container: PreparedTrainingData.
Stores the prepared training data that the fit stage will consume, together
with a short note describing whether the data was reused or regenerated.
"""


@dataclass(frozen=True)
class PreparedTrainingData:
    train_table: pd.DataFrame
    validation_table: pd.DataFrame
    source_mode: str


"""
Container: FitStageOutput.
Bundles the fitted model and the key files written during training so later
stages and `main.py` can report exactly what happened during the fit run.
"""


@dataclass(frozen=True)
class FitStageOutput:
    fitted_model: object
    fit_index: int
    train_case_count: int
    train_row_count: int
    source_mode: str
    latest_model_path: Path
    archived_model_path: Path
    summary_path: Path
    equations_history_path: Path


"""
Section 2: Prepared-data reuse and validation.
These helpers decide whether the fit stage can safely use the existing prepared
CSV outputs or whether it must rerun the read stage first.
"""


"""
Function: _load_existing_outputs.
Loads the prepared training table and the validation table from disk if both
files exist. If either file is missing, the fit stage cannot reuse them.
"""


def _load_existing_outputs() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    train_path = read.OUTPUT_DIR / read.OUTPUT_FILENAMES["train"]
    validation_path = read.OUTPUT_DIR / read.OUTPUT_FILENAMES["validation"]

    if not train_path.exists() or not validation_path.exists():
        return None

    train_table = pd.read_csv(train_path)
    validation_table = pd.read_csv(validation_path)
    return train_table, validation_table


"""
Function: _validate_reusable_outputs.
Checks whether the existing prepared outputs match the current fit-stage
expectations. The goal is to reuse saved work only when the data is complete
enough to train the model safely.
"""


def _validate_reusable_outputs(
    train_table: pd.DataFrame,
    validation_table: pd.DataFrame,
) -> tuple[bool, str]:
    missing_train_columns = REQUIRED_TRAIN_COLUMNS - set(train_table.columns)
    if missing_train_columns:
        return False, f"missing training columns: {sorted(missing_train_columns)}"

    missing_validation_columns = REQUIRED_VALIDATION_COLUMNS - set(validation_table.columns)
    if missing_validation_columns:
        return False, f"missing validation columns: {sorted(missing_validation_columns)}"

    if train_table.empty:
        return False, "training table is empty"

    if "case_name" not in train_table.columns or train_table["case_name"].nunique() == 0:
        return False, "no training cases were found in the prepared training table"

    if "dataset_role" not in train_table.columns or not train_table["dataset_role"].eq("train").all():
        return False, "prepared training table contains non-training rows"

    if "is_confirmation" not in train_table.columns:
        return False, "prepared training table is missing the is_confirmation column"

    train_is_confirmation = train_table["is_confirmation"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if train_is_confirmation.fillna(True).any():
        return False, "prepared training table contains confirmation rows"

    if validation_table.empty:
        return False, "validation table is empty"

    if "is_valid" not in validation_table.columns:
        return False, "validation table is missing the is_valid column"

    validation_is_valid = validation_table["is_valid"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if not validation_is_valid.fillna(False).all():
        return False, "validation table reports invalid source files"

    training_rows = validation_table.loc[validation_table["dataset_role"] == "train"]
    if training_rows.empty:
        return False, "validation table contains no training cases"

    return True, "prepared outputs passed reuse checks"


"""
Function: _obtain_prepared_training_data.
Implements the fit-stage reuse policy. It reuses saved prepared outputs when
they pass strict checks; otherwise it reruns the read stage to regenerate them.
"""


def _obtain_prepared_training_data() -> PreparedTrainingData:
    loaded_outputs = _load_existing_outputs()

    if loaded_outputs is not None:
        train_table, validation_table = loaded_outputs
        outputs_are_valid, message = _validate_reusable_outputs(
            train_table=train_table,
            validation_table=validation_table,
        )
        if outputs_are_valid:
            print("[fit] reusing prepared outputs from disk", flush=True)
            return PreparedTrainingData(
                train_table=train_table,
                validation_table=validation_table,
                source_mode="reused_prepared_outputs",
            )
        print(f"[fit] prepared outputs were not reusable: {message}", flush=True)

    print("[fit] running read stage to refresh prepared outputs", flush=True)
    read_output = read.run_read_stage()
    return PreparedTrainingData(
        train_table=read_output.train_table.copy(),
        validation_table=read_output.validation_table.copy(),
        source_mode="refreshed_via_read_stage",
    )


"""
Section 3: Training-data preparation for PySINDy.
The fit stage consumes the combined prepared training table, but PySINDy should
receive one trajectory per case. These helpers rebuild the grouped arrays.
"""


"""
Function: _build_training_trajectories.
Splits the combined training table back into per-case state, control, and time
arrays so the model can be fit across multiple trajectories.
"""


def _build_training_trajectories(
    train_table: pd.DataFrame,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    state_columns = model.get_state_columns()
    control_columns = model.get_control_columns()
    time_column = model.get_time_column()

    x_train: list[np.ndarray] = []
    u_train: list[np.ndarray] = []
    t_train: list[np.ndarray] = []
    case_names: list[str] = []

    for case_name, case_table in train_table.groupby("case_name", sort=True):
        ordered_case_table = case_table.sort_values(by=time_column).reset_index(drop=True)
        x_train.append(ordered_case_table[state_columns].to_numpy())
        u_train.append(ordered_case_table[control_columns].to_numpy())
        t_train.append(ordered_case_table[time_column].to_numpy())
        case_names.append(case_name)

    return x_train, u_train, t_train, case_names


"""
Function: _fit_model.
Builds the canonical untrained model and fits it on the full training set. The
helper remains defensive around PySINDy API differences for multi-trajectory
fitting, while keeping the stage itself focused on fitting only once.
"""


def _fit_model(train_table: pd.DataFrame):
    fitted_model = model.build_pysindy_model()
    x_train, u_train, t_train, _ = _build_training_trajectories(train_table)
    feature_names = model.get_state_columns() + model.get_control_columns()

    fit_signature = inspect.signature(fitted_model.fit)
    fit_kwargs = {
        "u": u_train,
        "t": t_train,
    }

    if "multiple_trajectories" in fit_signature.parameters:
        fit_kwargs["multiple_trajectories"] = True
    if "feature_names" in fit_signature.parameters:
        fit_kwargs["feature_names"] = feature_names

    fitted_model.fit(x_train, **fit_kwargs)
    return fitted_model


"""
Section 4: Model summaries and equations-history helpers.
These helpers convert the fitted model into readable exported information while
keeping the append-only equations history separate from overwriteable outputs.
"""


"""
Function: _extract_equations.
Returns the discovered equations in a consistently formatted list so they can
be written to both the fit summary and the append-only history file.
"""


def _extract_equations(fitted_model: object) -> list[str]:
    equations = fitted_model.equations()
    state_columns = model.get_state_columns()
    formatted_equations = []

    for index, equation in enumerate(equations):
        state_name = state_columns[index] if index < len(state_columns) else f"state_{index}"
        formatted_equations.append(f"d({state_name})/dt = {equation}")

    return formatted_equations


"""
Function: _build_fit_summary_payload.
Collects the small set of fit-stage facts that are useful for inspection and
later reporting without turning the fit stage into an evaluation stage.
"""


def _build_fit_summary_payload(
    fitted_model: object,
    train_table: pd.DataFrame,
    fit_index: int,
    source_mode: str,
) -> dict[str, object]:
    equations = _extract_equations(fitted_model)
    coefficient_matrix = fitted_model.coefficients()

    return {
        "fit_index": fit_index,
        "source_mode": source_mode,
        "train_case_count": int(train_table["case_name"].nunique()),
        "train_row_count": int(len(train_table)),
        "nonzero_coefficient_count": int(np.count_nonzero(coefficient_matrix)),
        "coefficient_matrix_shape": list(coefficient_matrix.shape),
        "model_summary": model.get_model_summary(),
        "equations": equations,
    }


"""
Function: _next_fit_index.
Reads the equations history file and returns the next sequential fit number.
The history file is the canonical source for run numbering because it is never
rewritten and therefore preserves the complete fit-stage equation record.
"""


def _next_fit_index(history_path: Path) -> int:
    if not history_path.exists():
        return 1

    content = history_path.read_text(encoding="utf-8")
    matches = re.findall(r"^## Fit (\d+)$", content, flags=re.MULTILINE)
    if not matches:
        return 1

    return max(int(value) for value in matches) + 1


"""
Function: _append_equations_history.
Adds one new `Fit N` section to the append-only equations history file. This is
the only exported fit-stage artifact that intentionally accumulates over time.
"""


def _append_equations_history(
    history_path: Path,
    fit_summary_payload: dict[str, object],
) -> None:
    model_summary = fit_summary_payload["model_summary"]
    equations = fit_summary_payload["equations"]

    section_lines = [
        f"## Fit {fit_summary_payload['fit_index']}",
        f"Source mode: {fit_summary_payload['source_mode']}",
        f"Training cases: {fit_summary_payload['train_case_count']}",
        f"Training rows: {fit_summary_payload['train_row_count']}",
        f"Library: {model_summary['library_type']}",
        f"Polynomial degree: {model_summary['polynomial_degree']}",
        f"Optimizer: {model_summary['optimizer_type']}",
        f"Threshold: {model_summary['threshold']}",
        f"Alpha: {model_summary['alpha']}",
        f"Non-zero coefficients: {fit_summary_payload['nonzero_coefficient_count']}",
        "Equations:",
    ]
    section_lines.extend(equations)
    section_text = "\n".join(section_lines) + "\n"

    if history_path.exists():
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(section_text)
    else:
        history_path.write_text(section_text, encoding="utf-8")


"""
Section 5: Saved artifact helpers.
These helpers create the output directories and save both the canonical latest
model artifact and the archived per-fit copy required for history tracking.
"""


"""
Function: _ensure_fit_output_directories.
Creates the directories needed for model pickles and fit-stage text outputs.
"""


def _ensure_fit_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


"""
Function: _save_model_artifacts.
Writes the fitted model twice: once as the overwriteable canonical latest model
and once as the archived snapshot for this specific fit run.
"""


def _save_model_artifacts(
    fitted_model: object,
    fit_index: int,
) -> tuple[Path, Path]:
    _ensure_fit_output_directories()

    latest_model_path = OUTPUT_DIR / LATEST_MODEL_FILENAME
    archived_model_path = MODEL_ARCHIVE_DIR / f"fit_{fit_index:03d}_model.pkl"

    with latest_model_path.open("wb") as handle:
        pickle.dump(fitted_model, handle)

    with archived_model_path.open("wb") as handle:
        pickle.dump(fitted_model, handle)

    return latest_model_path, archived_model_path


"""
Function: _save_fit_summary.
Writes the current fit summary to the overwriteable latest summary file.
"""


def _save_fit_summary(fit_summary_payload: dict[str, object]) -> Path:
    _ensure_fit_output_directories()

    summary_path = OUTPUT_DIR / LATEST_FIT_SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(fit_summary_payload, indent=2),
        encoding="utf-8",
    )
    return summary_path


"""
Section 6: Public entry point for the fit stage.
This final section coordinates the whole training stage: obtain valid prepared
data, fit the canonical model, save the artifacts, and append the equations
history entry for this run.
"""


"""
Function: run_fit_stage.
Runs the first V2 training stage. The function is intentionally focused on
training only: it does not perform holdout validation, confirmation prediction,
or tuning. Those responsibilities belong to later stages.
"""


def run_fit_stage() -> FitStageOutput:
    prepared_data = _obtain_prepared_training_data()
    fitted_model = _fit_model(prepared_data.train_table)

    equations_history_path = OUTPUT_DIR / EQUATIONS_HISTORY_FILENAME
    fit_index = _next_fit_index(equations_history_path)
    fit_summary_payload = _build_fit_summary_payload(
        fitted_model=fitted_model,
        train_table=prepared_data.train_table,
        fit_index=fit_index,
        source_mode=prepared_data.source_mode,
    )

    latest_model_path, archived_model_path = _save_model_artifacts(
        fitted_model=fitted_model,
        fit_index=fit_index,
    )
    summary_path = _save_fit_summary(fit_summary_payload)
    _append_equations_history(equations_history_path, fit_summary_payload)

    print(f"[fit] completed fit run {fit_index}", flush=True)
    print(
        f"[fit] training cases: {fit_summary_payload['train_case_count']}",
        flush=True,
    )
    print(
        f"[fit] training rows: {fit_summary_payload['train_row_count']}",
        flush=True,
    )
    print(f"[fit] latest model saved to {latest_model_path}", flush=True)

    return FitStageOutput(
        fitted_model=fitted_model,
        fit_index=fit_index,
        train_case_count=int(fit_summary_payload["train_case_count"]),
        train_row_count=int(fit_summary_payload["train_row_count"]),
        source_mode=prepared_data.source_mode,
        latest_model_path=latest_model_path,
        archived_model_path=archived_model_path,
        summary_path=summary_path,
        equations_history_path=equations_history_path,
    )


if __name__ == "__main__":
    run_fit_stage()

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
This fit stage trains two separate one-output PySINDy models and stores them as
one bundled artifact for later loading by the run stage.
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
class PreparedTrainingData:
    train_table: pd.DataFrame
    validation_table: pd.DataFrame
    source_mode: str


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
These helpers match the V2 behavior and make sure the split-model fit stage
still reads only valid prepared training outputs.
"""


def _load_existing_outputs() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    train_path = read.OUTPUT_DIR / read.OUTPUT_FILENAMES["train"]
    validation_path = read.OUTPUT_DIR / read.OUTPUT_FILENAMES["validation"]

    if not train_path.exists() or not validation_path.exists():
        return None

    return pd.read_csv(train_path), pd.read_csv(validation_path)


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

    if train_table["case_name"].nunique() == 0:
        return False, "no training cases were found in the prepared training table"

    if not train_table["dataset_role"].eq("train").all():
        return False, "prepared training table contains non-training rows"

    train_is_confirmation = train_table["is_confirmation"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if train_is_confirmation.fillna(True).any():
        return False, "prepared training table contains confirmation rows"

    validation_is_valid = validation_table["is_valid"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if not validation_is_valid.fillna(False).all():
        return False, "validation table reports invalid source files"

    training_rows = validation_table.loc[validation_table["dataset_role"] == "train"]
    if training_rows.empty:
        return False, "validation table contains no training cases"

    return True, "prepared outputs passed reuse checks"


def _obtain_prepared_training_data() -> PreparedTrainingData:
    loaded_outputs = _load_existing_outputs()
    if loaded_outputs is not None:
        train_table, validation_table = loaded_outputs
        outputs_are_valid, message = _validate_reusable_outputs(train_table, validation_table)
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
Section 3: Per-target trajectory rebuilding and model fitting.
These helpers split the combined training table into one-target trajectories and
fit one separate PySINDy model for `cl` and one for `cd`.
"""


def _build_training_trajectories(
    train_table: pd.DataFrame,
    target_column: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    control_columns = model.get_control_columns()
    time_column = model.get_time_column()

    x_train: list[np.ndarray] = []
    u_train: list[np.ndarray] = []
    t_train: list[np.ndarray] = []
    case_names: list[str] = []

    for case_name, case_table in train_table.groupby("case_name", sort=True):
        ordered_case_table = case_table.sort_values(by=time_column).reset_index(drop=True)
        x_train.append(ordered_case_table[[target_column]].to_numpy())
        u_train.append(ordered_case_table[control_columns].to_numpy())
        t_train.append(ordered_case_table[time_column].to_numpy())
        case_names.append(case_name)

    return x_train, u_train, t_train, case_names


def _fit_model_for_target(
    train_table: pd.DataFrame,
    target_column: str,
    settings: dict[str, object] | None = None,
):
    fitted_model = model.build_pysindy_model(
        target_column=target_column,
        settings=settings,
    )
    x_train, u_train, t_train, _ = _build_training_trajectories(train_table, target_column)
    feature_names = [target_column] + model.get_control_columns()

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


def _fit_models(
    train_table: pd.DataFrame,
    settings: dict[str, object] | None = None,
) -> dict[str, object]:
    fitted_models: dict[str, object] = {}
    for target_column in model.get_target_columns():
        print(f"[fit] fitting target model for {target_column}", flush=True)
        fitted_models[target_column] = _fit_model_for_target(
            train_table=train_table,
            target_column=target_column,
            settings=settings,
        )
    return fitted_models


"""
Section 4: Summary and history helpers.
These helpers export the two-model fit as one readable summary and one append-
only equations history entry.
"""


def _extract_equations(
    fitted_model: object,
    target_column: str,
) -> list[str]:
    equations = fitted_model.equations()
    formatted_equations = []
    for equation in equations:
        formatted_equations.append(f"d({target_column})/dt = {equation}")
    return formatted_equations


def _build_fit_summary_payload(
    fitted_models: dict[str, object],
    train_table: pd.DataFrame,
    fit_index: int,
    source_mode: str,
) -> dict[str, object]:
    target_summaries: dict[str, object] = {}

    for target_column, fitted_model in fitted_models.items():
        coefficient_matrix = fitted_model.coefficients()
        target_summaries[target_column] = {
            "nonzero_coefficient_count": int(np.count_nonzero(coefficient_matrix)),
            "coefficient_matrix_shape": list(coefficient_matrix.shape),
            "model_summary": model.get_model_summary(target_column=target_column),
            "equations": _extract_equations(fitted_model, target_column),
        }

    return {
        "fit_index": fit_index,
        "source_mode": source_mode,
        "train_case_count": int(train_table["case_name"].nunique()),
        "train_row_count": int(len(train_table)),
        "target_models": target_summaries,
    }


def _next_fit_index(history_path: Path) -> int:
    if not history_path.exists():
        return 1
    content = history_path.read_text(encoding="utf-8")
    matches = re.findall(r"^## Fit (\d+)$", content, flags=re.MULTILINE)
    if not matches:
        return 1
    return max(int(value) for value in matches) + 1


def _append_equations_history(
    history_path: Path,
    fit_summary_payload: dict[str, object],
) -> None:
    fit_index = int(fit_summary_payload["fit_index"])
    section_lines = [
        f"## Fit {fit_index}",
        f"Training cases: {fit_summary_payload['train_case_count']}",
        f"Training rows: {fit_summary_payload['train_row_count']}",
    ]

    for target_column in model.get_target_columns():
        target_summary = fit_summary_payload["target_models"][target_column]
        section_lines.append(f"{target_column.upper()} model:")
        section_lines.append(
            f"Nonzero coefficients: {target_summary['nonzero_coefficient_count']}"
        )
        section_lines.extend(target_summary["equations"])

    section_text = "\n".join(section_lines) + "\n"
    if history_path.exists():
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(section_text)
    else:
        history_path.write_text(section_text, encoding="utf-8")


"""
Section 5: Output saving.
These helpers write the bundled model artifact and the standard fit summary
files used by the rest of the V3 pipeline.
"""


def _ensure_fit_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _save_model_outputs(
    fitted_models: dict[str, object],
    fit_summary_payload: dict[str, object],
) -> tuple[Path, Path, Path, Path]:
    _ensure_fit_output_directories()

    fit_index = int(fit_summary_payload["fit_index"])
    latest_model_path = OUTPUT_DIR / LATEST_MODEL_FILENAME
    archived_model_path = MODEL_ARCHIVE_DIR / f"fit_{fit_index:03d}_model.pkl"
    summary_path = OUTPUT_DIR / LATEST_FIT_SUMMARY_FILENAME
    equations_history_path = OUTPUT_DIR / EQUATIONS_HISTORY_FILENAME

    with latest_model_path.open("wb") as handle:
        pickle.dump(fitted_models, handle)
    with archived_model_path.open("wb") as handle:
        pickle.dump(fitted_models, handle)

    summary_path.write_text(json.dumps(fit_summary_payload, indent=2), encoding="utf-8")
    _append_equations_history(equations_history_path, fit_summary_payload)

    return latest_model_path, archived_model_path, summary_path, equations_history_path


"""
Section 6: Public fit entry point.
This stage coordinates the full V3 split-model fit workflow and exports the
bundled `cl` and `cd` models for the run stage.
"""


def run_fit_stage() -> FitStageOutput:
    prepared_training_data = _obtain_prepared_training_data()
    fit_index = _next_fit_index(OUTPUT_DIR / EQUATIONS_HISTORY_FILENAME)

    fitted_models = _fit_models(prepared_training_data.train_table)
    fit_summary_payload = _build_fit_summary_payload(
        fitted_models=fitted_models,
        train_table=prepared_training_data.train_table,
        fit_index=fit_index,
        source_mode=prepared_training_data.source_mode,
    )

    (
        latest_model_path,
        archived_model_path,
        summary_path,
        equations_history_path,
    ) = _save_model_outputs(
        fitted_models=fitted_models,
        fit_summary_payload=fit_summary_payload,
    )

    print(f"[fit] completed fit run {fit_index}", flush=True)
    print(
        f"[fit] training cases: {prepared_training_data.train_table['case_name'].nunique()}",
        flush=True,
    )
    print(f"[fit] training rows: {len(prepared_training_data.train_table)}", flush=True)
    print(f"[fit] latest model saved to {latest_model_path}", flush=True)

    return FitStageOutput(
        fitted_model=fitted_models,
        fit_index=fit_index,
        train_case_count=int(prepared_training_data.train_table["case_name"].nunique()),
        train_row_count=int(len(prepared_training_data.train_table)),
        source_mode=prepared_training_data.source_mode,
        latest_model_path=latest_model_path,
        archived_model_path=archived_model_path,
        summary_path=summary_path,
        equations_history_path=equations_history_path,
    )


if __name__ == "__main__":
    run_fit_stage()

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd

try:
    from .. import fit, model, read, run as run_stage
except ImportError:  # pragma: no cover
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from assets import fit, model, read, run as run_stage


"""
Section 1: Search ranges, validation-case policy, output files, and returned
containers.
This section keeps the brute-force search space fully visible at the top of the
file, matching the explicit coursework style. It also fixes the reduced
leave-one-case-out validation set requested for this V2 tuning increment.
"""

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

THRESHOLD_VALUES = [0.01, 0.05, 0.1, 0.2]
ALPHA_VALUES = [0.0, 0.01, 0.05, 0.1]
POLYNOMIAL_DEGREE_VALUES = [2, 3, 4]

VALIDATION_CASE_NAMES = ("B1", "C1", "C4", "C5", "D1", "D4", "D5")
AMPLITUDE_PENALTY_WEIGHT = 0.5

RESULTS_FILENAME = "hyperparameter_results_latest.csv"
BEST_SUMMARY_FILENAME = "hyperparameter_best_summary_latest.json"
EQUATIONS_HISTORY_FILENAME = "hyperparameter_equations_history.txt"
CONFIRMATION_SUMMARY_FILENAME = "hyperparameter_confirmation_summary_latest.csv"
CONFIRMATION_PREDICTIONS_FILENAME = "hyperparameter_confirmation_predictions_latest.csv"


"""
Container: HyperparameterSearchOutput.
Bundles the main tuning outputs so the caller can see which files were written
and which parameter set won the brute-force search.
"""


@dataclass(frozen=True)
class HyperparameterSearchOutput:
    tuning_run_index: int
    combination_count: int
    best_parameters: dict[str, float | int]
    best_score: float
    results_path: Path
    best_summary_path: Path
    equations_history_path: Path
    confirmation_summary_path: Path | None
    confirmation_predictions_path: Path | None
    applied_to_model: bool


"""
Section 2: Shared helpers for fitting, scoring, and output numbering.
These helpers support the brute-force loop without hiding the actual search
logic. The visible nested loops later in the file remain the main workflow.
"""


"""
Function: _next_tuning_run_index.
Reads the append-only tuning equations history and returns the next run number.
"""


def _next_tuning_run_index(history_path: Path) -> int:
    if not history_path.exists():
        return 1

    content = history_path.read_text(encoding="utf-8")
    matches = re.findall(r"^## Tuning Run (\d+)$", content, flags=re.MULTILINE)
    if not matches:
        return 1

    return max(int(value) for value in matches) + 1


"""
Function: _relative_difference.
Returns the relative difference between two positive values. The helper keeps
the amplitude-penalty terms numerically stable when one value is very small.
"""


def _relative_difference(reference_value: float, candidate_value: float) -> float:
    scale = max(abs(reference_value), 1e-12)
    return abs(candidate_value - reference_value) / scale


"""
Function: _validation_metrics.
Builds the case-level metrics used to rank candidate parameter combinations.
The score combines RMSE with an amplitude-preservation penalty so heavily
damped trajectories no longer look artificially strong.
"""


def _validation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cl_residual = y_true[:, 0] - y_pred[:, 0]
    cd_residual = y_true[:, 1] - y_pred[:, 1]

    rmse_cl = float(np.sqrt(np.mean(np.square(cl_residual))))
    rmse_cd = float(np.sqrt(np.mean(np.square(cd_residual))))
    mae_cl = float(np.mean(np.abs(cl_residual)))
    mae_cd = float(np.mean(np.abs(cd_residual)))

    cl_true_range = float(np.max(y_true[:, 0]) - np.min(y_true[:, 0]))
    cl_pred_range = float(np.max(y_pred[:, 0]) - np.min(y_pred[:, 0]))
    cd_true_range = float(np.max(y_true[:, 1]) - np.min(y_true[:, 1]))
    cd_pred_range = float(np.max(y_pred[:, 1]) - np.min(y_pred[:, 1]))

    cl_true_std = float(np.std(y_true[:, 0]))
    cl_pred_std = float(np.std(y_pred[:, 0]))
    cd_true_std = float(np.std(y_true[:, 1]))
    cd_pred_std = float(np.std(y_pred[:, 1]))

    cl_range_penalty = _relative_difference(cl_true_range, cl_pred_range)
    cd_range_penalty = _relative_difference(cd_true_range, cd_pred_range)
    cl_std_penalty = _relative_difference(cl_true_std, cl_pred_std)
    cd_std_penalty = _relative_difference(cd_true_std, cd_pred_std)
    amplitude_penalty = float(
        np.mean([cl_range_penalty, cd_range_penalty, cl_std_penalty, cd_std_penalty])
    )

    rmse_score = float(np.mean([rmse_cl, rmse_cd]))
    score = float(rmse_score + AMPLITUDE_PENALTY_WEIGHT * amplitude_penalty)

    return {
        "rmse_cl": rmse_cl,
        "rmse_cd": rmse_cd,
        "mae_cl": mae_cl,
        "mae_cd": mae_cd,
        "cl_true_range": cl_true_range,
        "cl_pred_range": cl_pred_range,
        "cd_true_range": cd_true_range,
        "cd_pred_range": cd_pred_range,
        "cl_true_std": cl_true_std,
        "cl_pred_std": cl_pred_std,
        "cd_true_std": cd_true_std,
        "cd_pred_std": cd_pred_std,
        "amplitude_penalty": amplitude_penalty,
        "score": score,
    }


"""
Function: _fit_candidate_model.
Fits one PySINDy model definition for one candidate setting dictionary using the
same multi-trajectory conventions as the normal fit stage.
"""


def _fit_candidate_model(
    train_subset: pd.DataFrame,
    candidate_settings: dict[str, float | int],
):
    fitted_model = model.build_pysindy_model(settings=candidate_settings)
    x_train, u_train, t_train, _ = fit._build_training_trajectories(train_subset)
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
Function: _available_validation_case_names.
Intersects the requested reduced validation set with the cases actually present
in the prepared training data.
"""


def _available_validation_case_names(train_table: pd.DataFrame) -> list[str]:
    available_case_names = set(train_table["case_name"].astype(str).unique())
    selected_case_names = [
        case_name for case_name in VALIDATION_CASE_NAMES if case_name in available_case_names
    ]
    if not selected_case_names:
        raise ValueError(
            "None of the requested validation cases were found in the prepared training data."
        )
    return selected_case_names


"""
Function: _prompt_yes_no.
Prompts for the optional apply-to-model step after the tuning run finishes.
"""


def _prompt_yes_no(message: str) -> bool:
    while True:
        answer = input(f"{message} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("[tuning] please answer with 'y' or 'n'", flush=True)


"""
Function: _apply_best_settings_to_model_file.
Overwrites the live defaults in `model.py` with the best tuned values.
"""


def _apply_best_settings_to_model_file(best_row: dict[str, object]) -> None:
    model_path = Path(model.__file__).resolve()
    content = model_path.read_text(encoding="utf-8")

    replacements = {
        r"^DEFAULT_POLYNOMIAL_DEGREE = .+$": (
            f"DEFAULT_POLYNOMIAL_DEGREE = {int(best_row['polynomial_degree'])}"
        ),
        r"^DEFAULT_THRESHOLD = .+$": f"DEFAULT_THRESHOLD = {float(best_row['threshold'])}",
        r"^DEFAULT_ALPHA = .+$": f"DEFAULT_ALPHA = {float(best_row['alpha'])}",
    }

    updated_content = content
    for pattern, replacement in replacements.items():
        updated_content, replacement_count = re.subn(
            pattern,
            replacement,
            updated_content,
            count=1,
            flags=re.MULTILINE,
        )
        if replacement_count != 1:
            raise ValueError(f"Could not update model.py using pattern: {pattern}")

    model_path.write_text(updated_content, encoding="utf-8")


"""
Function: _append_equation_history.
Appends the equations of the final best model to the tuning equations history.
"""


def _append_equation_history(
    history_path: Path,
    tuning_run_index: int,
    best_row: dict[str, object],
    best_model: object,
) -> None:
    model_summary = model.get_model_summary(settings=best_row)
    equations = fit._extract_equations(best_model)

    section_lines = [
        f"## Tuning Run {tuning_run_index}",
        f"Threshold: {best_row['threshold']}",
        f"Alpha: {best_row['alpha']}",
        f"Polynomial degree: {best_row['polynomial_degree']}",
        f"Mean score: {best_row['mean_score']}",
        f"Mean RMSE cl: {best_row['mean_rmse_cl']}",
        f"Mean RMSE cd: {best_row['mean_rmse_cd']}",
        f"Mean MAE cl: {best_row['mean_mae_cl']}",
        f"Mean MAE cd: {best_row['mean_mae_cd']}",
        f"Mean amplitude penalty: {best_row['mean_amplitude_penalty']}",
        f"Validation cases: {best_row['validation_case_count']}",
        f"Library: {model_summary['library_type']}",
        f"Optimizer: {model_summary['optimizer_type']}",
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
Section 3: Confirmation-case export helpers.
These helpers write the one final confirmation test requested after the best
hyperparameters have been selected.
"""


"""
Function: _prediction_rows_for_case.
Builds one row-level prediction table for the final confirmation pass.
"""


def _prediction_rows_for_case(
    case_table: pd.DataFrame,
    prediction: np.ndarray,
    tuning_run_index: int,
    candidate_settings: dict[str, object],
) -> pd.DataFrame:
    ordered_case_table = case_table.sort_values(by=model.get_time_column()).reset_index(drop=True)
    output_table = ordered_case_table.copy()
    output_table.insert(0, "tuning_run_index", tuning_run_index)
    output_table.insert(1, "threshold", float(candidate_settings["threshold"]))
    output_table.insert(2, "alpha", float(candidate_settings["alpha"]))
    output_table.insert(3, "polynomial_degree", int(candidate_settings["polynomial_degree"]))
    output_table["cl_pred"] = prediction[:, 0]
    output_table["cd_pred"] = prediction[:, 1]
    output_table["cl_residual"] = output_table["cl"].to_numpy() - prediction[:, 0]
    output_table["cd_residual"] = output_table["cd"].to_numpy() - prediction[:, 1]
    return output_table


"""
Function: _summary_row_for_case.
Builds one per-case summary row for the final confirmation pass.
"""


def _summary_row_for_case(
    case_table: pd.DataFrame,
    prediction: np.ndarray,
    tuning_run_index: int,
    candidate_settings: dict[str, object],
) -> dict[str, object]:
    ordered_case_table = case_table.sort_values(by=model.get_time_column()).reset_index(drop=True)
    truth = ordered_case_table[model.get_state_columns()].to_numpy()
    metrics = _validation_metrics(truth, prediction)

    return {
        "tuning_run_index": tuning_run_index,
        "threshold": float(candidate_settings["threshold"]),
        "alpha": float(candidate_settings["alpha"]),
        "polynomial_degree": int(candidate_settings["polynomial_degree"]),
        "source_file": ordered_case_table["source_file"].iloc[0],
        "case_name": ordered_case_table["case_name"].iloc[0],
        "case_group": ordered_case_table["case_group"].iloc[0],
        "dataset_role": ordered_case_table["dataset_role"].iloc[0],
        "is_confirmation": ordered_case_table["is_confirmation"].iloc[0],
        "rows": int(len(ordered_case_table)),
        "rmse_cl": metrics["rmse_cl"],
        "rmse_cd": metrics["rmse_cd"],
        "mae_cl": metrics["mae_cl"],
        "mae_cd": metrics["mae_cd"],
    }


"""
Function: _run_final_confirmation_test.
Fits the best candidate on all available training data and then runs one final
confirmation evaluation on the current `T#` cases if any are present.
"""


def _run_final_confirmation_test(
    best_model: object,
    candidate_settings: dict[str, object],
    tuning_run_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fresh_read_output = read.run_read_stage()
    confirmation_table = fresh_read_output.confirmation_table.copy()

    if confirmation_table.empty:
        print("[tuning] no confirmation cases are currently available for the final test", flush=True)
        return pd.DataFrame(), pd.DataFrame()

    prediction_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for case_name, case_table in confirmation_table.groupby("case_name", sort=True):
        print(f"[tuning] final confirmation test on {case_name}", flush=True)
        prediction = run_stage._simulate_case(
            fitted_model=best_model,
            case_table=case_table,
        )
        prediction_tables.append(
            _prediction_rows_for_case(
                case_table=case_table,
                prediction=prediction,
                tuning_run_index=tuning_run_index,
                candidate_settings=candidate_settings,
            )
        )
        summary_rows.append(
            _summary_row_for_case(
                case_table=case_table,
                prediction=prediction,
                tuning_run_index=tuning_run_index,
                candidate_settings=candidate_settings,
            )
        )

    prediction_table = pd.concat(prediction_tables, ignore_index=True)
    summary_table = pd.DataFrame(summary_rows).sort_values(by="case_name").reset_index(drop=True)
    return prediction_table, summary_table


"""
Section 4: Output writers.
These helpers write the latest tuning outputs after the brute-force search and
the final confirmation pass have completed.
"""


"""
Function: _write_outputs.
Writes the latest tuning tables and summary files and returns their paths.
"""


def _write_outputs(
    results_table: pd.DataFrame,
    best_summary_payload: dict[str, object],
    confirmation_summary_table: pd.DataFrame,
    confirmation_prediction_table: pd.DataFrame,
) -> tuple[Path, Path, Path | None, Path | None]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / RESULTS_FILENAME
    best_summary_path = OUTPUT_DIR / BEST_SUMMARY_FILENAME
    confirmation_summary_path: Path | None = None
    confirmation_predictions_path: Path | None = None

    results_table.to_csv(results_path, index=False)
    best_summary_path.write_text(
        json.dumps(best_summary_payload, indent=2),
        encoding="utf-8",
    )

    if not confirmation_summary_table.empty:
        confirmation_summary_path = OUTPUT_DIR / CONFIRMATION_SUMMARY_FILENAME
        confirmation_summary_table.to_csv(confirmation_summary_path, index=False)

    if not confirmation_prediction_table.empty:
        confirmation_predictions_path = OUTPUT_DIR / CONFIRMATION_PREDICTIONS_FILENAME
        confirmation_prediction_table.to_csv(confirmation_predictions_path, index=False)

    return (
        results_path,
        best_summary_path,
        confirmation_summary_path,
        confirmation_predictions_path,
    )


"""
Section 5: Public entry point and brute-force search loop.
This final section keeps the search visible in one place. The candidate ranges,
the nested loops, the fold-by-fold validation, and the final best-candidate
confirmation test are all expressed explicitly here.
"""


"""
Function: run_hyperparameter_search.
Runs the explicit brute-force V2 tuning workflow using the selected validation
cases only, then performs one final confirmation test on `T#` if available.
"""


def run_hyperparameter_search() -> HyperparameterSearchOutput:
    prepared_training_data = fit._obtain_prepared_training_data()
    train_table = prepared_training_data.train_table.copy()
    validation_case_names = _available_validation_case_names(train_table)

    print(
        "[tuning] validation cases: "
        + ", ".join(validation_case_names),
        flush=True,
    )

    total_combinations = (
        len(THRESHOLD_VALUES)
        * len(ALPHA_VALUES)
        * len(POLYNOMIAL_DEGREE_VALUES)
    )

    results_rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_score = float("inf")
    combination_counter = 0

    for threshold in THRESHOLD_VALUES:
        for alpha in ALPHA_VALUES:
            for polynomial_degree in POLYNOMIAL_DEGREE_VALUES:
                combination_counter += 1
                candidate_settings = {
                    "threshold": threshold,
                    "alpha": alpha,
                    "polynomial_degree": polynomial_degree,
                }

                print(
                    f"Trying PySINDy: threshold={threshold}, "
                    f"alpha={alpha}, polynomial_degree={polynomial_degree}",
                    flush=True,
                )
                print(
                    f"[tuning] combination {combination_counter}/{total_combinations}",
                    flush=True,
                )

                fold_metrics: list[dict[str, float]] = []
                candidate_failed = False
                failed_holdout_case = ""
                failure_message = ""

                for fold_index, holdout_name in enumerate(validation_case_names, start=1):
                    print(
                        f"[tuning] testing holdout {fold_index}/{len(validation_case_names)}: {holdout_name}",
                        flush=True,
                    )

                    train_subset = train_table.loc[
                        train_table["case_name"].astype(str) != holdout_name
                    ].copy()
                    holdout_case = train_table.loc[
                        train_table["case_name"].astype(str) == holdout_name
                    ].copy()

                    try:
                        fitted_model = _fit_candidate_model(
                            train_subset=train_subset,
                            candidate_settings=candidate_settings,
                        )
                        prediction = run_stage._simulate_case(
                            fitted_model=fitted_model,
                            case_table=holdout_case,
                        )
                    except Exception as exc:
                        candidate_failed = True
                        failed_holdout_case = holdout_name
                        failure_message = str(exc)
                        print(
                            f"[tuning] holdout {holdout_name} failed: {failure_message}",
                            flush=True,
                        )
                        break

                    truth = (
                        holdout_case.sort_values(by=model.get_time_column())
                        .reset_index(drop=True)[model.get_state_columns()]
                        .to_numpy()
                    )
                    metrics = _validation_metrics(truth, prediction)
                    fold_metrics.append(metrics)

                    print(
                        f"[tuning] result for {holdout_name}: "
                        f"rmse_cl={metrics['rmse_cl']:.6f}, "
                        f"rmse_cd={metrics['rmse_cd']:.6f}, "
                        f"amplitude_penalty={metrics['amplitude_penalty']:.6f}, "
                        f"score={metrics['score']:.6f}",
                        flush=True,
                    )

                if candidate_failed:
                    candidate_row = {
                        "threshold": threshold,
                        "alpha": alpha,
                        "polynomial_degree": polynomial_degree,
                        "mean_rmse_cl": float("inf"),
                        "mean_rmse_cd": float("inf"),
                        "mean_mae_cl": float("inf"),
                        "mean_mae_cd": float("inf"),
                        "mean_amplitude_penalty": float("inf"),
                        "mean_score": float("inf"),
                        "validation_case_count": len(fold_metrics),
                        "candidate_status": "failed",
                        "failed_holdout_case": failed_holdout_case,
                        "failure_message": failure_message,
                    }
                else:
                    candidate_row = {
                        "threshold": threshold,
                        "alpha": alpha,
                        "polynomial_degree": polynomial_degree,
                        "mean_rmse_cl": float(np.mean([row["rmse_cl"] for row in fold_metrics])),
                        "mean_rmse_cd": float(np.mean([row["rmse_cd"] for row in fold_metrics])),
                        "mean_mae_cl": float(np.mean([row["mae_cl"] for row in fold_metrics])),
                        "mean_mae_cd": float(np.mean([row["mae_cd"] for row in fold_metrics])),
                        "mean_amplitude_penalty": float(
                            np.mean([row["amplitude_penalty"] for row in fold_metrics])
                        ),
                        "mean_score": float(np.mean([row["score"] for row in fold_metrics])),
                        "validation_case_count": len(validation_case_names),
                        "candidate_status": "ok",
                        "failed_holdout_case": "",
                        "failure_message": "",
                    }

                    if candidate_row["mean_score"] < best_score:
                        best_score = float(candidate_row["mean_score"])
                        best_row = candidate_row

                results_rows.append(candidate_row)

    if best_row is None:
        raise RuntimeError(
            "All tested hyperparameter combinations failed during validation."
        )

    results_table = (
        pd.DataFrame(results_rows)
        .sort_values(
            by=["mean_score", "mean_rmse_cl", "mean_rmse_cd"],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    print(
        "[tuning] best combination found: "
        f"threshold={best_row['threshold']}, "
        f"alpha={best_row['alpha']}, "
        f"degree={best_row['polynomial_degree']}",
        flush=True,
    )
    print(
        f"[tuning] best validation score: {best_row['mean_score']:.6f}",
        flush=True,
    )

    best_model = _fit_candidate_model(
        train_subset=train_table,
        candidate_settings=best_row,
    )

    history_path = OUTPUT_DIR / EQUATIONS_HISTORY_FILENAME
    tuning_run_index = _next_tuning_run_index(history_path)

    confirmation_prediction_table, confirmation_summary_table = _run_final_confirmation_test(
        best_model=best_model,
        candidate_settings=best_row,
        tuning_run_index=tuning_run_index,
    )

    applied_to_model = _prompt_yes_no(
        "Apply the best hyperparameters to model.py?"
    )
    if applied_to_model:
        _apply_best_settings_to_model_file(best_row)

    confirmation_summary_records = confirmation_summary_table.to_dict(orient="records")
    best_summary_payload = {
        "tuning_run_index": tuning_run_index,
        "source_mode": prepared_training_data.source_mode,
        "combination_count": combination_counter,
        "validation_case_names": validation_case_names,
        "search_grid": {
            "threshold": THRESHOLD_VALUES,
            "alpha": ALPHA_VALUES,
            "polynomial_degree": POLYNOMIAL_DEGREE_VALUES,
        },
        "score_metric": "mean_rmse_plus_amplitude_penalty",
        "validation_strategy": "leave_one_selected_training_case_out",
        "best_parameters": {
            "threshold": float(best_row["threshold"]),
            "alpha": float(best_row["alpha"]),
            "polynomial_degree": int(best_row["polynomial_degree"]),
        },
        "best_metrics": {
            "mean_score": float(best_row["mean_score"]),
            "mean_rmse_cl": float(best_row["mean_rmse_cl"]),
            "mean_rmse_cd": float(best_row["mean_rmse_cd"]),
            "mean_mae_cl": float(best_row["mean_mae_cl"]),
            "mean_mae_cd": float(best_row["mean_mae_cd"]),
            "mean_amplitude_penalty": float(best_row["mean_amplitude_penalty"]),
        },
        "applied_to_model": applied_to_model,
        "final_confirmation_available": not confirmation_summary_table.empty,
        "final_confirmation_case_count": int(len(confirmation_summary_table)),
        "final_confirmation_summary": confirmation_summary_records,
    }

    (
        results_path,
        best_summary_path,
        confirmation_summary_path,
        confirmation_predictions_path,
    ) = _write_outputs(
        results_table=results_table,
        best_summary_payload=best_summary_payload,
        confirmation_summary_table=confirmation_summary_table,
        confirmation_prediction_table=confirmation_prediction_table,
    )
    _append_equation_history(
        history_path=history_path,
        tuning_run_index=tuning_run_index,
        best_row=best_row,
        best_model=best_model,
    )

    return HyperparameterSearchOutput(
        tuning_run_index=tuning_run_index,
        combination_count=combination_counter,
        best_parameters=best_summary_payload["best_parameters"],
        best_score=float(best_row["mean_score"]),
        results_path=results_path,
        best_summary_path=best_summary_path,
        equations_history_path=history_path,
        confirmation_summary_path=confirmation_summary_path,
        confirmation_predictions_path=confirmation_predictions_path,
        applied_to_model=applied_to_model,
    )


if __name__ == "__main__":
    run_hyperparameter_search()

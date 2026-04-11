from __future__ import annotations

from dataclasses import dataclass
import inspect
from itertools import product
from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd

try:
    from .. import fit, model, run as run_stage
except ImportError:  # pragma: no cover
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from assets import fit, model, run as run_stage


"""
Section 1: Search grid, output paths, and returned containers.
This section defines the first tuning search space, the files written by the
utility, and the small containers returned to the caller.
"""

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

SEARCH_GRID = {
    "threshold": [0.01, 0.05, 0.1, 0.2],
    "alpha": [0.0, 0.01, 0.05, 0.1],
    "polynomial_degree": [2, 3, 4],
}

RESULTS_FILENAME = "hyperparameter_results_latest.csv"
BEST_SUMMARY_FILENAME = "hyperparameter_best_summary_latest.json"
EQUATIONS_HISTORY_FILENAME = "hyperparameter_equations_history.txt"


"""
Container: HyperparameterSearchOutput.
Bundles the main search outputs so the script can report what happened after
the tuning loop finishes.
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
    applied_to_model: bool


"""
Section 2: Search-space helpers and run numbering.
These helpers enumerate the candidate settings and keep a sequential history of
the best-equation snapshots produced by each tuning run.
"""


"""
Function: _parameter_combinations.
Returns every candidate parameter combination from the declared search grid.
The first tuning version uses a simple grid search rather than adaptive search.
"""


def _parameter_combinations() -> list[dict[str, float | int]]:
    keys = list(SEARCH_GRID.keys())
    return [
        dict(zip(keys, values))
        for values in product(*(SEARCH_GRID[key] for key in keys))
    ]


"""
Function: _next_tuning_run_index.
Reads the tuning equations-history file and returns the next tuning run number.
This keeps the best-equation history append-only across multiple searches.
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
Function: _format_progress_message.
Builds a compact textual progress bar that can be reused for both the outer
candidate loop and the inner validation-fold loop.
"""


def _format_progress_message(
    stage_name: str,
    completed_items: int,
    total_items: int,
    status_text: str,
    bar_width: int = 24,
) -> str:
    filled_width = int(bar_width * completed_items / total_items) if total_items else 0
    progress_bar = "#" * filled_width + "-" * (bar_width - filled_width)
    percentage = (100.0 * completed_items / total_items) if total_items else 0.0
    remaining_items = max(total_items - completed_items, 0)

    return (
        f"[tuning] {stage_name} "
        f"[{progress_bar}] "
        f"{completed_items}/{total_items} "
        f"({percentage:5.1f}%) "
        f"{status_text}; "
        f"{remaining_items} remaining"
    )


"""
Section 3: Validation-case scoring helpers.
These helpers define how one candidate is trained, simulated on a held-out
training case, and scored before the aggregate ranking is computed.
"""


"""
Function: _validation_metrics.
Builds the minimal case-level metrics used to rank candidate settings. The
search optimizes the mean of `rmse_cl` and `rmse_cd`.
"""


def _validation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cl_residual = y_true[:, 0] - y_pred[:, 0]
    cd_residual = y_true[:, 1] - y_pred[:, 1]

    rmse_cl = float(np.sqrt(np.mean(np.square(cl_residual))))
    rmse_cd = float(np.sqrt(np.mean(np.square(cd_residual))))
    mae_cl = float(np.mean(np.abs(cl_residual)))
    mae_cd = float(np.mean(np.abs(cd_residual)))
    score = float(np.mean([rmse_cl, rmse_cd]))

    return {
        "rmse_cl": rmse_cl,
        "rmse_cd": rmse_cd,
        "mae_cl": mae_cl,
        "mae_cd": mae_cd,
        "score": score,
    }


"""
Function: _fit_candidate_model.
Builds a PySINDy model with candidate settings and fits it on the supplied
training subset using the same multi-trajectory conventions as the fit stage.
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
Function: _evaluate_candidate_on_holdout.
Fits a candidate model on all training cases except one and evaluates it on the
held-out case using the same simulation contract as the run stage.
"""


def _evaluate_candidate_on_holdout(
    train_subset: pd.DataFrame,
    holdout_case: pd.DataFrame,
    candidate_settings: dict[str, float | int],
) -> tuple[dict[str, float], object]:
    fitted_model = _fit_candidate_model(
        train_subset=train_subset,
        candidate_settings=candidate_settings,
    )
    prediction = run_stage._simulate_case(
        fitted_model=fitted_model,
        case_table=holdout_case,
    )
    truth = (
        holdout_case.sort_values(by=model.get_time_column())
        .reset_index(drop=True)[model.get_state_columns()]
        .to_numpy()
    )
    return _validation_metrics(truth, prediction), fitted_model


"""
Section 4: Search execution and result aggregation.
These helpers run leave-one-case-out validation for each candidate and aggregate
the case-level scores into one ranked results table.
"""


"""
Function: _evaluate_candidate_settings.
Runs leave-one-case-out validation for one candidate parameter combination and
returns the aggregate scores needed to rank that candidate.
"""


def _evaluate_candidate_settings(
    train_table: pd.DataFrame,
    candidate_settings: dict[str, float | int],
    candidate_index: int,
    total_combinations: int,
) -> tuple[dict[str, object], object]:
    fold_metrics: list[dict[str, float]] = []
    best_fold_model = None
    best_fold_score = float("inf")

    grouped_cases = [
        (case_name, case_table.copy())
        for case_name, case_table in train_table.groupby("case_name", sort=True)
    ]
    total_folds = len(grouped_cases)

    print(
        _format_progress_message(
            stage_name="candidate",
            completed_items=candidate_index,
            total_items=total_combinations,
            status_text=(
                f"threshold={candidate_settings['threshold']}, "
                f"alpha={candidate_settings['alpha']}, "
                f"degree={candidate_settings['polynomial_degree']}"
            ),
        ),
        flush=True,
    )

    for fold_index, (holdout_name, holdout_case) in enumerate(grouped_cases, start=1):
        train_subset = train_table.loc[train_table["case_name"] != holdout_name].copy()
        metrics, fitted_model = _evaluate_candidate_on_holdout(
            train_subset=train_subset,
            holdout_case=holdout_case,
            candidate_settings=candidate_settings,
        )
        metrics["holdout_case"] = holdout_name
        fold_metrics.append(metrics)
        if metrics["score"] < best_fold_score:
            best_fold_score = metrics["score"]
            best_fold_model = fitted_model

        print(
            _format_progress_message(
                stage_name="fold",
                completed_items=fold_index,
                total_items=total_folds,
                status_text=(
                    f"{holdout_name} score={metrics['score']:.6f}"
                ),
            ),
            flush=True,
        )

    aggregate_row: dict[str, object] = {
        "threshold": candidate_settings["threshold"],
        "alpha": candidate_settings["alpha"],
        "polynomial_degree": candidate_settings["polynomial_degree"],
        "mean_rmse_cl": float(np.mean([row["rmse_cl"] for row in fold_metrics])),
        "mean_rmse_cd": float(np.mean([row["rmse_cd"] for row in fold_metrics])),
        "mean_mae_cl": float(np.mean([row["mae_cl"] for row in fold_metrics])),
        "mean_mae_cd": float(np.mean([row["mae_cd"] for row in fold_metrics])),
        "mean_score": float(np.mean([row["score"] for row in fold_metrics])),
        "validation_case_count": total_folds,
    }
    return aggregate_row, best_fold_model


"""
Function: _run_search.
Executes the full grid search and returns the ranked results table together with
the best candidate row and one representative fitted model for its equations.
"""


def _run_search(
    train_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], object]:
    combinations = _parameter_combinations()
    result_rows: list[dict[str, object]] = []
    best_model = None
    best_score = float("inf")

    for candidate_index, candidate_settings in enumerate(combinations, start=1):
        candidate_row, candidate_model = _evaluate_candidate_settings(
            train_table=train_table,
            candidate_settings=candidate_settings,
            candidate_index=candidate_index,
            total_combinations=len(combinations),
        )
        result_rows.append(candidate_row)
        if candidate_row["mean_score"] < best_score:
            best_score = candidate_row["mean_score"]
            best_model = candidate_model

    results_table = pd.DataFrame(result_rows).sort_values(
        by=["mean_score", "mean_rmse_cl", "mean_rmse_cd"]
    ).reset_index(drop=True)
    best_row = results_table.iloc[0].to_dict()
    return results_table, best_row, best_model


"""
Section 5: Output writers and optional auto-apply.
These helpers write the tuning outputs to disk, append the best-equation
history, and optionally overwrite the active defaults in `model.py`.
"""


"""
Function: _append_equation_history.
Appends the best candidate equations for one tuning run to the dedicated tuning
history file used for later methodology and report writing.
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
Function: _prompt_yes_no.
Prompts the user for a simple yes/no choice after the search completes. The
prompt is used for the optional auto-apply flow.
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
Overwrites the live default constants in `model.py` so future fit and run
stages use the tuned values immediately.
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
            raise ValueError(
                f"Could not update model.py using pattern: {pattern}"
            )

    model_path.write_text(updated_content, encoding="utf-8")


"""
Function: _write_outputs.
Writes the overwriteable results and best-summary files for the latest tuning
run and returns their paths.
"""


def _write_outputs(
    results_table: pd.DataFrame,
    best_summary_payload: dict[str, object],
) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / RESULTS_FILENAME
    best_summary_path = OUTPUT_DIR / BEST_SUMMARY_FILENAME

    results_table.to_csv(results_path, index=False)
    best_summary_path.write_text(
        json.dumps(best_summary_payload, indent=2),
        encoding="utf-8",
    )

    return results_path, best_summary_path


"""
Section 6: Public entry point for the tuning utility.
This final section coordinates data loading, search execution, output writing,
equation-history logging, and the optional update of `model.py`.
"""


"""
Function: run_hyperparameter_search.
Runs the first V2 tuning workflow using training cases only. It searches the
declared candidate grid, ranks combinations by mean RMSE score, writes the
results, and optionally overwrites the active model defaults.
"""


def run_hyperparameter_search() -> HyperparameterSearchOutput:
    prepared_training_data = fit._obtain_prepared_training_data()
    train_table = prepared_training_data.train_table.copy()
    results_table, best_row, best_model = _run_search(train_table)

    equations_history_path = OUTPUT_DIR / EQUATIONS_HISTORY_FILENAME
    tuning_run_index = _next_tuning_run_index(equations_history_path)

    applied_to_model = _prompt_yes_no(
        "Apply the best hyperparameters to model.py?"
    )
    if applied_to_model:
        _apply_best_settings_to_model_file(best_row)

    best_summary_payload = {
        "tuning_run_index": tuning_run_index,
        "source_mode": prepared_training_data.source_mode,
        "combination_count": int(len(results_table)),
        "search_grid": SEARCH_GRID,
        "score_metric": "mean_rmse_of_cl_and_cd",
        "validation_strategy": "leave_one_training_case_out",
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
        },
        "validation_case_count": int(best_row["validation_case_count"]),
        "applied_to_model": applied_to_model,
    }

    results_path, best_summary_path = _write_outputs(
        results_table=results_table,
        best_summary_payload=best_summary_payload,
    )
    _append_equation_history(
        history_path=equations_history_path,
        tuning_run_index=tuning_run_index,
        best_row=best_row,
        best_model=best_model,
    )

    print(f"[tuning] tested {len(results_table)} combinations", flush=True)
    print(
        "[tuning] best parameters: "
        f"threshold={best_row['threshold']}, "
        f"alpha={best_row['alpha']}, "
        f"degree={best_row['polynomial_degree']}",
        flush=True,
    )
    print(f"[tuning] best score: {best_row['mean_score']:.6f}", flush=True)

    return HyperparameterSearchOutput(
        tuning_run_index=tuning_run_index,
        combination_count=int(len(results_table)),
        best_parameters=best_summary_payload["best_parameters"],
        best_score=float(best_row["mean_score"]),
        results_path=results_path,
        best_summary_path=best_summary_path,
        equations_history_path=equations_history_path,
        applied_to_model=applied_to_model,
    )


if __name__ == "__main__":
    run_hyperparameter_search()

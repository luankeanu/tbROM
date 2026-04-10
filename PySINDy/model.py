"""
PySINDy model training, hyperparameter search, and prediction helpers.
"""

from __future__ import annotations

import inspect
from itertools import product
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import config
import read

try:
    import pysindy as ps
except ImportError:  # pragma: no cover
    ps = None


def log_progress(message: str) -> None:
    if config.PROGRESS_MESSAGES:
        print(message, flush=True)


def format_params(params: dict) -> str:
    return (
        f"optimizer={params['optimizer']}, "
        f"threshold={params['threshold']}, "
        f"alpha={params['alpha']}, "
        f"degree={params['degree']}, "
        f"fourier={params['fourier_n_frequencies']}, "
        f"pitch_rate={params['include_pitch_rate']}, "
        f"pitch_accel={params['include_pitch_acceleration']}"
    )


def require_pysindy() -> None:
    if ps is None:
        raise ImportError("PySINDy is not installed. Install it before running this project.")


def build_optimizer(name: str, threshold: float, alpha: float):
    require_pysindy()
    if name == "stlsq":
        return ps.STLSQ(threshold=threshold, alpha=alpha)
    if name == "sr3":
        sr3_signature = inspect.signature(ps.SR3)
        if "threshold" in sr3_signature.parameters:
            return ps.SR3(threshold=threshold, nu=1.0, thresholder="l1", max_iter=1000)
        return ps.SR3(
            reg_weight_lam=threshold,
            regularizer="L1",
            relax_coeff_nu=1.0,
            max_iter=1000,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_library(degree: int, fourier_n_frequencies: int):
    require_pysindy()
    libraries = [ps.PolynomialLibrary(degree=degree, include_bias=True)]
    if fourier_n_frequencies > 0:
        libraries.append(ps.FourierLibrary(n_frequencies=fourier_n_frequencies))
    return libraries[0] if len(libraries) == 1 else ps.ConcatLibrary(libraries)


def build_model(params: dict):
    require_pysindy()
    optimizer = build_optimizer(params["optimizer"], params["threshold"], params["alpha"])
    library = build_library(params["degree"], params["fourier_n_frequencies"])
    return ps.SINDy(feature_library=library, optimizer=optimizer)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.shape != y_pred.shape or not np.isfinite(y_pred).all():
        return {
            "rmse_cl": float("inf"),
            "rmse_cd": float("inf"),
            "rmse_mean": float("inf"),
            "mae_cl": float("inf"),
            "mae_cd": float("inf"),
            "mae_mean": float("inf"),
            "r2_cl": float("-inf"),
            "r2_cd": float("-inf"),
            "r2_mean": float("-inf"),
        }

    residual = y_true - y_pred
    mse = np.mean(np.square(residual), axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual), axis=0)
    total_var = np.sum(np.square(y_true - np.mean(y_true, axis=0)), axis=0)
    resid_var = np.sum(np.square(residual), axis=0)
    r2_per_target = np.zeros_like(total_var, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(resid_var, total_var, out=r2_per_target, where=total_var != 0)
    r2_per_target = 1.0 - r2_per_target

    return {
        "rmse_cl": float(rmse[0]),
        "rmse_cd": float(rmse[1]),
        "rmse_mean": float(np.mean(rmse)),
        "mae_cl": float(mae[0]),
        "mae_cd": float(mae[1]),
        "mae_mean": float(np.mean(mae)),
        "r2_cl": float(r2_per_target[0]),
        "r2_cd": float(r2_per_target[1]),
        "r2_mean": float(np.mean(r2_per_target)),
    }


def fit_model(params: dict, train_cases: list[read.CaseData]):
    model = build_model(params)
    x_train, u_train, t_train, _ = read.trajectory_matrices(
        train_cases,
        include_pitch_rate=params["include_pitch_rate"],
        include_pitch_acceleration=params["include_pitch_acceleration"],
    )
    fit_signature = inspect.signature(model.fit)
    fit_kwargs = {"u": u_train, "t": t_train}
    if "multiple_trajectories" in fit_signature.parameters:
        fit_kwargs["multiple_trajectories"] = True
    model.fit(x_train, **fit_kwargs)
    return model


def control_function(time: np.ndarray, control: np.ndarray):
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


def simulate_case(model, case: read.CaseData, params: dict) -> np.ndarray:
    feature_cols = read.feature_columns(
        params["include_pitch_rate"],
        params["include_pitch_acceleration"],
    )
    x0 = case.frame[["cl", "cd"]].iloc[0].to_numpy()
    t = case.frame["flow_time"].to_numpy()
    u = case.frame[feature_cols].to_numpy()
    prediction = model.simulate(x0=x0, t=t, u=control_function(t, u))
    return np.asarray(prediction)


def evaluate_cases(model, cases: list[read.CaseData], params: dict, split_name: str) -> pd.DataFrame:
    rows = []
    for case in cases:
        truth = case.frame[["cl", "cd"]].to_numpy()
        pred = simulate_case(model, case, params)
        rows.append(
            {
                "split": split_name,
                "case_name": case.name,
                "case_group": case.case_group,
                **regression_metrics(truth, pred),
            }
        )
    return pd.DataFrame(rows)


def prediction_frame(model, cases: list[read.CaseData], params: dict, split_name: str) -> pd.DataFrame:
    rows = []
    for case in cases:
        truth = case.frame[["cl", "cd"]].to_numpy()
        pred = simulate_case(model, case, params)
        out = case.frame.copy()
        out.insert(0, "split", split_name)
        out.insert(1, "case_name", case.name)
        out.insert(2, "case_group", case.case_group)
        out["cl_pred"] = pred[:, 0]
        out["cd_pred"] = pred[:, 1]
        out["cl_residual"] = truth[:, 0] - pred[:, 0]
        out["cd_residual"] = truth[:, 1] - pred[:, 1]
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def hyperparameter_combinations() -> list[dict]:
    grid = config.HYPERPARAMETER_GRID
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def search(
    train_cases: list[read.CaseData],
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    require_pysindy()

    all_summary_rows = []
    best_result = None
    best_train_metrics = pd.DataFrame()
    best_validation_metrics = pd.DataFrame()
    best_validation_predictions = pd.DataFrame()
    best_model = None
    combinations = hyperparameter_combinations()

    log_progress(
        f"Hyperparameter search will evaluate {len(combinations)} combinations "
        f"across {len(read.validation_splits(train_cases))} validation splits."
    )

    for index, params in enumerate(combinations, start=1):
        log_progress(f"[{index}/{len(combinations)}] Testing {format_params(params)}")
        split_rows = []
        split_train_tables = []
        split_validation_tables = []
        split_validation_predictions = []

        for train_fold, validation_fold, split_name in read.validation_splits(train_cases):
            if not validation_fold:
                continue

            try:
                log_progress(
                    f"  - Fitting split {split_name}: "
                    f"{len(train_fold)} train case(s), {len(validation_fold)} validation case(s)"
                )
                current_model = fit_model(params, train_fold)
                train_metrics = evaluate_cases(
                    current_model,
                    train_fold,
                    params,
                    split_name + "_train",
                )
                validation_metrics = evaluate_cases(
                    current_model,
                    validation_fold,
                    params,
                    split_name + "_validation",
                )
                validation_prediction = prediction_frame(
                    current_model,
                    validation_fold,
                    params,
                    split_name + "_validation",
                )
            except Exception:
                log_progress(f"  - Split {split_name} failed and was skipped.")
                continue

            split_train_tables.append(train_metrics)
            split_validation_tables.append(validation_metrics)
            split_validation_predictions.append(validation_prediction)
            split_rows.append(
                {
                    "split": split_name,
                    "validation_rmse_mean": float(validation_metrics["rmse_mean"].mean()),
                    "validation_mae_mean": float(validation_metrics["mae_mean"].mean()),
                    "validation_r2_mean": float(validation_metrics["r2_mean"].mean()),
                    "train_rmse_mean": float(train_metrics["rmse_mean"].mean()),
                }
            )
            log_progress(
                "  - "
                f"{split_name} complete: "
                f"validation RMSE={float(validation_metrics['rmse_mean'].mean()):.6e}"
            )

        if not split_rows:
            log_progress("  - No valid splits completed for this parameter set.")
            continue

        split_df = pd.DataFrame(split_rows)
        summary = {
            **params,
            "train_rmse_mean": float(split_df["train_rmse_mean"].mean()),
            "validation_rmse_mean": float(split_df["validation_rmse_mean"].mean()),
            "validation_mae_mean": float(split_df["validation_mae_mean"].mean()),
            "validation_r2_mean": float(split_df["validation_r2_mean"].mean()),
            "rmse_mean": float(split_df["validation_rmse_mean"].mean()),
        }
        all_summary_rows.append(summary)
        log_progress(
            "  - Combination summary: "
            f"mean validation RMSE={summary['validation_rmse_mean']:.6e}, "
            f"mean validation R2={summary['validation_r2_mean']:.6e}"
        )

        current_score = summary[config.SELECTION_METRIC]
        if best_result is None or current_score < best_result[config.SELECTION_METRIC]:
            best_result = summary
            best_train_metrics = pd.concat(split_train_tables, ignore_index=True)
            best_validation_metrics = pd.concat(split_validation_tables, ignore_index=True)
            best_validation_predictions = pd.concat(
                split_validation_predictions,
                ignore_index=True,
            )
            best_model = fit_model(params, train_cases)
            log_progress(
                "  - New best model found: "
                f"{config.SELECTION_METRIC}={current_score:.6e}"
            )

    if best_result is None or best_model is None:
        raise RuntimeError("Hyperparameter search did not produce a valid model.")

    results_df = pd.DataFrame(all_summary_rows).sort_values(
        by=config.SELECTION_METRIC,
        ascending=True,
    )
    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    return (
        best_result,
        results_df,
        best_train_metrics,
        best_validation_metrics,
        best_validation_predictions,
        best_model,
    )


def save_model(model, path: Path | None = None) -> Path:
    target = path or (config.OUTPUT_DIR / config.MODEL_ARTIFACT_FILE)
    with target.open("wb") as handle:
        pickle.dump(model, handle)
    return target


def save_best_summary(best_result: dict, path: Path | None = None) -> Path:
    target = path or (config.OUTPUT_DIR / config.BEST_MODEL_SUMMARY_FILE)
    pd.DataFrame([best_result]).to_csv(target, index=False)
    return target


def export_predictions(model, cases: list[read.CaseData], params: dict) -> pd.DataFrame:
    return prediction_frame(model, cases, params, split_name="full_model")

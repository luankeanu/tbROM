"""
Read module for buffet simulations.
This file is intentionally designed as a sectioned script-style data layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd

import config


"""
Section 1: Schema and case container.
Creates a dataclass used as the container for one simulation;
    Contains metadata (name, path etc) and the data itself in 'frame'
"""

RAW_COLUMNS = ["time_step", "flow_time", "vy", "vx", "cl", "cd"]


@dataclass
class CaseData:
    name: str
    path: Path
    case_group: str
    is_test: bool
    frame: pd.DataFrame


"""
Section 2: Filesystem and split helpers.

ensure_output creates \outputs\ folder
the other 3 functions extract the necessary info from the simulation data files.
"""

# potentially move this into utils.py
def ensure_output_dir() -> Path:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return config.OUTPUT_DIR

# potentially move this into utils.py
def case_group_from_name(name: str) -> str:
    return name[:1].upper() if name else "?"

# potentially move this into utils.py
def is_test_case(name: str) -> bool:
    return name.upper().startswith(config.TEST_PREFIXES)

# potentially move this into utils.py
def discover_data_files(data_dir: Path | None = None) -> list[Path]:
    target_dir = data_dir or config.DATA_DIR
    return sorted(
        path for path in target_dir.glob("*.out") if not path.name.startswith("._")
    )


"""
Section 3
Import and validation.
"""


def read_case_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=config.HEADER_LINES,
        names=RAW_COLUMNS,
        engine="python",
    )


def validate_raw_frame(frame: pd.DataFrame, expected_rows: int) -> dict:
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    flow_time = numeric_frame["flow_time"].to_numpy()
    time_deltas = np.diff(flow_time) if len(flow_time) >= 2 else np.array([])

    validation = {
        "raw_rows": int(len(frame)),
        "expected_rows": int(expected_rows),
        "row_count_ok": bool(len(frame) == expected_rows),
        "missing_values": int(numeric_frame.isna().sum().sum()),
        "duplicate_time_rows": int(numeric_frame["flow_time"].duplicated().sum()),
        "monotonic_time": bool(numeric_frame["flow_time"].is_monotonic_increasing),
        "dt_mean": float(np.mean(time_deltas)) if len(time_deltas) else np.nan,
        "dt_std": float(np.std(time_deltas)) if len(time_deltas) else np.nan,
    }
    validation["is_valid"] = bool(
        validation["row_count_ok"]
        and validation["missing_values"] == 0
        and validation["duplicate_time_rows"] == 0
        and validation["monotonic_time"]
    )
    return validation


"""
Section 4
Treatment and feature engineering.
"""


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def differentiate(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.zeros_like(values)
    return np.gradient(values, time, edge_order=2)


def trim_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[config.ROWS_TO_SKIP :].reset_index(drop=True)


def add_pitch_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    pitch_rad = np.arctan2(data["vy"].to_numpy(), data["vx"].to_numpy())
    pitch_deg = np.degrees(pitch_rad)
    pitch_working = pitch_deg if config.PITCH_IN_DEGREES else pitch_rad
    pitch_smooth = smooth_series(pitch_working, config.SMOOTHING_WINDOW)

    pitch_rate = smooth_series(
        differentiate(pitch_smooth, data["flow_time"].to_numpy()),
        config.SMOOTHING_WINDOW,
    )
    pitch_accel = smooth_series(
        differentiate(pitch_rate, data["flow_time"].to_numpy()),
        config.SMOOTHING_WINDOW,
    )

    data["pitch_rad"] = pitch_rad
    data["pitch_deg"] = pitch_deg
    data["pitch"] = pitch_working
    data["pitch_rate"] = pitch_rate
    data["pitch_accel"] = pitch_accel
    data["cl_lag1"] = data["cl"].shift(1).bfill()
    data["cd_lag1"] = data["cd"].shift(1).bfill()
    return data


"""
Section 5
Per-case build and runtime debug output.
"""


def prepare_case(path: Path) -> tuple[CaseData, dict]:
    raw = read_case_file(path)
    validation = validate_raw_frame(raw, config.EXPECTED_NUMERIC_ROWS)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    trimmed = trim_frame(numeric)
    enriched = add_pitch_features(trimmed)

    case_name = path.stem.replace("_OUTPUT", "")
    case = CaseData(
        name=case_name,
        path=path,
        case_group=case_group_from_name(case_name),
        is_test=is_test_case(case_name),
        frame=enriched,
    )

    validation.update(
        {
            "file_name": path.name,
            "case_name": case_name,
            "case_group": case.case_group,
            "is_test": case.is_test,
            "retained_rows": int(len(enriched)),
            "retained_row_count_ok": bool(len(enriched) == config.EXPECTED_RETAINED_ROWS),
        }
    )
    validation["is_valid"] = bool(
        validation["is_valid"] and validation["retained_row_count_ok"]
    )
    return case, validation


def load_all_cases(data_dir: Path | None = None) -> tuple[list[CaseData], pd.DataFrame]:
    files = discover_data_files(data_dir)
    print(f"[read] files found: {len(files)}", flush=True)

    cases: list[CaseData] = []
    rows: list[dict] = []

    for file_path in files:
        case, validation = prepare_case(file_path)
        cases.append(case)
        rows.append(validation)
        split_label = "TEST" if case.is_test else "TRAIN"
        print(
            f"[read] loaded {file_path.name} as {split_label} "
            f"with {validation['retained_rows']} retained rows",
            flush=True,
        )

    validation_df = pd.DataFrame(rows)
    train_count = sum(not case.is_test for case in cases)
    test_count = sum(case.is_test for case in cases)
    print(f"[read] train cases: {train_count}", flush=True)
    print(f"[read] test cases: {test_count}", flush=True)

    if not validation_df.empty:
        valid_count = int(validation_df["is_valid"].sum())
        print(f"[read] valid files: {valid_count}/{len(validation_df)}", flush=True)

    return cases, validation_df


"""
Section 6
Combined exports and summary tables.
"""


def combined_database(cases: Iterable[CaseData]) -> pd.DataFrame:
    frames = []
    for case in cases:
        frame = case.frame.copy()
        frame.insert(0, "source_file", case.path.name)
        frame.insert(1, "case_name", case.name)
        frame.insert(2, "case_group", case.case_group)
        frame.insert(3, "is_test", case.is_test)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def case_summary(cases: Iterable[CaseData]) -> pd.DataFrame:
    rows = []
    for case in cases:
        frame = case.frame
        rows.append(
            {
                "case_name": case.name,
                "case_group": case.case_group,
                "is_test": case.is_test,
                "rows": len(frame),
                "flow_time_start": float(frame["flow_time"].min()),
                "flow_time_end": float(frame["flow_time"].max()),
                "pitch_min": float(frame["pitch"].min()),
                "pitch_max": float(frame["pitch"].max()),
                "cl_min": float(frame["cl"].min()),
                "cl_max": float(frame["cl"].max()),
                "cd_min": float(frame["cd"].min()),
                "cd_max": float(frame["cd"].max()),
            }
        )
    return pd.DataFrame(rows)


def export_prepared_data(cases: list[CaseData], validation_df: pd.DataFrame) -> dict[str, Path]:
    ensure_output_dir()

    database_df = combined_database(cases)
    summary_df = case_summary(cases)
    correlation_cols = ["vx", "vy", "pitch_deg", "pitch_rate", "pitch_accel", "cl", "cd"]
    correlation_df = database_df[correlation_cols].corr() if not database_df.empty else pd.DataFrame()

    paths = {
        "validation": config.OUTPUT_DIR / config.VALIDATION_FILE,
        "database": config.OUTPUT_DIR / config.DATABASE_FILE,
        "summary": config.OUTPUT_DIR / config.CASE_SUMMARY_FILE,
        "correlation": config.OUTPUT_DIR / config.CORRELATION_FILE,
    }

    validation_df.to_csv(paths["validation"], index=False)
    database_df.to_csv(paths["database"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    correlation_df.to_csv(paths["correlation"])

    return paths


"""
Section 7
Model-facing trajectory utilities.
"""


def split_cases(cases: list[CaseData]) -> tuple[list[CaseData], list[CaseData]]:
    train_cases = [case for case in cases if not case.is_test]
    test_cases = [case for case in cases if case.is_test]
    return train_cases, test_cases


def feature_columns(include_pitch_rate: bool, include_pitch_acceleration: bool) -> list[str]:
    columns = ["pitch"]
    if include_pitch_rate:
        columns.append("pitch_rate")
    if include_pitch_acceleration:
        columns.append("pitch_accel")
    return columns


def trajectory_matrices(
    cases: list[CaseData],
    include_pitch_rate: bool,
    include_pitch_acceleration: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    feature_cols = feature_columns(include_pitch_rate, include_pitch_acceleration)
    x_list = [case.frame[["cl", "cd"]].to_numpy() for case in cases]
    u_list = [case.frame[feature_cols].to_numpy() for case in cases]
    t_list = [case.frame["flow_time"].to_numpy() for case in cases]
    names = [case.name for case in cases]
    return x_list, u_list, t_list, names


def validation_splits(cases: list[CaseData]) -> list[tuple[list[CaseData], list[CaseData], str]]:
    if len(cases) < 2:
        return [(cases, [], "no_validation_split")]

    if config.VALIDATION_MODE == "single_holdout":
        holdout_names = set(config.HOLDOUT_FILES)
        val_cases = [case for case in cases if case.name in holdout_names]
        train_cases = [case for case in cases if case.name not in holdout_names]
        if train_cases and val_cases:
            label = "holdout_" + "_".join(case.name for case in val_cases)
            return [(train_cases, val_cases, label)]

    return [
        ([item for item in cases if item.name != case.name], [case], f"holdout_{case.name}")
        for case in cases
    ]


"""
Section 8
Run configuration snapshot.
"""


def export_run_config() -> Path:
    ensure_output_dir()
    payload = {
        "data_dir": str(config.DATA_DIR),
        "output_dir": str(config.OUTPUT_DIR),
        "test_prefixes": list(config.TEST_PREFIXES),
        "header_lines": config.HEADER_LINES,
        "rows_to_skip": config.ROWS_TO_SKIP,
        "expected_numeric_rows": config.EXPECTED_NUMERIC_ROWS,
        "expected_retained_rows": config.EXPECTED_RETAINED_ROWS,
        "pitch_in_degrees": config.PITCH_IN_DEGREES,
        "include_pitch_acceleration": config.INCLUDE_PITCH_ACCELERATION,
        "plot_window": list(config.PLOT_WINDOW) if config.PLOT_WINDOW is not None else None,
        "validation_mode": config.VALIDATION_MODE,
        "holdout_files": list(config.HOLDOUT_FILES),
        "hyperparameter_grid": config.HYPERPARAMETER_GRID,
        "selection_metric": config.SELECTION_METRIC,
    }
    path = config.OUTPUT_DIR / config.RUN_CONFIG_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

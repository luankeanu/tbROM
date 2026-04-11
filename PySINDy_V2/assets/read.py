from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

###Section 1: Project paths, read-stage assumptions, and returned containers.
"""
This section fixes the filesystem layout and the Fluent-format assumptions that
the whole read stage depends on. It also defines the small containers returned
to `main.py` and, later, to the rest of the workflow.
"""

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

RAW_COLUMNS = ["time_step", "flow_time", "vy", "vx", "cl", "cd"]
HEADER_LINE_COUNT = 3
EXPECTED_NUMERIC_ROWS = 8000
ROWS_TO_DISCARD = 500
EXPECTED_RETAINED_ROWS = EXPECTED_NUMERIC_ROWS - ROWS_TO_DISCARD

OUTPUT_FILENAMES = {
    "train": "prepared_train_cases.csv",
    "confirmation": "prepared_confirmation_cases.csv",
    "validation": "file_validation.csv",
    "summary": "case_summary.csv",
}


"""
Container: CaseRecord.
Stores one prepared trajectory together with the metadata that identifies where
it came from and whether it belongs to the training or confirmation split.
"""


@dataclass(frozen=True)
class CaseRecord:
    case_name: str
    source_file: str
    case_group: str
    is_confirmation: bool
    table: pd.DataFrame


"""
Container: ReadStageOutput.
Bundles all Pandas objects produced by the read stage so later scripts can
reuse the prepared data without re-reading the raw Fluent files.
"""


@dataclass(frozen=True)
class ReadStageOutput:
    cases: dict[str, pd.DataFrame]
    validation_table: pd.DataFrame
    summary_table: pd.DataFrame
    train_table: pd.DataFrame
    confirmation_table: pd.DataFrame
    output_paths: dict[str, Path]



### Section 2: File discovery and case naming.
"""
These helpers answer three basic questions before any data is read:
which files should be loaded, what case name should each file map to, and
whether a case belongs to the training or confirmation dataset.
"""


"""
Function: discover_case_files.
Looks inside the project `data/` directory, keeps only valid `.out` files, and
ignores hidden macOS clutter if it ever appears again.
"""


def discover_case_files(data_dir: Path = DATA_DIR) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    case_files = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix == ".out" and not path.name.startswith("._")
    )

    if not case_files:
        raise FileNotFoundError(f"No .out files found in {data_dir}")

    return case_files


"""
Function: case_name_from_path.
Converts a Fluent filename such as `B1_OUTPUT.out` into the shorter case label
used throughout the workflow, for example `B1`.
"""


def case_name_from_path(path: Path) -> str:
    return path.stem.replace("_OUTPUT", "")


"""
Function: case_group_from_name.
Extracts the leading letter from a case name so related cases can be grouped
later in summaries, comparisons, and plots.
"""


def case_group_from_name(case_name: str) -> str:
    return case_name[:1].upper() if case_name else "?"


"""
Function: is_confirmation_case.
Marks files that begin with `T` as confirmation cases. Everything else is
treated as development and training data.
"""


def is_confirmation_case(case_name: str) -> bool:
    return case_name.upper().startswith("T")


### Section 3: Raw import and validation.
"""
The goal of this section is to read one Fluent file exactly once, coerce it to
numeric form, and create a full validation record before any exports happen.
The workflow is intentionally strict: a bad file should stop the run.
"""


"""
Function: read_raw_case.
Reads the numeric portion of one Fluent output file using the fixed six-column
schema agreed for this project.
"""


def read_raw_case(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=HEADER_LINE_COUNT,
        names=RAW_COLUMNS,
        engine="python",
    )


"""
Function: build_validation_row.
Builds one row of the validation table. This row records whether the input file
matches the expected size and whether the time column is usable for later
trajectory-based modelling.
"""


def build_validation_row(
    path: Path,
    numeric_table: pd.DataFrame,
    case_name: str,
    case_group: str,
    is_confirmation: bool,
) -> dict[str, object]:
    errors: list[str] = []

    raw_row_count = int(len(numeric_table))
    missing_values = int(numeric_table.isna().sum().sum())
    duplicate_time_rows = int(numeric_table["flow_time"].duplicated().sum())
    monotonic_time = bool(numeric_table["flow_time"].is_monotonic_increasing)

    if raw_row_count != EXPECTED_NUMERIC_ROWS:
        errors.append(
            f"expected {EXPECTED_NUMERIC_ROWS} numeric rows, found {raw_row_count}"
        )

    if missing_values:
        errors.append(f"numeric coercion produced {missing_values} missing values")

    if duplicate_time_rows:
        errors.append(f"found {duplicate_time_rows} duplicate flow_time rows")

    if not monotonic_time:
        errors.append("flow_time is not monotonically increasing")

    retained_row_count = max(raw_row_count - ROWS_TO_DISCARD, 0)
    if retained_row_count != EXPECTED_RETAINED_ROWS:
        errors.append(
            f"expected {EXPECTED_RETAINED_ROWS} retained rows, found {retained_row_count}"
        )

    flow_time = numeric_table["flow_time"].dropna().to_numpy()
    time_steps = np.diff(flow_time) if len(flow_time) >= 2 else np.array([])

    return {
        "source_file": path.name,
        "case_name": case_name,
        "case_group": case_group,
        "dataset_role": "confirmation" if is_confirmation else "train",
        "is_confirmation": is_confirmation,
        "raw_rows": raw_row_count,
        "retained_rows": retained_row_count,
        "missing_values": missing_values,
        "duplicate_time_rows": duplicate_time_rows,
        "monotonic_time": monotonic_time,
        "dt_mean": float(np.mean(time_steps)) if len(time_steps) else np.nan,
        "dt_std": float(np.std(time_steps)) if len(time_steps) else np.nan,
        "is_valid": not errors,
        "validation_message": "OK" if not errors else "; ".join(errors),
    }


"""
Function: raise_for_invalid_cases.
Examines the full validation table and aborts the read stage if any file fails.
This prevents later scripts from training on a partial or corrupted dataset.
"""


def raise_for_invalid_cases(validation_table: pd.DataFrame) -> None:
    invalid_rows = validation_table.loc[~validation_table["is_valid"]]

    if invalid_rows.empty:
        return

    report_lines = ["Invalid input files detected:"]
    for row in invalid_rows.itertuples(index=False):
        report_lines.append(f"- {row.source_file}: {row.validation_message}")

    raise ValueError("\n".join(report_lines))


### Section 4: Case treatment and feature creation.
"""
This section contains the actual transformation from raw Fluent data to the V2
prepared trajectory format. The agreed treatment is simple on purpose:
discard the first 500 numeric rows and compute only the canonical `pitch(t)`.
"""


"""
Function: trim_numeric_rows.
Drops the transient opening part of each simulation so all cases start from the
same retained window used by the rest of the project.
"""


def trim_numeric_rows(numeric_table: pd.DataFrame) -> pd.DataFrame:
    return numeric_table.iloc[ROWS_TO_DISCARD:].reset_index(drop=True)


"""
Function: compute_pitch_degrees.
Converts the velocity components into a pitch history using `atan2(vy, vx)` and
stores the result in degrees, which is the canonical V2 unit.
"""


def compute_pitch_degrees(prepared_table: pd.DataFrame) -> np.ndarray:
    return np.degrees(
        np.arctan2(
            prepared_table["vy"].to_numpy(),
            prepared_table["vx"].to_numpy(),
        )
    )


"""
Function: build_prepared_case_table.
Creates the table that later scripts will actually consume. This is where the
case metadata is attached to every retained row and where `pitch(t)` is added.
"""


def build_prepared_case_table(
    numeric_table: pd.DataFrame,
    source_file: str,
    case_name: str,
    case_group: str,
    is_confirmation: bool,
) -> pd.DataFrame:
    prepared_table = trim_numeric_rows(numeric_table)
    prepared_table.insert(0, "source_file", source_file)
    prepared_table.insert(1, "case_name", case_name)
    prepared_table.insert(2, "case_group", case_group)
    prepared_table.insert(
        3,
        "dataset_role",
        "confirmation" if is_confirmation else "train",
    )
    prepared_table.insert(4, "is_confirmation", is_confirmation)
    prepared_table["pitch"] = compute_pitch_degrees(prepared_table)
    return prepared_table


"""
Function: prepare_case.
Runs the full per-file workflow: derive metadata, read the raw file, validate
the numeric content, build the prepared table, and return both outputs together.
"""


def prepare_case(path: Path) -> tuple[CaseRecord, dict[str, object]]:
    case_name = case_name_from_path(path)
    case_group = case_group_from_name(case_name)
    is_confirmation = is_confirmation_case(case_name)

    raw_table = read_raw_case(path)
    numeric_table = raw_table.apply(pd.to_numeric, errors="coerce")
    validation_row = build_validation_row(
        path=path,
        numeric_table=numeric_table,
        case_name=case_name,
        case_group=case_group,
        is_confirmation=is_confirmation,
    )

    prepared_table = build_prepared_case_table(
        numeric_table=numeric_table,
        source_file=path.name,
        case_name=case_name,
        case_group=case_group,
        is_confirmation=is_confirmation,
    )

    case_record = CaseRecord(
        case_name=case_name,
        source_file=path.name,
        case_group=case_group,
        is_confirmation=is_confirmation,
        table=prepared_table,
    )

    return case_record, validation_row


### Section 5: Combined tables and CSV exports.
"""
After every case has been prepared, this section creates the shared tables used
for downstream modelling and writes the read-stage CSV files for debugging and
report traceability.
"""


"""
Function: combine_case_tables.
Stacks multiple prepared trajectories into one table while preserving the case
metadata columns needed to recover the file-level grouping later.
"""


def combine_case_tables(case_records: list[CaseRecord]) -> pd.DataFrame:
    if not case_records:
        return pd.DataFrame(
            columns=[
                "source_file",
                "case_name",
                "case_group",
                "dataset_role",
                "is_confirmation",
                *RAW_COLUMNS,
                "pitch",
            ]
        )

    return pd.concat([case.table for case in case_records], ignore_index=True)


"""
Function: build_summary_table.
Creates one compact row per case so the useful range of each trajectory can be
checked quickly without reopening the raw or prepared datasets manually.
"""


def build_summary_table(case_records: list[CaseRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for case in case_records:
        table = case.table
        rows.append(
            {
                "source_file": case.source_file,
                "case_name": case.case_name,
                "case_group": case.case_group,
                "dataset_role": "confirmation" if case.is_confirmation else "train",
                "is_confirmation": case.is_confirmation,
                "rows": int(len(table)),
                "flow_time_start": float(table["flow_time"].min()),
                "flow_time_end": float(table["flow_time"].max()),
                "pitch_min": float(table["pitch"].min()),
                "pitch_max": float(table["pitch"].max()),
                "cl_min": float(table["cl"].min()),
                "cl_max": float(table["cl"].max()),
                "cd_min": float(table["cd"].min()),
                "cd_max": float(table["cd"].max()),
            }
        )

    return pd.DataFrame(rows)


"""
Function: ensure_output_directory.
Makes sure the project-level `outputs/` folder exists before any CSV exports are
written there.
"""


def ensure_output_directory() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


"""
Function: export_read_outputs.
Writes the prepared training table, prepared confirmation table, validation
table, and summary table to disk using fixed filenames.
"""


def export_read_outputs(
    train_table: pd.DataFrame,
    confirmation_table: pd.DataFrame,
    validation_table: pd.DataFrame,
    summary_table: pd.DataFrame,
) -> dict[str, Path]:
    ensure_output_directory()

    output_paths = {
        key: OUTPUT_DIR / filename for key, filename in OUTPUT_FILENAMES.items()
    }

    train_table.to_csv(output_paths["train"], index=False)
    confirmation_table.to_csv(output_paths["confirmation"], index=False)
    validation_table.to_csv(output_paths["validation"], index=False)
    summary_table.to_csv(output_paths["summary"], index=False)

    return output_paths


### Section 6: Public entry point for the read stage.
"""
This final section runs the whole read workflow in order, prints a compact
runtime summary, and returns all prepared Pandas objects for reuse by `main.py`
and the future fit, run, and analyse stages.
"""


"""
Function: run_read_stage.
Coordinates the full read-stage pipeline from discovery to export. It only
writes the CSV outputs after all files have passed validation successfully.
"""


def run_read_stage() -> ReadStageOutput:
    case_files = discover_case_files()
    print(f"[read] discovered {len(case_files)} Fluent output files", flush=True)

    case_records: list[CaseRecord] = []
    validation_rows: list[dict[str, object]] = []

    for path in case_files:
        case_record, validation_row = prepare_case(path)
        case_records.append(case_record)
        validation_rows.append(validation_row)
        print(
            f"[read] prepared {path.name} as {validation_row['dataset_role']}",
            flush=True,
        )

    validation_table = pd.DataFrame(validation_rows).sort_values(
        by=["dataset_role", "case_name"]
    )
    raise_for_invalid_cases(validation_table)

    train_cases = [case for case in case_records if not case.is_confirmation]
    confirmation_cases = [case for case in case_records if case.is_confirmation]

    train_table = combine_case_tables(train_cases)
    confirmation_table = combine_case_tables(confirmation_cases)
    summary_table = build_summary_table(case_records).sort_values(
        by=["dataset_role", "case_name"]
    )
    output_paths = export_read_outputs(
        train_table=train_table,
        confirmation_table=confirmation_table,
        validation_table=validation_table,
        summary_table=summary_table,
    )

    print(f"[read] training cases: {len(train_cases)}", flush=True)
    print(f"[read] confirmation cases: {len(confirmation_cases)}", flush=True)
    print(f"[read] training rows: {len(train_table)}", flush=True)
    print(f"[read] confirmation rows: {len(confirmation_table)}", flush=True)

    cases = {case.case_name: case.table.copy() for case in case_records}
    return ReadStageOutput(
        cases=cases,
        validation_table=validation_table.reset_index(drop=True),
        summary_table=summary_table.reset_index(drop=True),
        train_table=train_table,
        confirmation_table=confirmation_table,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    run_read_stage()

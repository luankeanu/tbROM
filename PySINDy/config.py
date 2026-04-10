"""
Project configuration for the buffet PySINDy workflow.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "outputs"

TEST_PREFIXES = ("T",)

HEADER_LINES = 3
ROWS_TO_SKIP = 500
EXPECTED_NUMERIC_ROWS = 8000
EXPECTED_RETAINED_ROWS = EXPECTED_NUMERIC_ROWS - ROWS_TO_SKIP

PITCH_IN_DEGREES = True
INCLUDE_PITCH_ACCELERATION = False
SMOOTHING_WINDOW = 9

PLOT_WINDOW = None
PLOT_RENDERER = "browser"
OPEN_PLOTS_IN_BROWSER = True

VALIDATION_MODE = "leave_one_file_out"
HOLDOUT_FILES = ()
PROGRESS_MESSAGES = True

HYPERPARAMETER_GRID = {
    "optimizer": ["stlsq", "sr3"],
    "threshold": [0.01, 0.05, 0.1],
    "alpha": [0.0, 0.01, 0.05],
    "degree": [1, 2, 3],
    "fourier_n_frequencies": [0, 1, 2],
    "include_pitch_rate": [True],
    "include_pitch_acceleration": [False],
}

SELECTION_METRIC = "rmse_mean"

VALIDATION_FILE = "file_validation.csv"
DATABASE_FILE = "combined_database.csv"
CASE_SUMMARY_FILE = "case_summary.csv"
CORRELATION_FILE = "feature_correlation.csv"
TRAIN_METRICS_FILE = "train_metrics.csv"
VALIDATION_METRICS_FILE = "validation_metrics.csv"
HYPERPARAMETER_RESULTS_FILE = "hyperparameter_results.csv"
BEST_MODEL_SUMMARY_FILE = "best_model_summary.csv"
MODEL_ARTIFACT_FILE = "best_model.pkl"
RUN_CONFIG_FILE = "run_config.json"

"""
Main entrypoint for the buffet PySINDy project.
"""

from __future__ import annotations

import analyse
import config
import model
import pipeline


def log_step(message: str) -> None:
    if config.PROGRESS_MESSAGES:
        print(message, flush=True)


def main() -> None:
    log_step("Initialising output directory and exporting run config...")
    pipeline.ensure_output_dir()
    pipeline.export_run_config()

    log_step(f"Loading and validating cases from {config.DATA_DIR}...")
    cases, validation_df = pipeline.load_all_cases()
    if validation_df.empty:
        raise RuntimeError(f"No .out files were found in {config.DATA_DIR}")
    if not validation_df["is_valid"].all():
        invalid = validation_df.loc[~validation_df["is_valid"], "file_name"].tolist()
        raise RuntimeError(f"Invalid data files detected: {invalid}")
    log_step(f"Loaded {len(cases)} valid case files.")

    log_step("Exporting prepared data tables...")
    pipeline.export_prepared_data(cases, validation_df)
    database_df = pipeline.combined_database(cases)
    log_step("Generating correlation heatmap and overlay plots...")
    analyse.plot_correlation_heatmap(database_df)
    analyse.plot_case_overlays(cases)

    train_cases, test_cases = pipeline.split_cases(cases)
    if len(train_cases) < 2:
        raise RuntimeError("At least two non-test files are needed for grouped validation.")
    log_step(
        f"Prepared {len(train_cases)} development cases and {len(test_cases)} test cases."
    )

    log_step("Starting hyperparameter search...")
    (
        best_result,
        hyper_df,
        train_metrics,
        validation_metrics,
        validation_predictions,
        best_model,
    ) = model.search(train_cases)
    log_step("Hyperparameter search complete. Saving metrics and model artefacts...")
    analyse.save_hyperparameter_results(hyper_df)
    analyse.save_metrics(train_metrics, validation_metrics)
    model.save_best_summary(best_result)
    model.save_model(best_model)

    best_params = {
        "optimizer": best_result["optimizer"],
        "threshold": best_result["threshold"],
        "alpha": best_result["alpha"],
        "degree": best_result["degree"],
        "fourier_n_frequencies": best_result["fourier_n_frequencies"],
        "include_pitch_rate": best_result["include_pitch_rate"],
        "include_pitch_acceleration": best_result["include_pitch_acceleration"],
    }

    log_step("Exporting validation predictions and plots...")
    validation_predictions.to_csv(config.OUTPUT_DIR / "validation_predictions.csv", index=False)
    analyse.prediction_plots(validation_predictions, prefix="validation")

    log_step("Exporting test predictions and plots...")
    test_predictions = model.export_predictions(best_model, test_cases, best_params)
    test_predictions.to_csv(config.OUTPUT_DIR / "test_predictions.csv", index=False)
    analyse.prediction_plots(test_predictions, prefix="test")

    print("Run complete.", flush=True)
    print(f"Prepared files: {len(cases)}", flush=True)
    print(f"Training files: {len(train_cases)}", flush=True)
    print(f"Test files: {len(test_cases)}", flush=True)
    print(f"Best validation RMSE: {best_result[config.SELECTION_METRIC]:.6f}", flush=True)


if __name__ == "__main__":
    main()

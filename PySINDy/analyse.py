"""
Analysis and plotting utilities for the buffet PySINDy workflow.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
import pipeline


def save_metrics(train_metrics: pd.DataFrame, validation_metrics: pd.DataFrame) -> dict[str, Path]:
    pipeline.ensure_output_dir()
    train_path = config.OUTPUT_DIR / config.TRAIN_METRICS_FILE
    validation_path = config.OUTPUT_DIR / config.VALIDATION_METRICS_FILE
    train_metrics.to_csv(train_path, index=False)
    validation_metrics.to_csv(validation_path, index=False)
    return {"train": train_path, "validation": validation_path}


def save_hyperparameter_results(results: pd.DataFrame) -> Path:
    path = config.OUTPUT_DIR / config.HYPERPARAMETER_RESULTS_FILE
    results.to_csv(path, index=False)
    return path


def save_and_show_plot(fig: go.Figure, path: Path) -> Path:
    fig.write_html(path, include_plotlyjs="cdn")
    if config.OPEN_PLOTS_IN_BROWSER:
        fig.show(renderer=config.PLOT_RENDERER)
    return path


def plot_correlation_heatmap(database_df: pd.DataFrame) -> Path | None:
    if database_df.empty:
        return None

    cols = ["vx", "vy", "pitch_deg", "pitch_rate", "pitch_accel", "cl", "cd"]
    corr = database_df[cols].corr()
    text = corr.round(2).astype(str).to_numpy()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.to_numpy(),
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
            colorscale="RdBu",
            text=text,
            texttemplate="%{text}",
            hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Feature Correlation Heatmap",
        template="plotly_white",
        width=900,
        height=700,
    )

    path = config.OUTPUT_DIR / "feature_correlation_heatmap.html"
    return save_and_show_plot(fig, path)


def plot_case_overlays(cases: list[pipeline.CaseData]) -> list[Path]:
    if not cases:
        return []

    if config.PLOT_WINDOW is None:
        start_time = min(float(case.frame["flow_time"].min()) for case in cases)
        end_time = max(float(case.frame["flow_time"].max()) for case in cases)
        title = "Cross-Case Overlays on Full Retained Time Range"
    else:
        start_time, end_time = config.PLOT_WINDOW
        title = f"Cross-Case Overlays on {start_time:.2f}s to {end_time:.2f}s"

    outputs = []
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Pitch", "Cl", "Cd"),
    )

    for case in cases:
        mask = case.frame["flow_time"].between(start_time, end_time)
        window = case.frame.loc[mask]
        if window.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=window["flow_time"],
                y=window["pitch"],
                mode="lines",
                name=case.name,
                legendgroup=case.name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=window["flow_time"],
                y=window["cl"],
                mode="lines",
                name=case.name,
                legendgroup=case.name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=window["flow_time"],
                y=window["cd"],
                mode="lines",
                name=case.name,
                legendgroup=case.name,
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.update_yaxes(title_text="Pitch", row=1, col=1)
    fig.update_yaxes(title_text="Cl", row=2, col=1)
    fig.update_yaxes(title_text="Cd", row=3, col=1)
    fig.update_xaxes(title_text="Flow Time", row=3, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1100,
        height=900,
        legend_title="Case",
    )

    overlay_path = config.OUTPUT_DIR / "overlay_pitch_cl_cd.html"
    outputs.append(save_and_show_plot(fig, overlay_path))
    return outputs


def prediction_plots(prediction_df: pd.DataFrame, prefix: str) -> list[Path]:
    if prediction_df.empty:
        return []

    outputs = []
    for case_name, case_df in prediction_df.groupby("case_name"):
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Cl", "Cd"),
        )
        fig.add_trace(
            go.Scatter(x=case_df["flow_time"], y=case_df["cl"], mode="lines", name="Cl actual"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=case_df["flow_time"],
                y=case_df["cl_pred"],
                mode="lines",
                name="Cl predicted",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=case_df["flow_time"], y=case_df["cd"], mode="lines", name="Cd actual"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=case_df["flow_time"],
                y=case_df["cd_pred"],
                mode="lines",
                name="Cd predicted",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Cl", row=1, col=1)
        fig.update_yaxes(title_text="Cd", row=2, col=1)
        fig.update_xaxes(title_text="Flow Time", row=2, col=1)
        fig.update_layout(
            title=f"{prefix}: {case_name}",
            template="plotly_white",
            width=1100,
            height=700,
        )

        comparison_path = config.OUTPUT_DIR / f"{prefix.lower()}_{case_name}_comparison.html"
        outputs.append(save_and_show_plot(fig, comparison_path))

        residual_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Cl residual", "Cd residual"),
        )
        residual_fig.add_trace(
            go.Scatter(
                x=case_df["flow_time"],
                y=case_df["cl_residual"],
                mode="lines",
                name="Cl residual",
            ),
            row=1,
            col=1,
        )
        residual_fig.add_hline(y=0.0, line_color="black", line_width=1, row=1, col=1)
        residual_fig.add_trace(
            go.Scatter(
                x=case_df["flow_time"],
                y=case_df["cd_residual"],
                mode="lines",
                name="Cd residual",
            ),
            row=2,
            col=1,
        )
        residual_fig.add_hline(y=0.0, line_color="black", line_width=1, row=2, col=1)
        residual_fig.update_yaxes(title_text="Cl residual", row=1, col=1)
        residual_fig.update_yaxes(title_text="Cd residual", row=2, col=1)
        residual_fig.update_xaxes(title_text="Flow Time", row=2, col=1)
        residual_fig.update_layout(
            title=f"{prefix} Residuals: {case_name}",
            template="plotly_white",
            width=1100,
            height=700,
        )

        residual_path = config.OUTPUT_DIR / f"{prefix.lower()}_{case_name}_residuals.html"
        outputs.append(save_and_show_plot(residual_fig, residual_path))

    return outputs

from __future__ import annotations

import inspect
from typing import Any


try:
    import pysindy as ps
except ImportError:  # pragma: no cover
    ps = None


"""
Section 1: Canonical V3 metadata and default settings.
This version of the project uses two separate one-output PySINDy models:
one for `cl` and one for `cd`. Both models share the same control inputs.
"""

MODEL_NAME = "split_pysindy_control_models"
MODEL_DESCRIPTION = (
    "Two sparse polynomial SINDy models: one for cl and one for cd, both driven by pitch and pitch_rate."
)

TARGET_COLUMNS = ("cl", "cd")
CONTROL_COLUMNS = ("pitch", "pitch_rate")
TIME_COLUMN = "flow_time"

DEFAULT_LIBRARY_TYPE = "polynomial"
DEFAULT_POLYNOMIAL_DEGREE = 2
DEFAULT_INCLUDE_BIAS = True
DEFAULT_INCLUDE_INTERACTION = True

DEFAULT_OPTIMIZER_TYPE = "stlsq"
DEFAULT_THRESHOLD = 0.01
DEFAULT_ALPHA = 0.0
DEFAULT_MAX_ITER = 20
DEFAULT_NORMALIZE_COLUMNS = False

DEFAULT_DIFFERENTIATION_TYPE = "finite_difference"
DEFAULT_FINITE_DIFFERENCE_ORDER = 2


"""
Section 2: Public metadata helpers.
These helpers expose the overall V3 signal definitions so later stages can keep
one shared understanding of the target and control columns.
"""


def get_state_columns() -> list[str]:
    return list(TARGET_COLUMNS)


def get_target_columns() -> list[str]:
    return list(TARGET_COLUMNS)


def get_control_columns() -> list[str]:
    return list(CONTROL_COLUMNS)


def get_time_column() -> str:
    return TIME_COLUMN


def get_default_settings() -> dict[str, Any]:
    return {
        "library_type": DEFAULT_LIBRARY_TYPE,
        "polynomial_degree": DEFAULT_POLYNOMIAL_DEGREE,
        "include_bias": DEFAULT_INCLUDE_BIAS,
        "include_interaction": DEFAULT_INCLUDE_INTERACTION,
        "optimizer_type": DEFAULT_OPTIMIZER_TYPE,
        "threshold": DEFAULT_THRESHOLD,
        "alpha": DEFAULT_ALPHA,
        "max_iter": DEFAULT_MAX_ITER,
        "normalize_columns": DEFAULT_NORMALIZE_COLUMNS,
        "differentiation_type": DEFAULT_DIFFERENTIATION_TYPE,
        "finite_difference_order": DEFAULT_FINITE_DIFFERENCE_ORDER,
    }


def _resolve_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved_settings = get_default_settings()
    if settings is not None:
        resolved_settings.update(settings)
    return resolved_settings


def get_model_summary(
    target_column: str | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_settings = _resolve_settings(settings)
    summary = {
        "model_name": MODEL_NAME,
        "description": MODEL_DESCRIPTION,
        "target_columns": get_target_columns(),
        "control_columns": get_control_columns(),
        "time_column": get_time_column(),
        "library_type": resolved_settings["library_type"],
        "polynomial_degree": resolved_settings["polynomial_degree"],
        "include_bias": resolved_settings["include_bias"],
        "include_interaction": resolved_settings["include_interaction"],
        "optimizer_type": resolved_settings["optimizer_type"],
        "threshold": resolved_settings["threshold"],
        "alpha": resolved_settings["alpha"],
        "max_iter": resolved_settings["max_iter"],
        "normalize_columns": resolved_settings["normalize_columns"],
        "differentiation_type": resolved_settings["differentiation_type"],
        "finite_difference_order": resolved_settings["finite_difference_order"],
    }
    if target_column is not None:
        summary["target_column"] = target_column
    return summary


"""
Section 3: Internal PySINDy builders.
These helpers build the shared PySINDy pieces used by both the `cl` and `cd`
target models.
"""


def _require_pysindy() -> None:
    if ps is None:
        raise ImportError(
            "PySINDy is not installed. Install `pysindy` before building the model."
        )


def _build_feature_library(settings: dict[str, Any] | None = None):
    _require_pysindy()
    resolved_settings = _resolve_settings(settings)
    if resolved_settings["library_type"] != "polynomial":
        raise ValueError(
            f"Unsupported library type: {resolved_settings['library_type']}"
        )
    return ps.PolynomialLibrary(
        degree=resolved_settings["polynomial_degree"],
        include_bias=resolved_settings["include_bias"],
        include_interaction=resolved_settings["include_interaction"],
    )


def _build_optimizer(settings: dict[str, Any] | None = None):
    _require_pysindy()
    resolved_settings = _resolve_settings(settings)
    if resolved_settings["optimizer_type"] != "stlsq":
        raise ValueError(
            f"Unsupported optimizer type: {resolved_settings['optimizer_type']}"
        )
    return ps.STLSQ(
        threshold=resolved_settings["threshold"],
        alpha=resolved_settings["alpha"],
        max_iter=resolved_settings["max_iter"],
        normalize_columns=resolved_settings["normalize_columns"],
    )


def _build_differentiation_method(settings: dict[str, Any] | None = None):
    _require_pysindy()
    resolved_settings = _resolve_settings(settings)
    if resolved_settings["differentiation_type"] != "finite_difference":
        raise ValueError(
            "Unsupported differentiation type: "
            f"{resolved_settings['differentiation_type']}"
        )
    return ps.FiniteDifference(order=resolved_settings["finite_difference_order"])


"""
Section 4: Public model builders.
This section exposes the one-target model builder used by both the fit and run
stages in V3.
"""


def build_pysindy_model(
    target_column: str,
    settings: dict[str, Any] | None = None,
):
    if target_column not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported target column: {target_column}")

    _require_pysindy()

    feature_library = _build_feature_library(settings)
    optimizer = _build_optimizer(settings)
    differentiation_method = _build_differentiation_method(settings)

    sindy_signature = inspect.signature(ps.SINDy)
    build_kwargs = {
        "feature_library": feature_library,
        "optimizer": optimizer,
        "differentiation_method": differentiation_method,
    }
    if "discrete_time" in sindy_signature.parameters:
        build_kwargs["discrete_time"] = False

    return ps.SINDy(**build_kwargs)

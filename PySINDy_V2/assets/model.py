from __future__ import annotations

import inspect
from typing import Any


try:
    import pysindy as ps
except ImportError:  # pragma: no cover
    ps = None


"""
Section 1: Canonical model metadata and default settings.
This section fixes the shared modelling assumptions for the whole V2 workflow.
It defines which prepared columns belong to the state vector, which column is
treated as the external control input, and which sparse-regression settings are
used by default when building the baseline PySINDy model.
"""

MODEL_NAME = "baseline_pysindy_control_model"
MODEL_DESCRIPTION = (
    "Sparse polynomial SINDy model with cl/cd as states and pitch as control."
)

STATE_COLUMNS = ("cl", "cd")
CONTROL_COLUMNS = ("pitch",)
TIME_COLUMN = "flow_time"

DEFAULT_LIBRARY_TYPE = "polynomial"
DEFAULT_POLYNOMIAL_DEGREE = 3
DEFAULT_INCLUDE_BIAS = True
DEFAULT_INCLUDE_INTERACTION = True

DEFAULT_OPTIMIZER_TYPE = "stlsq"
DEFAULT_THRESHOLD = 0.1
DEFAULT_ALPHA = 0.05
DEFAULT_MAX_ITER = 20
DEFAULT_NORMALIZE_COLUMNS = False

DEFAULT_DIFFERENTIATION_TYPE = "finite_difference"
DEFAULT_FINITE_DIFFERENCE_ORDER = 2


"""
Section 2: Public metadata helpers.
These functions expose the canonical V2 signal definitions so later stages do
not hard-code column names or silently drift away from the read-stage schema.
"""


"""
Function: get_state_columns.
Returns the ordered names of the signals that form the PySINDy state vector.
For this project, the model evolves lift and drag coefficients in time.
"""


def get_state_columns() -> list[str]:
    return list(STATE_COLUMNS)


"""
Function: get_control_columns.
Returns the ordered names of the external control inputs. The current V2 design
uses the pitch history as the only forcing signal.
"""


def get_control_columns() -> list[str]:
    return list(CONTROL_COLUMNS)


"""
Function: get_time_column.
Returns the name of the prepared time column that later stages should pass to
PySINDy during fitting and simulation.
"""


def get_time_column() -> str:
    return TIME_COLUMN


"""
Function: get_default_settings.
Returns the live default model settings as a plain dictionary so other stages
can inspect or override them without duplicating the canonical values.
"""


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


"""
Function: _resolve_settings.
Merges optional overrides onto the live defaults so builders can construct
alternative candidate models while the rest of the project still sees one
canonical default configuration.
"""


def _resolve_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved_settings = get_default_settings()
    if settings is not None:
        resolved_settings.update(settings)
    return resolved_settings


"""
Function: get_model_summary.
Builds a compact description of the canonical model definition. This summary is
intended for later use in logging, reporting, or saved run metadata.
"""


def get_model_summary(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved_settings = _resolve_settings(settings)
    return {
        "model_name": MODEL_NAME,
        "description": MODEL_DESCRIPTION,
        "state_columns": get_state_columns(),
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


"""
Section 3: Internal PySINDy builders.
These helpers keep the construction steps readable while preserving the project
style of exposing one main public model-building function.
"""


"""
Function: _require_pysindy.
Guards the public builder from failing silently when PySINDy is not available.
Importing this module remains safe, but actually building the model requires the
package to be installed in the active Python environment.
"""


def _require_pysindy() -> None:
    if ps is None:
        raise ImportError(
            "PySINDy is not installed. Install `pysindy` before building the model."
        )


"""
Function: _build_feature_library.
Creates the candidate library used by the V2 model definition. The default
choice is a polynomial library, but the same builder can also construct tuned
variants when alternative settings are supplied.
"""


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


"""
Function: _build_optimizer.
Creates the sparse optimizer attached to the model. STLSQ is the current live
default, but the builder can also use tuned override values when supplied.
"""


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


"""
Function: _build_differentiation_method.
Creates the derivative estimator used by PySINDy during fitting. The current
project uses finite differences, with optional override support for tuning.
"""


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
Section 4: Public model builder.
This is the only public constructor that later stages need. It assembles the
baseline PySINDy object but does not fit, simulate, validate, or save anything.
"""


"""
Function: build_pysindy_model.
Constructs and returns an untrained V2 PySINDy model. Without overrides it
returns the canonical live default model; with overrides it can also build a
tuned candidate model for search utilities such as `hyperparameters.py`.
"""


def build_pysindy_model(settings: dict[str, Any] | None = None):
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

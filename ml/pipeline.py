"""Build sklearn Pipelines dynamically from experiment configuration."""

from typing import Any, Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelType = Literal["logistic_regression", "random_forest"]
ScalerType = Literal["none", "standard"]


def build_pipeline(
    model_type: ModelType,
    scaler_type: ScalerType = "none",
    *,
    lr_C: float = 1.0,
    rf_n_estimators: int = 100,
    rf_max_depth: int = 0,
) -> Pipeline:
    """Build a sklearn Pipeline from the given configuration.

    Args:
        model_type: One of ``'logistic_regression'`` or ``'random_forest'``.
        scaler_type: One of ``'none'`` (no scaling) or ``'standard'``
            (StandardScaler inserted before the model).
        lr_C: Inverse regularization strength for Logistic Regression.
            Smaller values mean stronger regularization.
        rf_n_estimators: Number of trees for Random Forest.
        rf_max_depth: Maximum tree depth for Random Forest.
            ``0`` or negative means unlimited depth (``None``).

    Returns:
        An unfitted sklearn ``Pipeline`` ready to call ``.fit()`` on.

    Raises:
        ValueError: If ``model_type`` is not recognised.
    """
    steps: list[tuple[str, Any]] = []

    if scaler_type == "standard":
        steps.append(("scaler", StandardScaler()))

    if model_type == "logistic_regression":
        estimator = LogisticRegression(
            C=lr_C,
            max_iter=1000,
            random_state=42,
        )
    elif model_type == "random_forest":
        depth = None if rf_max_depth <= 0 else rf_max_depth
        estimator = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=depth,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    steps.append(("model", estimator))
    return Pipeline(steps)

"""MLflow experiment tracker using a local SQLite backend."""

import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sqlite:///C:/... — three slashes then the drive letter works on Windows
TRACKING_URI = "sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix()
EXPERIMENT_NAME = "ml-playground"


def _setup() -> None:
    """Configure MLflow tracking URI and ensure the experiment exists."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def start_run(run_name: str | None = None) -> mlflow.ActiveRun:
    """Start and return a new MLflow run context manager.

    Args:
        run_name: Optional human-readable label for the run.

    Returns:
        An active MLflow run context manager (use with ``with`` statement).
    """
    _setup()
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict[str, Any]) -> None:
    """Log a dictionary of parameters to the active run.

    All values are coerced to strings so MLflow accepts them safely.

    Args:
        params: Key-value pairs of experiment parameters.
    """
    mlflow.log_params({k: str(v) for k, v in params.items()})


def log_metrics(metrics: dict[str, float]) -> None:
    """Log a dictionary of scalar metrics to the active run.

    Args:
        metrics: Key-value pairs of metric names and float values.
    """
    mlflow.log_metrics(metrics)


def log_model(model: Any, artifact_path: str = "model") -> None:
    """Log a fitted sklearn model as an MLflow artifact.

    Args:
        model: A fitted sklearn estimator or Pipeline.
        artifact_path: Sub-directory name within the run's artifact store.
    """
    mlflow.sklearn.log_model(model, artifact_path)


def log_artifact_dict(data: dict, filename: str) -> None:
    """Log a Python dict as a JSON artifact in the active run.

    Args:
        data: Serialisable dict (e.g. ROC curve arrays).
        filename: Artifact filename, e.g. ``'roc_data.json'``.
    """
    mlflow.log_dict(data, filename)


def load_roc_data(run_id: str) -> dict | None:
    """Load ROC curve data previously logged for a run.

    Args:
        run_id: MLflow run ID.

    Returns:
        Dict with ``'fpr'`` and ``'tpr'`` lists, or ``None`` if the
        artifact does not exist or cannot be read.
    """
    _setup()
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="roc_data.json",
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def delete_all_runs() -> int:
    """Delete every run in the experiment from the MLflow tracking store.

    Returns:
        Number of runs deleted.
    """
    _setup()
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return 0

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    for run in runs:
        client.delete_run(run.info.run_id)
    return len(runs)


def list_runs() -> pd.DataFrame:
    """Return a DataFrame of all runs for the experiment, newest first.

    Columns include run metadata, logged params (flat), and metrics
    prefixed with ``metric_``.

    Returns:
        DataFrame with one row per run. Empty DataFrame if no runs exist.
    """
    _setup()
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    if not runs:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for run in runs:
        record: dict[str, Any] = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name or run.info.run_id[:8],
            "start_time": pd.Timestamp(run.info.start_time, unit="ms"),
        }
        record.update(run.data.params)
        record.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
        records.append(record)

    return pd.DataFrame(records)

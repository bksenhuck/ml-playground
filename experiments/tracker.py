"""Simple MLflow wrapper utilities.

This module reads MLFLOW_TRACKING_URI and MLFLOW_ARTIFACT_URI from environment variables
and provides convenience functions used by the Dash app.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd


_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
_ARTIFACT_URI = os.environ.get("MLFLOW_ARTIFACT_URI")


def _ensure_tracking():
    """Configure mlflow tracking URI; default to a local `mlruns` folder."""
    if _TRACKING_URI:
        mlflow.set_tracking_uri(_TRACKING_URI)
    else:
        mlflow.set_tracking_uri(f"file:{os.path.abspath('mlruns')}")


def start_run(run_name: Optional[str] = None):
    _ensure_tracking()
    return mlflow.start_run(run_name=run_name)


def log_params(params: Dict) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: Dict) -> None:
    mlflow.log_metrics(metrics)


def log_model(model, artifact_path: str = "model") -> None:
    mlflow.sklearn.log_model(model, artifact_path)


def list_runs() -> pd.DataFrame:
    """Return a DataFrame with recent runs and selected metrics.

    Uses MlflowClient.search_runs to assemble a tabular view.
    """
    _ensure_tracking()
    client = MlflowClient()
    runs = client.search_runs(run_view_type=1, filter_string="", max_results=100)
    rows = []
    for r in runs:
        data = {
            "run_id": r.info.run_id,
            "name": r.data.tags.get("mlflow.runName", ""),
            "model": r.data.tags.get("model", ""),
        }
        # include a few metrics if present
        for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            data[m] = float(r.data.metrics.get(m, 0.0))
        rows.append(data)
    if not rows:
        # return empty DataFrame with expected columns so callers can rely on schema
        cols = ["run_id", "name", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    # ensure metric columns exist
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if m not in df.columns:
            df[m] = 0.0
    return df


def client():
    _ensure_tracking()
    return MlflowClient()


def run_experiment_and_log(features, scaling, model_name, params, run_name: Optional[str] = None):
    """Convenience wrapper that trains via ml.train and returns run info.

    The `run_name` is optional and will be passed through to the training run.
    To avoid circular imports, import training at call time.
    """
    from ml.train import run_training

    _ensure_tracking()
    # run_training will create and log the mlflow run; pass run_name through
    run_id, metrics, est = run_training(features, scaling, model_name, params, run_name=run_name)
    return run_id, metrics


def delete_all_runs() -> int:
    """Delete all runs from the configured MLflow tracking store.

    Returns the number of runs deleted.
    """
    _ensure_tracking()
    client = MlflowClient()
    runs = client.search_runs(run_view_type=1, filter_string="", max_results=1000)
    count = 0
    for r in runs:
        try:
            client.delete_run(r.info.run_id)
            count += 1
        except Exception:
            continue
    return count
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

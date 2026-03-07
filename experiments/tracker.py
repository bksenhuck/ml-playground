"""Per-session in-memory experiment tracking.

Each user's runs are stored in the browser via dcc.Store — no shared
database, no persistence across page refreshes.  This module is responsible
only for running training and assembling the row dict that goes into the store.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ml.train import run_training


def run_experiment_and_log(
    features: List[str],
    scaling: str,
    model_name: str,
    params: Dict,
    run_name: Optional[str] = None,
    test_size: float = 0.2,
    class_weight: str = "none",
    poly_features: bool = False,
    cv_folds: int = 0,
    dataset: str = "titanic"
) -> tuple[str, Dict[str, Any]]:
    """Train a model and return ``(run_id, row_dict)`` for the experiment table.

    The fitted pipeline is cached in ``app.plots._model_cache`` so that
    chart functions can retrieve it without re-training.
    """
    run_id, metrics, all_metrics, pipeline = run_training(
        features,
        scaling,
        model_name,
        params,
        run_name=run_name,
        test_size=test_size,
        class_weight=class_weight,
        poly_features=poly_features,
        cv_folds=cv_folds,
        dataset=dataset
    )

    # Cache fitted pipeline for chart use (process-level, cleared on restart)
    from app import plots as _plots  # noqa: PLC0415
    _plots._model_cache[run_id] = pipeline

    # Build the flat row dict that matches the DataTable columns
    row: Dict[str, Any] = {
        "run_id": run_id,
        "run_name": run_name or run_id[:8],
        "dataset": dataset,
        "model": model_name,
        "n_features": len(features) if features else 0,
        "features": ",".join(features or []),
        "test_size": test_size,
        "scaling": scaling,
        "class_weight": class_weight,
        "poly_features": poly_features,
    }
    # Merge hyperparams (C, n_estimators, max_depth, learning_rate…)
    row.update({k: v for k, v in params.items()})

    # Prefix all metrics with "metric_"
    for k, v in all_metrics.items():
        row[f"metric_{k}"] = v

    return run_id, row

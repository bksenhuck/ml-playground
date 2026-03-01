"""Training orchestration: load data, train, compute metrics and return results."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer,
)
import seaborn as sns
import warnings
import logging

from ml.pipeline import build_pipeline
from experiments import tracker

logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.*")


def _load_data() -> pd.DataFrame:
    df = sns.load_dataset("titanic")
    return df.dropna(subset=["survived"])


def _prepare_Xy(df: pd.DataFrame, features: List[str]):
    """Select features and cast target. Imputation is handled by the pipeline."""
    if not features:
        features = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
    X = df[features].copy()
    y = df["survived"].astype(int)
    return X, y


def run_training(
    features: List[str],
    scaling: str,
    model_name: str,
    hyperparams: Dict,
    run_name: str | None = None,
    test_size: float = 0.2,
    class_weight: str = "none",
    poly_features: bool = False,
    cv_folds: int = 0,
) -> Tuple[str, Dict, object]:
    """Train model, compute metrics and return (run_id, metrics, estimator)."""
    df = _load_data()
    X, y = _prepare_Xy(df, features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    pipeline = build_pipeline(
        features, scaling, model_name, hyperparams,
        class_weight=class_weight, poly_features=poly_features,
    )

    with tracker.start_run(run_name=run_name) as mlrun:
        run_id = mlrun.info.run_id
        log_p = {
            "model": model_name,
            **hyperparams,
            "features": ",".join(features),
            "test_size": test_size,
            "class_weight": class_weight,
            "poly_features": poly_features,
        }
        if cv_folds > 0:
            log_p["cv_folds"] = cv_folds
        tracker.log_params(log_p)
        pipeline.fit(X_train, y_train)

        # ── Test metrics ──────────────────────────────────────────────────────
        preds = pipeline.predict(X_test)
        probs_available = hasattr(pipeline, "predict_proba")
        probs = pipeline.predict_proba(X_test)[:, 1] if probs_available else None

        metrics = {
            "accuracy":  float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall":    float(recall_score(y_test, preds, zero_division=0)),
            "f1":        float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_test, probs)) if probs is not None else 0.0,
        }

        # ── Train metrics (overfitting detection) ─────────────────────────────
        train_preds = pipeline.predict(X_train)
        train_probs = pipeline.predict_proba(X_train)[:, 1] if probs_available else None

        train_metrics = {
            "train_accuracy":  float(accuracy_score(y_train, train_preds)),
            "train_precision": float(precision_score(y_train, train_preds, zero_division=0)),
            "train_recall":    float(recall_score(y_train, train_preds, zero_division=0)),
            "train_f1":        float(f1_score(y_train, train_preds, zero_division=0)),
            "train_roc_auc":   float(roc_auc_score(y_train, train_probs)) if train_probs is not None else 0.0,
        }

        all_metrics = {**metrics, **train_metrics}

        # ── Cross-validation metrics (optional) ───────────────────────────────
        if cv_folds > 0:
            _cv_scoring = {
                "accuracy":  "accuracy",
                "precision": make_scorer(precision_score, zero_division=0),
                "recall":    make_scorer(recall_score, zero_division=0),
                "f1":        make_scorer(f1_score, zero_division=0),
                "roc_auc":   "roc_auc",
            }
            cv_pipeline = build_pipeline(
                features, scaling, model_name, hyperparams,
                class_weight=class_weight, poly_features=poly_features,
            )
            cv_res = cross_validate(cv_pipeline, X, y, cv=cv_folds, scoring=_cv_scoring)
            for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                all_metrics[f"cv_mean_{m}"] = float(np.mean(cv_res[f"test_{m}"]))
                all_metrics[f"cv_std_{m}"]  = float(np.std(cv_res[f"test_{m}"]))

        tracker.log_metrics(all_metrics)
        try:
            tracker.log_model(pipeline)
        except Exception:
            pass

    return run_id, metrics, pipeline

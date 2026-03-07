"""Training orchestration: load data, train, compute metrics and return results."""
from __future__ import annotations

import uuid
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer,
)

from ml.pipeline import build_pipeline

warnings.filterwarnings("ignore", category=UserWarning)


def _load_data(dataset: str = "titanic") -> pd.DataFrame:
    if dataset == "housing":
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing(as_frame=True)
        return housing.frame
    df = sns.load_dataset("titanic")
    return df.dropna(subset=["survived"])


def _prepare_Xy(df: pd.DataFrame, features: List[str], dataset: str = "titanic"):
    """Select features and cast target. Imputation is handled by the pipeline."""
    if not features:
        if dataset == "housing":
            features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
        else:
            features = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
    X = df[features].copy()
    if dataset == "housing":
        y = df["MedHouseVal"]
    else:
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
    dataset: str = "titanic"
) -> Tuple[str, Dict, Dict, object]:
    """Train model, compute metrics and return (run_id, test_metrics, all_metrics, pipeline)."""
    df = _load_data(dataset)
    X, y = _prepare_Xy(df, features, dataset)
    
    task = "regression" if dataset == "housing" else "classification"
    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    pipeline = build_pipeline(
        features, scaling, model_name, hyperparams,
        class_weight=class_weight, poly_features=poly_features, task=task
    )
    pipeline.fit(X_train, y_train)

    # ── Prediction & Metrics ──────────────────────────────────────────────
    preds = pipeline.predict(X_test)
    train_preds = pipeline.predict(X_train)
    
    if task == "classification":
        probs_available = hasattr(pipeline, "predict_proba")
        probs = pipeline.predict_proba(X_test)[:, 1] if probs_available else None
        train_probs = pipeline.predict_proba(X_train)[:, 1] if probs_available else None

        metrics = {
            "accuracy":  float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall":    float(recall_score(y_test, preds, zero_division=0)),
            "f1":        float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_test, probs)) if probs is not None else 0.0,
        }
        train_metrics = {
            "train_accuracy":  float(accuracy_score(y_train, train_preds)),
            "train_precision": float(precision_score(y_train, train_preds, zero_division=0)),
            "train_recall":    float(recall_score(y_train, train_preds, zero_division=0)),
            "train_f1":        float(f1_score(y_train, train_preds, zero_division=0)),
            "train_roc_auc":   float(roc_auc_score(y_train, train_probs)) if train_probs is not None else 0.0,
        }
    else: # Regression
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds)),
        }
        train_metrics = {
            "train_mae": float(mean_absolute_error(y_train, train_preds)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
            "train_r2": float(r2_score(y_train, train_preds)),
        }

    all_metrics = {**metrics, **train_metrics}

    # ── Cross-validation metrics (optional) ───────────────────────────────
    if cv_folds > 0:
        if task == "classification":
            _cv_scoring = {
                "accuracy":  "accuracy",
                "precision": make_scorer(precision_score, zero_division=0),
                "recall":    make_scorer(recall_score, zero_division=0),
                "f1":        make_scorer(f1_score, zero_division=0),
                "roc_auc":   "roc_auc",
            }
        else: # Regression
            _cv_scoring = {
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            }

        cv_pipeline = build_pipeline(
            features, scaling, model_name, hyperparams,
            class_weight=class_weight, poly_features=poly_features, task=task
        )
        cv_res = cross_validate(cv_pipeline, X, y, cv=cv_folds, scoring=_cv_scoring)
        
        m_list = ("accuracy", "precision", "recall", "f1", "roc_auc") if task == "classification" else ("mae", "rmse", "r2")
        for m in m_list:
            vals = cv_res[f"test_{m}"]
            if task == "regression" and m in ("mae", "rmse"):
                vals = -vals # Flip negative metrics from sklearn
            all_metrics[f"cv_mean_{m}"] = float(np.mean(vals))
            all_metrics[f"cv_std_{m}"]  = float(np.std(vals))

    run_id = str(uuid.uuid4())
    return run_id, metrics, all_metrics, pipeline

"""Reusable Plotly figures for the visualization dropdown.

Each public function accepts either a list of run-row dicts (multi-run)
or a single dict, and returns a go.Figure.  Model-heavy plots load the
saved sklearn pipeline from MLflow artifacts and reconstruct the exact
same train/test split that was used during training (random_state=42).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import mlflow.sklearn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.calibration import calibration_curve as sk_calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ── MLflow setup ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRACKING_URI = "sqlite:///" + (_PROJECT_ROOT / "mlflow.db").as_posix()


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(_TRACKING_URI)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_model(run_id: str):
    """Load a fitted sklearn Pipeline from MLflow artifacts."""
    _setup_mlflow()
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def _reconstruct_test_data(run_row: dict):
    """Recreate (X_test, y_test, feature_list) using the params stored in MLflow.

    Mirrors the preprocessing in ml/train.py::_prepare_Xy so the split is
    byte-for-byte identical (random_state=42, no stratify).
    """
    features_str = str(run_row.get("features", "") or "")
    features = [f.strip() for f in features_str.split(",") if f.strip()]
    if not features:
        features = ["pclass", "sex", "age", "sibsp", "parch", "fare"]

    test_size = float(run_row.get("test_size", 0.2) or 0.2)

    df = sns.load_dataset("titanic")
    df = df.dropna(subset=["survived"])

    X = df[features].copy()
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            if "missing" not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories("missing")
            X[col] = X[col].fillna("missing")
        elif X[col].dtype.name == "object":
            X[col] = X[col].fillna("missing")
        else:
            X[col] = X[col].fillna(X[col].median())

    y = df["survived"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_test, y_test, features


def _run_label(row: dict) -> str:
    return str(row.get("run_name") or row.get("name") or str(row.get("run_id", ""))[:8])


def _empty_fig(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font={"size": 13, "color": "#6c757d"},
    )
    fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, margin={"t": 30})
    return fig


def _clean_feat_name(name: str) -> str:
    """Strip ColumnTransformer prefixes like 'num__', 'cat__' for readability."""
    for prefix in ("num__poly__", "num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_roc_curve(runs_data: List[dict]) -> go.Figure:
    """Multi-run ROC curve overlay."""
    if not runs_data:
        return _empty_fig("No runs selected")

    fig = go.Figure()
    # Random-classifier diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line={"dash": "dash", "color": "lightgray", "width": 1},
        name="Random", showlegend=False,
    ))

    for row in runs_data:
        run_id = row.get("run_id", "")
        if not run_id:
            continue
        try:
            model = _load_model(run_id)
            X_test, y_test, _ = _reconstruct_test_data(row)
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = float(row.get("metric_roc_auc") or 0)
            fig.add_trace(go.Scatter(
                x=fpr.tolist(), y=tpr.tolist(), mode="lines",
                name=f"{_run_label(row)} (AUC={auc:.3f})",
            ))
        except Exception:
            continue

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        xaxis={"range": [0, 1]}, yaxis={"range": [0, 1]},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def plot_pr_curve(runs_data: List[dict]) -> go.Figure:
    """Multi-run Precision-Recall curve overlay."""
    if not runs_data:
        return _empty_fig("No runs selected")

    fig = go.Figure()
    for row in runs_data:
        run_id = row.get("run_id", "")
        if not run_id:
            continue
        try:
            model = _load_model(run_id)
            X_test, y_test, _ = _reconstruct_test_data(row)
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            f1 = float(row.get("metric_f1") or 0)
            fig.add_trace(go.Scatter(
                x=recall.tolist(), y=precision.tolist(), mode="lines",
                name=f"{_run_label(row)} (F1={f1:.3f})",
            ))
        except Exception:
            continue

    fig.update_layout(
        title="Precision–Recall Curve",
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis={"range": [0, 1]}, yaxis={"range": [0, 1]},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def plot_confusion_matrix(runs_data: List[dict]) -> go.Figure:
    """Confusion matrix for the first selected run (normalized + raw counts)."""
    valid = [r for r in runs_data if r.get("run_id")]
    if not valid:
        return _empty_fig("No runs selected")

    row = valid[0]
    run_id = row["run_id"]
    try:
        model = _load_model(run_id)
        X_test, y_test, _ = _reconstruct_test_data(row)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        labels = ["Not Survived (0)", "Survived (1)"]
        text = [
            [f"{cm[i][j]}<br>({cm_norm[i][j]:.1%})" for j in range(2)]
            for i in range(2)
        ]

        fig = go.Figure(go.Heatmap(
            z=cm_norm, x=labels, y=labels,
            text=text, texttemplate="%{text}",
            colorscale="Blues", showscale=True,
            zmin=0, zmax=1,
        ))
        fig.update_layout(
            title=f"Confusion Matrix — {_run_label(row)}",
            xaxis_title="Predicted", yaxis_title="Actual",
        )
        return fig
    except Exception as e:
        return _empty_fig(f"Could not load model: {e}")


def plot_feature_importance(runs_data: List[dict]) -> go.Figure:
    """Feature importance (RF) or |coefficient| (LogReg) for selected runs."""
    valid = [r for r in runs_data if r.get("run_id")]
    if not valid:
        return _empty_fig("No runs selected")

    fig = go.Figure()
    any_plotted = False

    for row in valid:
        run_id = row["run_id"]
        try:
            model = _load_model(run_id)
            _, _, _ = _reconstruct_test_data(row)

            clf = model.named_steps["clf"]
            preprocessor = model.named_steps["preproc"]

            # Get feature names from the preprocessor
            try:
                raw_names = preprocessor.get_feature_names_out()
                feat_names = np.array([_clean_feat_name(n) for n in raw_names])
            except Exception:
                n_feats = (
                    len(clf.feature_importances_)
                    if hasattr(clf, "feature_importances_")
                    else len(clf.coef_[0])
                )
                feat_names = np.array([f"feat_{i}" for i in range(n_feats)])

            # Importances: RF uses feature_importances_, LogReg uses |coef_|
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                importances = np.abs(clf.coef_[0])
            else:
                continue

            top_n = min(15, len(importances))
            idx = np.argsort(importances)[::-1][:top_n]

            fig.add_trace(go.Bar(
                x=importances[idx].tolist(),
                y=feat_names[idx].tolist(),
                orientation="h",
                name=_run_label(row),
            ))
            any_plotted = True
        except Exception:
            continue

    if not any_plotted:
        return _empty_fig("Could not compute feature importance for selected runs")

    fig.update_layout(
        title="Feature Importance (top 15)",
        xaxis_title="Importance / |Coefficient|",
        yaxis={"autorange": "reversed"},
        barmode="group",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def plot_calibration_curve(runs_data: List[dict]) -> go.Figure:
    """Calibration curve: predicted probability vs actual positive rate."""
    if not runs_data:
        return _empty_fig("No runs selected")

    fig = go.Figure()
    # Perfect calibration reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line={"dash": "dash", "color": "lightgray", "width": 1},
        name="Perfect calibration", showlegend=True,
    ))

    for row in runs_data:
        run_id = row.get("run_id", "")
        if not run_id:
            continue
        try:
            model = _load_model(run_id)
            X_test, y_test, _ = _reconstruct_test_data(row)
            y_proba = model.predict_proba(X_test)[:, 1]
            frac_pos, mean_pred = sk_calibration_curve(y_test, y_proba, n_bins=10)
            fig.add_trace(go.Scatter(
                x=mean_pred.tolist(), y=frac_pos.tolist(),
                mode="lines+markers",
                name=_run_label(row),
            ))
        except Exception:
            continue

    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        xaxis={"range": [0, 1]}, yaxis={"range": [0, 1]},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def plot_metric_distribution(runs_data: List[dict]) -> go.Figure:
    """Box-plot distribution of every metric across all visible runs."""
    if not runs_data:
        return _empty_fig("No runs to display")

    df = pd.DataFrame(runs_data)
    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    if not metric_cols:
        return _empty_fig("No metric columns found")

    fig = go.Figure()
    for col in metric_cols:
        display = col.replace("metric_", "").replace("_", " ").title()
        values = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        fig.add_trace(go.Box(
            y=values, name=display,
            boxpoints="all", jitter=0.3, pointpos=-1.5,
        ))

    fig.update_layout(
        title="Metric Distribution Across Runs",
        yaxis_title="Score",
        yaxis={"range": [0, 1]},
        showlegend=False,
    )
    return fig

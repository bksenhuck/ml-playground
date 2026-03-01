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
_TRACKING_URI = (
    "sqlite:///" + (_PROJECT_ROOT / "mlflow.db").as_posix()
)


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(_TRACKING_URI)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_model(run_id: str):
    """Load a fitted sklearn Pipeline from MLflow artifacts."""
    _setup_mlflow()
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def _reconstruct_test_data(run_row: dict):
    """Recreate (X_test, y_test, feature_list) from params in MLflow.

    Must mirror train.py::run_training exactly:
    - same random_state=42, same stratify=y
    - no manual imputation (pipeline handles it)
    """
    features_str = str(run_row.get("features", "") or "")
    features = [f.strip() for f in features_str.split(",") if f.strip()]
    if not features:
        features = ["pclass", "sex", "age", "sibsp", "parch", "fare"]

    test_size = float(run_row.get("test_size", 0.2) or 0.2)

    df = sns.load_dataset("titanic")
    df = df.dropna(subset=["survived"])

    X = df[features].copy()
    y = df["survived"].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_test, y_test, features


def _run_label(row: dict) -> str:
    return str(
        row.get("run_name")
        or row.get("name")
        or str(row.get("run_id", ""))[:8]
    )


def _empty_fig(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font={"size": 13, "color": "#6c757d"},
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"t": 30},
    )
    return fig


def _clean_feat_name(name: str) -> str:
    """Strip ColumnTransformer prefixes like 'num__', 'cat__'."""
    for prefix in ("num__poly__", "num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


# ── Shared layout constants ───────────────────────────────────────────────────
_LEGEND = {
    "orientation": "v",
    "x": 1.02, "xanchor": "left",
    "y": 1,    "yanchor": "top",
}
_MARGIN = {"t": 15, "b": 40, "l": 50, "r": 150}


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_roc_curve(runs_data: List[dict]) -> go.Figure:
    """Multi-run ROC curve overlay."""
    if not runs_data:
        return _empty_fig("No runs selected")

    fig = go.Figure()
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
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        legend=_LEGEND,
        margin=_MARGIN,
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
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        legend=_LEGEND,
        margin=_MARGIN,
    )
    return fig


def plot_confusion_matrix(runs_data: List[dict]) -> go.Figure:
    """Confusion matrix for the first selected run."""
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
            xaxis_title="Predicted",
            yaxis_title="Actual",
            margin=_MARGIN,
        )
        return fig
    except Exception as e:
        return _empty_fig(f"Could not load model: {e}")


def plot_feature_importance(runs_data: List[dict]) -> go.Figure:
    """Feature importance (RF) or |coef| (LogReg) for selected runs."""
    valid = [r for r in runs_data if r.get("run_id")]
    if not valid:
        return _empty_fig("No runs selected")

    fig = go.Figure()
    any_plotted = False

    for row in valid:
        run_id = row["run_id"]
        try:
            model = _load_model(run_id)
            _reconstruct_test_data(row)

            clf = model.named_steps["clf"]
            preprocessor = model.named_steps["preproc"]

            try:
                raw_names = preprocessor.get_feature_names_out()
                feat_names = np.array(
                    [_clean_feat_name(n) for n in raw_names]
                )
            except Exception:
                n_feats = (
                    len(clf.feature_importances_)
                    if hasattr(clf, "feature_importances_")
                    else len(clf.coef_[0])
                )
                feat_names = np.array(
                    [f"feat_{i}" for i in range(n_feats)]
                )

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
        return _empty_fig(
            "Could not compute feature importance for selected runs"
        )

    fig.update_layout(
        xaxis_title="Importance / |Coefficient|",
        yaxis={"autorange": "reversed"},
        barmode="group",
        legend=_LEGEND,
        margin=_MARGIN,
    )
    return fig


def plot_calibration_curve(runs_data: List[dict]) -> go.Figure:
    """Calibration curve: predicted probability vs actual positive rate."""
    if not runs_data:
        return _empty_fig("No runs selected")

    fig = go.Figure()
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
            frac_pos, mean_pred = sk_calibration_curve(
                y_test, y_proba, n_bins=10
            )
            fig.add_trace(go.Scatter(
                x=mean_pred.tolist(), y=frac_pos.tolist(),
                mode="lines+markers",
                name=_run_label(row),
            ))
        except Exception:
            continue

    fig.update_layout(
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        legend=_LEGEND,
        margin=_MARGIN,
    )
    return fig


def plot_metric_distribution(runs_data: List[dict]) -> go.Figure:
    """Box-plot distribution of every metric across all visible runs."""
    if not runs_data:
        return _empty_fig("No runs to display")

    df = pd.DataFrame(runs_data)
    metric_cols = [
        c for c in df.columns
        if c.startswith("metric_") and "train" not in c
    ]
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
        yaxis_title="Score",
        yaxis={"range": [0, 1]},
        showlegend=False,
        margin=_MARGIN,
    )
    return fig


# ── SHAP visualizations ───────────────────────────────────────────────────────
# Cache: run_id → (shap_values, feat_names, X_t) — instant repeat calls.
_shap_cache: dict[str, tuple] = {}


def _shap_available() -> bool:
    """Return True if the shap package can be imported."""
    import importlib.util  # noqa: PLC0415
    return importlib.util.find_spec("shap") is not None


def _compute_shap_values(run_id: str, model, X_test: pd.DataFrame):
    """Compute and cache SHAP values for a single run.

    Selects the explainer automatically:
      - TreeExplainer  for tree-based models (Random Forest)
      - LinearExplainer for linear models (Logistic Regression)
    Samples at most 500 rows for performance.

    Returns:
        (shap_values, feat_names, X_t) or (None, None, None) on failure.
    """
    if run_id in _shap_cache:
        return _shap_cache[run_id]

    try:
        import shap  # noqa: PLC0415

        clf = model.named_steps["clf"]
        preprocessor = model.named_steps["preproc"]

        X_t = preprocessor.transform(X_test)

        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()

        if len(X_t) > 500:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_t), 500, replace=False)
            X_t = X_t[idx]

        try:
            raw_names = preprocessor.get_feature_names_out()
            feat_names = [_clean_feat_name(n) for n in raw_names]
        except Exception:
            feat_names = [f"feat_{i}" for i in range(X_t.shape[1])]

        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_t)
        elif hasattr(clf, "coef_"):
            explainer = shap.LinearExplainer(clf, X_t)
            sv = explainer.shap_values(X_t)
        else:
            return None, None, None

        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]

        result = (np.array(sv), feat_names, X_t)
        _shap_cache[run_id] = result
        return result

    except ImportError:
        return None, None, None
    except Exception:
        return None, None, None


def plot_shap_summary(runs_data: List[dict]) -> go.Figure:
    """SHAP beeswarm summary.

    Each dot = one test sample.
    X axis : SHAP value (positive = toward survived=1).
    Y axis : feature (top 15 by mean |SHAP|, most important at top).
    Colour : normalised feature value — red = high, blue = low.
    Uses only the first selected run.
    """
    if not _shap_available():
        return _empty_fig("SHAP não instalado. Execute: pip install shap")

    valid = [r for r in runs_data if r.get("run_id")]
    if not valid:
        return _empty_fig("No runs selected")

    row = valid[0]
    run_id = row["run_id"]
    try:
        model = _load_model(run_id)
        X_test, _, _ = _reconstruct_test_data(row)
        sv, feat_names, X_t = _compute_shap_values(run_id, model, X_test)

        if sv is None:
            return _empty_fig("SHAP não disponível para este modelo")

        mean_abs = np.abs(sv).mean(axis=0)
        top_n = min(15, len(mean_abs))
        top_idx = np.argsort(mean_abs)[::-1][:top_n][::-1]

        n_samples = sv.shape[0]
        rng = np.random.RandomState(42)

        fig = go.Figure()
        show_colorbar = True

        for rank, feat_i in enumerate(top_idx):
            shap_vals = sv[:, feat_i]
            feat_vals = X_t[:, feat_i]

            f_min, f_max = feat_vals.min(), feat_vals.max()
            feat_norm = (
                (feat_vals - f_min) / (f_max - f_min)
                if f_max > f_min
                else np.full_like(feat_vals, 0.5)
            )

            y_pos = rank + rng.uniform(-0.3, 0.3, size=n_samples)

            fig.add_trace(go.Scatter(
                x=shap_vals.tolist(),
                y=y_pos.tolist(),
                mode="markers",
                name=feat_names[feat_i],
                showlegend=False,
                marker={
                    "size": 5,
                    "opacity": 0.65,
                    "color": feat_norm.tolist(),
                    "colorscale": "RdBu_r",
                    "cmin": 0,
                    "cmax": 1,
                    "showscale": show_colorbar,
                    "colorbar": {
                        "title": "Valor da<br>feature",
                        "tickvals": [0, 1],
                        "ticktext": ["baixo", "alto"],
                        "len": 0.5,
                        "y": 0.5,
                    },
                },
                hovertemplate=(
                    f"<b>{feat_names[feat_i]}</b><br>"
                    "SHAP: %{x:.3f}<br>"
                    "Feature (norm): %{marker.color:.2f}"
                    "<extra></extra>"
                ),
            ))
            show_colorbar = False

        feat_labels = [feat_names[i] for i in top_idx]

        fig.update_layout(
            xaxis={
                "title": "SHAP  ← survived=0  |  survived=1 →",
                "zeroline": True,
                "zerolinecolor": "lightgray",
                "zerolinewidth": 1,
            },
            yaxis={
                "tickmode": "array",
                "tickvals": list(range(len(top_idx))),
                "ticktext": feat_labels,
                "showgrid": False,
            },
            showlegend=False,
            margin={"l": 130, "r": 20, "t": 15, "b": 40},
        )
        return fig

    except Exception as exc:
        return _empty_fig(f"Erro ao calcular SHAP: {exc}")


def plot_shap_dependence(
    runs_data: List[dict],
    feature_name: str | None,
) -> go.Figure:
    """SHAP dependence plot for one feature on the first selected run.

    X axis: transformed feature value.
    Y axis: SHAP contribution (positive = towards class 1 / survived).
    """
    if not _shap_available():
        return _empty_fig("SHAP não instalado. Execute: pip install shap")

    if not feature_name:
        return _empty_fig("Selecione uma feature no seletor acima")

    valid = [r for r in runs_data if r.get("run_id")]
    if not valid:
        return _empty_fig("No runs selected")

    row = valid[0]
    run_id = row["run_id"]
    try:
        model = _load_model(run_id)
        X_test, _, _ = _reconstruct_test_data(row)
        sv, feat_names, X_t = _compute_shap_values(run_id, model, X_test)

        if sv is None:
            return _empty_fig("SHAP não disponível para este modelo")

        if feature_name in feat_names:
            feat_idx = feat_names.index(feature_name)
        else:
            matched = [
                i for i, fn in enumerate(feat_names)
                if fn.startswith(feature_name)
            ]
            if not matched:
                return _empty_fig(
                    f"Feature '{feature_name}' não encontrada"
                )
            feat_idx = matched[0]

        shap_col = sv[:, feat_idx]
        feat_col = X_t[:, feat_idx]
        col_name = feat_names[feat_idx]

        fig = go.Figure(go.Scatter(
            x=feat_col.tolist(),
            y=shap_col.tolist(),
            mode="markers",
            marker={
                "size": 6,
                "opacity": 0.7,
                "color": shap_col.tolist(),
                "colorscale": "RdBu_r",
                "showscale": True,
                "colorbar": {"title": "SHAP"},
            },
            hovertemplate=(
                f"<b>{col_name}</b>: %{{x:.3f}}<br>"
                "SHAP: %{y:.3f}<extra></extra>"
            ),
        ))

        fig.update_layout(
            xaxis_title=f"{col_name} (valor transformado)",
            yaxis_title="SHAP Value",
            margin=_MARGIN,
        )
        return fig

    except Exception as exc:
        return _empty_fig(f"Erro ao calcular SHAP: {exc}")

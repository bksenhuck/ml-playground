"""Reusable Plotly figures for the visualization dropdown.

Each public function accepts either a list of run-row dicts (multi-run)
or a single dict, and returns a go.Figure.

Models are loaded from ``_model_cache`` (populated when the user clicks
"Run Experiment").  If a run_id is not in the cache (e.g. after a server
restart) the pipeline is rebuilt and refitted from the stored params.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

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

# ── In-memory model cache ─────────────────────────────────────────────────────
# Populated by experiments/tracker.py immediately after each training run.
# Keyed by run_id (UUID string).  Cleared on process restart (acceptable for
# Cloud Run — store data in the browser is also lost on refresh).
_model_cache: Dict[str, Any] = {}


def _get_model(run_row: dict):
    """Return the fitted pipeline for a run, rebuilding from params if needed."""
    run_id = run_row.get("run_id", "")
    if run_id and run_id in _model_cache:
        return _model_cache[run_id]

    # Fallback: rebuild from stored params (covers server-restart edge case)
    from ml.pipeline import build_pipeline  # noqa: PLC0415

    features_str = str(run_row.get("features", "") or "")
    features = [f.strip() for f in features_str.split(",") if f.strip()]
    
    test_size = float(run_row.get("test_size", 0.2) or 0.2)
    model_name = str(run_row.get("model", "logreg"))
    scaling = str(run_row.get("scaling", "standard"))
    class_weight = str(run_row.get("class_weight", "none"))
    poly_features = bool(run_row.get("poly_features", False))
    dataset = str(run_row.get("dataset", "titanic"))

    if not features:
        if dataset == "housing":
            features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
        else:
            features = ["pclass", "sex", "age", "sibsp", "parch", "fare"]

    params: dict = {}
    c = run_row.get("C")
    if c is not None:
        params["C"] = float(c)
    alpha = run_row.get("alpha")
    if alpha is not None:
        params["alpha"] = float(alpha)
    n = run_row.get("n_estimators")
    if n is not None:
        params["n_estimators"] = int(n)
    d = run_row.get("max_depth")
    if d is not None:
        params["max_depth"] = int(d)
    lr = run_row.get("learning_rate")
    if lr is not None:
        params["learning_rate"] = float(lr)

    if dataset == "housing":
        from sklearn.datasets import fetch_california_housing
        df = fetch_california_housing(as_frame=True).frame
        X = df[features].copy()
        y = df["MedHouseVal"]
        stratify = None
    else:
        df = sns.load_dataset("titanic").dropna(subset=["survived"])
        X = df[features].copy()
        y = df["survived"].astype(int)
        stratify = y

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    task = "regression" if dataset == "housing" else "classification"
    pipeline = build_pipeline(
        features, scaling, model_name, params,
        class_weight=class_weight, poly_features=poly_features, task=task
    )
    pipeline.fit(X_train, y_train)

    if run_id:
        _model_cache[run_id] = pipeline
    return pipeline


# ── Shared helpers ────────────────────────────────────────────────────────────

_DEFAULT_FEATURES: dict[str, list[str]] = {
    "housing": ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup", "Latitude", "Longitude"],
    "titanic": ["pclass", "sex", "age", "sibsp", "parch", "fare"],
}


def _load_dataset_for_row(run_row: dict):
    """Return (X, y, features, stratify) from a run-row's stored params."""
    features_str = str(run_row.get("features", "") or "")
    features = [f.strip() for f in features_str.split(",") if f.strip()]
    dataset = str(run_row.get("dataset", "titanic"))

    if not features:
        features = _DEFAULT_FEATURES.get(dataset, _DEFAULT_FEATURES["titanic"])

    if dataset == "housing":
        from sklearn.datasets import fetch_california_housing  # noqa: PLC0415
        df = fetch_california_housing(as_frame=True).frame
        X, y, stratify = df[features].copy(), df["MedHouseVal"], None
    else:
        df = sns.load_dataset("titanic").dropna(subset=["survived"])
        X = df[features].copy()
        y = df["survived"].astype(int)
        stratify = y

    return X, y, features, stratify


def _reconstruct_split(run_row: dict, split: str = "test"):
    """Return (X, y, features) for the requested split ('test' or 'train').

    Mirrors train.py: random_state=42, same stratify, no manual imputation.
    """
    X, y, features, stratify = _load_dataset_for_row(run_row)
    test_size = float(run_row.get("test_size", 0.2) or 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
    if split == "train":
        return X_train, y_train, features
    return X_test, y_test, features


def _reconstruct_test_data(run_row: dict):
    return _reconstruct_split(run_row, "test")


def _reconstruct_train_data(run_row: dict):
    return _reconstruct_split(run_row, "train")


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
        **_CLEAN,
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
_CLEAN = {"paper_bgcolor": "white", "plot_bgcolor": "white"}
_NO_GRID = {"showgrid": False, "zeroline": False}


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_roc_curve(
    runs_data: List[dict], data_split: str = "test"
) -> go.Figure:
    """Multi-run ROC curve overlay. data_split: 'test' | 'train' | 'both'."""
    if not runs_data:
        return _empty_fig("No runs selected")

    from app.plot_config import COLOR_REF, PALETTE  # noqa: PLC0415
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line={"dash": "dash", "color": COLOR_REF, "width": 1},
        name="Random", showlegend=False,
    ))

    for i, row in enumerate(runs_data):
        if not row.get("run_id"):
            continue
        try:
            model = _get_model(row)
            label = _run_label(row)
            color = PALETTE[i % len(PALETTE)]

            if data_split in ("test", "both"):
                X_t, y_t, _ = _reconstruct_test_data(row)
                fpr, tpr, _ = roc_curve(y_t, model.predict_proba(X_t)[:, 1])
                auc = float(row.get("metric_roc_auc") or 0)
                suffix = " (Teste)" if data_split == "both" else ""
                fig.add_trace(go.Scatter(
                    x=fpr.tolist(), y=tpr.tolist(), mode="lines",
                    name=f"{label}{suffix} (AUC={auc:.3f})",
                    line={"color": color},
                ))

            if data_split in ("train", "both"):
                X_tr, y_tr, _ = _reconstruct_train_data(row)
                fpr, tpr, _ = roc_curve(y_tr, model.predict_proba(X_tr)[:, 1])
                auc = float(row.get("metric_train_roc_auc") or 0)
                suffix = " (Treino)" if data_split == "both" else ""
                fig.add_trace(go.Scatter(
                    x=fpr.tolist(), y=tpr.tolist(), mode="lines",
                    name=f"{label}{suffix} (AUC={auc:.3f})",
                    line={"dash": "dash", "color": color},
                ))
        except Exception:
            continue

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis={"range": [0, 1], **_NO_GRID},
        yaxis={"range": [0, 1], **_NO_GRID},
        legend=_LEGEND,
        margin=_MARGIN,
        **_CLEAN,
    )
    return fig


def plot_pr_curve(
    runs_data: List[dict], data_split: str = "test"
) -> go.Figure:
    """Multi-run Precision-Recall curve overlay. data_split: 'test'|'train'|'both'."""
    if not runs_data:
        return _empty_fig("No runs selected")

    from app.plot_config import PALETTE  # noqa: PLC0415
    fig = go.Figure()
    for i, row in enumerate(runs_data):
        if not row.get("run_id"):
            continue
        try:
            model = _get_model(row)
            label = _run_label(row)
            color = PALETTE[i % len(PALETTE)]

            if data_split in ("test", "both"):
                X_t, y_t, _ = _reconstruct_test_data(row)
                prec, rec, _ = precision_recall_curve(
                    y_t, model.predict_proba(X_t)[:, 1]
                )
                f1 = float(row.get("metric_f1") or 0)
                suffix = " (Teste)" if data_split == "both" else ""
                fig.add_trace(go.Scatter(
                    x=rec.tolist(), y=prec.tolist(), mode="lines",
                    name=f"{label}{suffix} (F1={f1:.3f})",
                    line={"color": color},
                ))

            if data_split in ("train", "both"):
                X_tr, y_tr, _ = _reconstruct_train_data(row)
                prec, rec, _ = precision_recall_curve(
                    y_tr, model.predict_proba(X_tr)[:, 1]
                )
                f1 = float(row.get("metric_train_f1") or 0)
                suffix = " (Treino)" if data_split == "both" else ""
                fig.add_trace(go.Scatter(
                    x=rec.tolist(), y=prec.tolist(), mode="lines",
                    name=f"{label}{suffix} (F1={f1:.3f})",
                    line={"dash": "dash", "color": color},
                ))
        except Exception:
            continue

    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis={"range": [0, 1], **_NO_GRID},
        yaxis={"range": [0, 1], **_NO_GRID},
        legend=_LEGEND,
        margin=_MARGIN,
        **_CLEAN,
    )
    return fig


def plot_confusion_matrix(
    runs_data: List[dict], data_split: str = "test"
) -> go.Figure:
    """Confusion matrix for the first selected run. data_split: 'test'|'train'|'both'."""
    valid = [r for r in runs_data if r.get("run_id")]
    if not valid:
        return _empty_fig("No runs selected")

    row = valid[0]
    # "both" falls back to test with an explanatory note
    effective_split = "test" if data_split == "both" else data_split
    try:
        model = _get_model(row)
        if effective_split == "train":
            X_data, y_data, _ = _reconstruct_train_data(row)
        else:
            X_data, y_data, _ = _reconstruct_test_data(row)

        y_pred = model.predict(X_data)
        cm = confusion_matrix(y_data, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        labels = ["Not Survived (0)", "Survived (1)"]
        text = [
            [f"{cm[i][j]}<br>({cm_norm[i][j]:.1%})" for j in range(2)]
            for i in range(2)
        ]

        split_label = "Treino" if effective_split == "train" else "Teste"
        fig = go.Figure(go.Heatmap(
            z=cm_norm, x=labels, y=labels,
            text=text, texttemplate="%{text}",
            colorscale="Blues", showscale=True,
            zmin=0, zmax=1,
        ))
        fig.update_layout(
            xaxis_title=f"Predicted ({split_label})",
            yaxis_title="Actual",
            margin=_MARGIN,
            **_CLEAN,
        )
        if data_split == "both":
            fig.add_annotation(
                text="Ambos: exibindo apenas dados de Teste",
                x=0.5, y=-0.18,
                xref="paper", yref="paper",
                showarrow=False,
                font={"size": 11, "color": "#adb5bd"},
                align="center",
            )
        return fig
    except Exception as e:
        return _empty_fig(f"Could not load model: {e}")


def plot_feature_importance(runs_data: List[dict]) -> go.Figure:
    """Feature importance (RF) or |coef| (LogReg/Ridge) for selected runs."""
    valid = [r for r in runs_data if r.get("run_id")]
    if not valid:
        return _empty_fig("No runs selected")

    from app.plot_config import PALETTE  # noqa: PLC0415
    fig = go.Figure()
    any_plotted = False

    for i, row in enumerate(valid):
        try:
            model = _get_model(row)
            clf = model.named_steps["clf"]
            preprocessor = model.named_steps["preproc"]

            try:
                raw_names = preprocessor.get_feature_names_out()
                feat_names = np.array(
                    [_clean_feat_name(n) for n in raw_names]
                )
            except Exception:
                n_feats = 0
                if hasattr(clf, "feature_importances_"):
                    n_feats = len(clf.feature_importances_)
                elif hasattr(clf, "coef_"):
                    n_feats = len(clf.coef_.flatten())
                feat_names = np.array([f"feat_{i}" for i in range(n_feats)])

            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                importances = np.abs(clf.coef_.flatten())
            else:
                continue

            top_n = min(15, len(importances))
            idx = np.argsort(importances)[::-1][:top_n]

            fig.add_trace(go.Bar(
                x=importances[idx].tolist(),
                y=feat_names[idx].tolist(),
                orientation="h",
                name=_run_label(row),
                marker_color=PALETTE[i % len(PALETTE)],
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
        xaxis=_NO_GRID,
        yaxis={"autorange": "reversed", **_NO_GRID},
        barmode="group",
        legend=_LEGEND,
        margin=_MARGIN,
        **_CLEAN,
    )
    return fig


def plot_calibration_curve(
    runs_data: List[dict], data_split: str = "test"
) -> go.Figure:
    """Calibration curve. data_split: 'test' | 'train' | 'both'."""
    if not runs_data:
        return _empty_fig("No runs selected")

    from app.plot_config import COLOR_REF, PALETTE  # noqa: PLC0415
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line={"dash": "dash", "color": COLOR_REF, "width": 1},
        name="Perfect calibration", showlegend=True,
    ))

    for i, row in enumerate(runs_data):
        if not row.get("run_id"):
            continue
        try:
            model = _get_model(row)
            label = _run_label(row)
            color = PALETTE[i % len(PALETTE)]

            if data_split in ("test", "both"):
                X_t, y_t, _ = _reconstruct_test_data(row)
                frac, pred = sk_calibration_curve(
                    y_t, model.predict_proba(X_t)[:, 1], n_bins=10
                )
                suffix = " (Teste)" if data_split == "both" else ""
                fig.add_trace(go.Scatter(
                    x=pred.tolist(), y=frac.tolist(),
                    mode="lines+markers",
                    name=f"{label}{suffix}",
                    line={"color": color},
                ))

            if data_split in ("train", "both"):
                X_tr, y_tr, _ = _reconstruct_train_data(row)
                frac, pred = sk_calibration_curve(
                    y_tr, model.predict_proba(X_tr)[:, 1], n_bins=10
                )
                suffix = " (Treino)" if data_split == "both" else ""
                fig.add_trace(go.Scatter(
                    x=pred.tolist(), y=frac.tolist(),
                    mode="lines+markers",
                    name=f"{label}{suffix}",
                    line={"dash": "dash", "color": color},
                ))
        except Exception:
            continue

    fig.update_layout(
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        xaxis={"range": [0, 1], **_NO_GRID},
        yaxis={"range": [0, 1], **_NO_GRID},
        legend=_LEGEND,
        margin=_MARGIN,
        **_CLEAN,
    )
    return fig


def plot_metric_distribution(
    runs_data: List[dict], data_split: str = "test"
) -> go.Figure:
    """Grouped bar chart of metrics broken by experiment."""
    if not runs_data:
        return _empty_fig("No runs to display")

    df = pd.DataFrame(runs_data)
    if data_split == "train":
        metric_cols = [
            c for c in df.columns if c.startswith("metric_train_")
        ]
    elif data_split == "both":
        metric_cols = [c for c in df.columns if c.startswith("metric_")]
    else:  # "test"
        metric_cols = [
            c for c in df.columns
            if c.startswith("metric_") and "train" not in c
        ]
    if not metric_cols:
        return _empty_fig("No metric columns found")

    from app.plot_config import PALETTE  # noqa: PLC0415
    display_cols = [
        col.replace("metric_train_", "Train ")
           .replace("metric_", "")
           .replace("_", " ")
           .title()
        for col in metric_cols
    ]

    fig = go.Figure()
    for i, row in enumerate(runs_data):
        vals = [float(row.get(c) or 0) for c in metric_cols]
        fig.add_trace(go.Bar(
            name=_run_label(row),
            x=display_cols,
            y=vals,
            marker_color=PALETTE[i % len(PALETTE)],
        ))

    fig.update_layout(
        yaxis_title="Score",
        barmode="group",
        xaxis=_NO_GRID,
        yaxis={"range": [0, 1], **_NO_GRID},
        showlegend=True,
        legend=_LEGEND,
        margin=_MARGIN,
        **_CLEAN,
    )
    return fig


# ── SHAP visualizations ───────────────────────────────────────────────────────
# Cache: run_id → (shap_values, feat_names, X_t) — instant repeat calls.
_shap_cache: dict[str, tuple] = {}
_shap_cache_train: dict[str, tuple] = {}


def _shap_available() -> bool:
    """Return True if the shap package can be imported."""
    import importlib.util  # noqa: PLC0415
    return importlib.util.find_spec("shap") is not None


def _compute_shap_values(
    run_id: str,
    model,
    X_data: pd.DataFrame,
    cache: "dict[str, tuple] | None" = None,
):
    """Compute and cache SHAP values for a single run.

    Selects the explainer automatically:
      - TreeExplainer  for tree-based models (Random Forest, XGBoost)
      - LinearExplainer for linear models (Logistic Regression)
    Samples at most 500 rows for performance.

    Pass ``cache=_shap_cache`` for test data,
    ``cache=_shap_cache_train`` for train data.
    """
    _cache = _shap_cache if cache is None else cache
    if run_id in _cache:
        return _cache[run_id]

    try:
        import shap  # noqa: PLC0415

        clf = model.named_steps["clf"]
        preprocessor = model.named_steps["preproc"]

        X_t = preprocessor.transform(X_data)

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
        _cache[run_id] = result
        return result

    except ImportError:
        return None, None, None
    except Exception:
        return None, None, None


def plot_shap_summary(
    runs_data: List[dict], data_split: str = "test"
) -> go.Figure:
    """SHAP beeswarm summary (first selected run only). data_split: 'test'|'train'|'both'."""
    if not _shap_available():
        return _empty_fig("SHAP não disponível")

    valid = [r for r in runs_data if r.get("run_id")]
    if not valid:
        return _empty_fig("No runs selected")

    row = valid[0]
    run_id = row["run_id"]
    try:
        model = _get_model(row)
        # "both" uses test data with a note annotation
        if data_split == "train":
            X_data, _, _ = _reconstruct_train_data(row)
            shap_cache = _shap_cache_train
        else:
            X_data, _, _ = _reconstruct_test_data(row)
            shap_cache = _shap_cache
        sv, feat_names, X_t = _compute_shap_values(
            run_id, model, X_data, cache=shap_cache
        )

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
                "showgrid": False,
                "zeroline": True,
                "zerolinecolor": "#cccccc",
                "zerolinewidth": 1,
            },
            yaxis={
                "tickmode": "array",
                "tickvals": list(range(len(top_idx))),
                "ticktext": feat_labels,
                "showgrid": False,
                "zeroline": False,
            },
            showlegend=False,
            margin={"l": 130, "r": 20, "t": 15, "b": 40},
            **_CLEAN,
        )
        if data_split == "both":
            fig.add_annotation(
                text="Ambos: SHAP exibe apenas dados de Teste",
                x=0.5, y=-0.12,
                xref="paper", yref="paper",
                showarrow=False,
                font={"size": 11, "color": "#adb5bd"},
                align="center",
            )
        return fig

    except Exception as exc:
        return _empty_fig(f"Erro ao calcular SHAP: {exc}")


def plot_shap_dependence(
    runs_data: List[dict],
    feature_name: Optional[str],
    data_split: str = "test",
) -> go.Figure:
    """SHAP dependence plot for one feature. data_split: 'test'|'train'|'both'."""
    if not _shap_available():
        return _empty_fig("SHAP não disponível")

    if not feature_name:
        return _empty_fig("Selecione uma feature no seletor acima")

    valid = [r for r in runs_data if r.get("run_id")]
    if not valid:
        return _empty_fig("No runs selected")

    row = valid[0]
    run_id = row["run_id"]
    try:
        model = _get_model(row)
        # "both" uses test data with a note annotation
        if data_split == "train":
            X_data, _, _ = _reconstruct_train_data(row)
            shap_cache = _shap_cache_train
        else:
            X_data, _, _ = _reconstruct_test_data(row)
            shap_cache = _shap_cache
        sv, feat_names, X_t = _compute_shap_values(
            run_id, model, X_data, cache=shap_cache
        )

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
            xaxis=_NO_GRID,
            yaxis=_NO_GRID,
            margin=_MARGIN,
            **_CLEAN,
        )
        if data_split == "both":
            fig.add_annotation(
                text="Ambos: SHAP exibe apenas dados de Teste",
                x=0.5, y=-0.15,
                xref="paper", yref="paper",
                showarrow=False,
                font={"size": 11, "color": "#adb5bd"},
                align="center",
            )
        return fig

    except Exception as exc:
        return _empty_fig(f"Erro ao calcular SHAP: {exc}")


def plot_prediction_error(runs_data: List[dict], data_split: str = "test") -> go.Figure:
    """Regression only: Predicted vs Actual scatter plot."""
    if not runs_data:
        return _empty_fig("No runs selected")

    from app.plot_config import PALETTE  # noqa: PLC0415
    fig = go.Figure()

    all_y = []
    for i, row in enumerate(runs_data):
        try:
            model = _get_model(row)
            label = _run_label(row)
            color = PALETTE[i % len(PALETTE)]
            
            if data_split == "train":
                X_d, y_d, _ = _reconstruct_train_data(row)
            else:
                X_d, y_d, _ = _reconstruct_test_data(row)
                
            y_pred = model.predict(X_d)
            all_y.extend(y_d.tolist())
            all_y.extend(y_pred.tolist())

            fig.add_trace(go.Scatter(
                x=y_d, y=y_pred,
                mode="markers",
                name=label,
                marker={"color": color, "opacity": 0.5, "size": 5},
                hovertemplate="Real: %{x}<br>Pred: %{y}<extra></extra>"
            ))
        except Exception:
            continue

    if not all_y:
        return _empty_fig("Erro ao gerar gráfico de predição")

    # Identity line (45 degrees)
    mn, mx = min(all_y), max(all_y)
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line={"color": "black", "dash": "dash", "width": 1},
        name="Ideal (45°)",
        showlegend=True
    ))

    fig.update_layout(
        xaxis_title="Valor Real (Target)",
        yaxis_title="Valor Predito",
        xaxis=_NO_GRID, yaxis=_NO_GRID,
        margin=_MARGIN,
        **_CLEAN,
    )
    return fig


# ── Shared metric config ──────────────────────────────────────────────────────

def _get_metric_config(runs_data: List[dict]) -> dict:
    """Return metric column names and labels for the given runs.

    Returns a dict with keys:
      is_regression, m_cols, m_lbls,
      test_metrics, test_labels, train_metrics, train_labels
    """
    is_regression = any(r.get("dataset") == "housing" for r in runs_data)
    if is_regression:
        m_cols = ["mae", "rmse", "r2"]
        m_lbls = ["MAE", "RMSE", "R2"]
    else:
        m_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        m_lbls = ["Accuracy", "Precision", "Recall", "F1", "Roc Auc"]
    return {
        "is_regression":  is_regression,
        "m_cols":         m_cols,
        "m_lbls":         m_lbls,
        "test_metrics":   [f"metric_{m}" for m in m_cols],
        "test_labels":    m_lbls,
        "train_metrics":  [f"metric_train_{m}" for m in m_cols],
        "train_labels":   [f"Train {l}" for l in m_lbls],
    }


# ── Bar-F1 chart ──────────────────────────────────────────────────────────────

def plot_bar_f1(runs_data: List[dict], data_split: str = "test") -> go.Figure:
    """Grouped bar chart of F1 score per run. data_split: 'test'|'train'|'both'."""
    from app.plot_config import PALETTE  # noqa: PLC0415

    df = pd.DataFrame(runs_data)
    sel_df = df  # runs_data is already the selected subset

    _layout = dict(
        barmode="group",
        yaxis_title="F1 Score",
        legend={"orientation": "v", "x": 1.02, "xanchor": "left", "y": 1, "yanchor": "top"},
        margin={"t": 15, "b": 40, "l": 50, "r": 150},
        **_CLEAN,
        xaxis=_NO_GRID,
        yaxis=_NO_GRID,
    )

    fig = go.Figure()

    if data_split == "both":
        for i, row in enumerate(runs_data):
            run_lbl = _run_label(row)
            color = PALETTE[i % len(PALETTE)]
            if "metric_f1" in df.columns:
                fig.add_trace(go.Bar(
                    name=f"{run_lbl} (Teste)",
                    x=[str(run_lbl)],
                    y=[float(row.get("metric_f1", 0) or 0)],
                    marker={"color": color},
                ))
            if "metric_train_f1" in df.columns:
                fig.add_trace(go.Bar(
                    name=f"{run_lbl} (Treino)",
                    x=[str(run_lbl)],
                    y=[float(row.get("metric_train_f1", 0) or 0)],
                    marker={"color": color, "opacity": 0.5, "pattern": {"shape": "/"}},
                ))
        fig.update_layout(**_layout)
        return fig

    y_col = "metric_train_f1" if data_split == "train" else "metric_f1"
    if y_col not in df.columns:
        y_col = "metric_f1" if "metric_f1" in df.columns else "f1"
    if y_col not in df.columns:
        df[y_col] = 0.0

    run_color = {
        str(r.get("run_name") or r.get("run_id") or "Run"): PALETTE[j % len(PALETTE)]
        for j, r in enumerate(runs_data)
    }
    sorted_runs = (
        sorted(runs_data, key=lambda r: float(r.get(y_col, 0) or 0), reverse=True)
        if y_col in df.columns else runs_data
    )
    for row in sorted_runs:
        lbl = str(row.get("run_name") or row.get("run_id") or "Run")
        fig.add_trace(go.Bar(
            name=lbl, x=[lbl],
            y=[float(row.get(y_col, 0) or 0)],
            marker={"color": run_color.get(lbl, PALETTE[0])},
        ))

    fig.update_layout(**_layout)
    return fig


# ── Radar chart ───────────────────────────────────────────────────────────────

def plot_radar(runs_data: List[dict], data_split: str = "test") -> go.Figure:
    """Multi-run radar/spider chart. data_split: 'test'|'train'|'both'."""
    from app.plot_config import PALETTE, hex_rgba  # noqa: PLC0415

    cfg = _get_metric_config(runs_data)
    df = pd.DataFrame(runs_data)
    fig = go.Figure()

    radar_range = None if cfg["is_regression"] else [0, 1]

    if data_split == "both":
        avail_test  = [m for m in cfg["test_metrics"]  if m in df.columns]
        avail_train = [m for m in cfg["train_metrics"] if m in df.columns]
        disp_test  = [cfg["test_labels"] [cfg["test_metrics"] .index(m)] for m in avail_test]
        disp_train = [cfg["train_labels"][cfg["train_metrics"].index(m)] for m in avail_train]

        for i, row in enumerate(runs_data):
            run_lbl = _run_label(row)
            color = PALETTE[i % len(PALETTE)]
            if avail_test:
                vals = [float(row.get(m, 0) or 0) for m in avail_test]
                fig.add_scatterpolar(
                    r=vals + [vals[0]], theta=disp_test + [disp_test[0]],
                    name=f"{run_lbl} (Teste)", fill="toself",
                    line={"color": color}, fillcolor=hex_rgba(color, 0.2),
                )
            if avail_train:
                vals = [float(row.get(m, 0) or 0) for m in avail_train]
                fig.add_scatterpolar(
                    r=vals + [vals[0]], theta=disp_train + [disp_train[0]],
                    name=f"{run_lbl} (Treino)", fill="toself",
                    line={"dash": "dot", "color": color},
                    fillcolor=hex_rgba(color, 0.1),
                )
    else:
        internal = cfg["train_metrics"] if data_split == "train" else cfg["test_metrics"]
        display  = cfg["train_labels"]  if data_split == "train" else cfg["test_labels"]
        avail    = [m for m in internal if m in df.columns]
        avail_d  = [display[internal.index(m)] for m in avail]
        if not avail:
            return fig
        for i, row in enumerate(runs_data):
            vals = [float(row.get(m, 0) or 0) for m in avail]
            color = PALETTE[i % len(PALETTE)]
            fig.add_scatterpolar(
                r=vals + [vals[0]], theta=avail_d + [avail_d[0]],
                name=str(_run_label(row)), fill="toself",
                line={"color": color}, fillcolor=hex_rgba(color),
            )

    fig.update_layout(
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                visible=bool(cfg["is_regression"]),
                range=radar_range,
                showgrid=False,
            ),
            angularaxis=dict(showgrid=False),
        ),
        showlegend=True,
        legend={"orientation": "v", "x": 1.02, "xanchor": "left", "y": 1, "yanchor": "top"},
        margin={"t": 15, "b": 20, "l": 20, "r": 150},
        paper_bgcolor="white",
    )
    return fig

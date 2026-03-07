"""All Dash callbacks for the ML Playground app.

Register with: register_callbacks(app)
"""
from __future__ import annotations

import logging
from typing import List

import dash
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback_context, dcc, html, no_update

import dash_bootstrap_components as dbc

from app.data_config import (
    HOUSING_DEFAULT, HOUSING_OPTS,
    TITANIC_DEFAULT, TITANIC_OPTS, TITANIC_REDUNDANT,
)
from app.plot_config import PALETTE, hex_rgba
from app.plot_help import PLOT_DESCRIPTIONS
from experiments.tracker import run_experiment_and_log

logger = logging.getLogger(__name__)

# ── Constants used by the runs-table callback ─────────────────────────────────
_CLS_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
_REG_METRICS = ["mae", "rmse", "r2"]
_INFO_COLS   = ["run_name", "model", "n_features", "scaling"]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _extract_val(children, target_id, default=None):
    """Recursively find the value of a component by its id in a children tree."""
    if not children:
        return default
    items = children if isinstance(children, list) else [children]
    for it in items:
        try:
            props = it.get("props", {})
            if props.get("id") == target_id:
                return props.get("value", default)
            nested = props.get("children")
            if nested:
                v = _extract_val(nested, target_id, None)
                if v is not None:
                    return v
        except Exception:
            continue
    return default


def _col_def(c: str) -> dict:
    """Build a DataTable column definition with a human-readable name."""
    if c.startswith("metric_cv_mean_"):
        name = "CV μ " + c.replace("metric_cv_mean_", "").replace("_", " ").title()
    elif c.startswith("metric_cv_std_"):
        name = "CV σ " + c.replace("metric_cv_std_", "").replace("_", " ").title()
    elif c.startswith("metric_train_"):
        name = "Train " + c.replace("metric_train_", "").replace("_", " ").title()
    elif c.startswith("metric_"):
        name = c.replace("metric_", "").replace("_", " ").title()
    elif c.startswith("delta_"):
        name = "Δ " + c.replace("delta_", "").replace("_", " ").title()
    else:
        name = c.replace("_", " ").title()
    is_num = "metric_" in c or "delta_" in c
    return {
        "name": name, "id": c,
        "type": "numeric" if is_num else "text",
        "format": {"specifier": ".3f"} if is_num else None,
    }


# ── Callback registration ─────────────────────────────────────────────────────

def register_callbacks(app: dash.Dash) -> None:

    # ── Page routing ──────────────────────────────────────────────────────────
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def render_page_content(pathname: str):
        from app.pages.welcome     import welcome_layout
        from app.pages.experiments import experiments_layout
        from app.pages.datasets    import datasets_layout
        from app.pages.models      import models_layout
        from app.pages.about       import about_layout
        try:
            if pathname == "/experimentos":
                return experiments_layout()
            if pathname == "/datasets":
                return datasets_layout()
            if pathname == "/modelos":
                return models_layout()
            if pathname == "/sobre":
                return about_layout()
            return welcome_layout()
        except Exception as exc:
            import traceback
            logger.exception("Error rendering layout for %s", pathname)
            return html.Div([
                html.H3("Erro ao carregar a página"),
                html.Pre(str(exc)),
                html.Pre(traceback.format_exc()),
            ])

    # ── Feature selector management ───────────────────────────────────────────
    @app.callback(
        Output("feature-select", "value"),
        Output("feature-select", "options"),
        Input("dataset-select", "value"),
        Input("remove-redundant-btn", "n_clicks"),
        Input("clear-features-btn", "n_clicks"),
        State("feature-select", "value"),
    )
    def manage_features(dataset, _remove, _clear, current_features):
        triggered_id = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered else None
        )
        if triggered_id in (None, "dataset-select"):
            if dataset == "housing":
                return HOUSING_DEFAULT, HOUSING_OPTS
            return TITANIC_DEFAULT, TITANIC_OPTS
        if triggered_id == "clear-features-btn":
            return [], no_update
        if triggered_id == "remove-redundant-btn":
            current = current_features or []
            return [f for f in current if f not in TITANIC_REDUNDANT], no_update
        return no_update, no_update

    # ── Runs table: switch columns + enrich data based on active tab ──────────
    @app.callback(
        Output("runs-table", "data"),
        Output("runs-table", "columns"),
        Output("runs-table", "style_data_conditional"),
        Input("runs-table-tabs", "active_tab"),
        Input("runs-data-store", "data"),
    )
    def render_runs_table(active_tab: str, raw_data):
        data = list(raw_data or [])
        if not data:
            return [], [], []

        is_regression = any(r.get("dataset") == "housing" for r in data)
        metrics = _REG_METRICS if is_regression else _CLS_METRICS

        if active_tab == "tab-train":
            cols = _INFO_COLS + [f"metric_train_{m}" for m in metrics]
            return data, [_col_def(c) for c in cols], []

        if active_tab == "tab-overfit":
            for row in data:
                for m in metrics:
                    t  = float(row.get(f"metric_{m}", 0) or 0)
                    tr = float(row.get(f"metric_train_{m}", 0) or 0)
                    row[f"delta_{m}"] = round(abs(tr - t), 4)
            cols = ["run_name", "model"] + [f"delta_{m}" for m in metrics]
            _RED    = {"backgroundColor": "#f8d7da", "color": "#721c24"}
            _YELLOW = {"backgroundColor": "#fff3cd", "color": "#856404"}
            _GREEN  = {"backgroundColor": "#d4edda", "color": "#155724"}
            style_cond = []
            for col in [f"delta_{m}" for m in metrics]:
                style_cond += [
                    {"if": {"filter_query": f"{{{col}}} >= 0.15",                    "column_id": col}, **_RED},
                    {"if": {"filter_query": f"{{{col}}} >= 0.05 && {{{col}}} < 0.15","column_id": col}, **_YELLOW},
                    {"if": {"filter_query": f"{{{col}}} < 0.05",                     "column_id": col}, **_GREEN},
                ]
            return data, [_col_def(c) for c in cols], style_cond

        if active_tab == "tab-cv":
            cols = (
                ["run_name", "model"]
                + [f"metric_cv_mean_{m}" for m in metrics]
                + [f"metric_cv_std_{m}"  for m in metrics]
            )
            return data, [_col_def(c) for c in cols], []

        # Default: tab-test
        cols = (
            ["run_name", "dataset", "model", "n_features", "C", "alpha",
             "n_estimators", "max_depth", "scaling", "class_weight",
             "poly_features", "test_size"]
            + [f"metric_{m}" for m in metrics]
        )
        return data, [_col_def(c) for c in cols], []

    # ── Select-all toggle ─────────────────────────────────────────────────────
    @app.callback(
        Output("runs-table", "selected_rows"),
        Input("select-all-btn", "n_clicks"),
        State("runs-data-store", "data"),
        State("runs-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def toggle_select_all(_, data, current):
        n = len(data or [])
        if n == 0 or len(current or []) == n:
            return []
        return list(range(n))

    # ── Reset workspace when dataset changes ──────────────────────────────────
    @app.callback(
        Output("runs-data-store", "data", allow_duplicate=True),
        Output("runs-table", "selected_rows", allow_duplicate=True),
        Input("dataset-select", "value"),
        prevent_initial_call=True,
    )
    def reset_workspace_on_dataset_change(_dataset):
        return [], []

    # ── Model selector + class-weight options ─────────────────────────────────
    @app.callback(
        Output("model-select", "options"),
        Output("model-select", "value"),
        Output("class-weight", "options"),
        Output("class-weight", "value"),
        Input("dataset-select", "value"),
    )
    def update_model_options(dataset: str):
        if dataset == "housing":
            model_opts = [
                {"label": " Ridge Regression",          "value": "logreg"},
                {"label": " Random Forest Regressor",   "value": "rf"},
                {"label": " XGBoost Regressor",         "value": "xgb"},
            ]
            cw_opts = [{"label": " N/A (Regressão)", "value": "none", "disabled": True}]
            return model_opts, "rf", cw_opts, "none"
        model_opts = [
            {"label": " Logistic Regression",       "value": "logreg"},
            {"label": " Random Forest Classifier",  "value": "rf"},
            {"label": " XGBoost Classifier",        "value": "xgb"},
        ]
        cw_opts = [
            {"label": " Nenhum",   "value": "none"},
            {"label": " Balanced", "value": "balanced"},
        ]
        return model_opts, "logreg", cw_opts, "none"

    # ── Hyperparameter area (dynamic per model) ───────────────────────────────
    @app.callback(
        Output("hyperparams-area", "children"),
        Input("model-select", "value"),
        State("dataset-select", "value"),
    )
    def render_hyperparams(model: str, dataset: str):
        _hidden = {"display": "none"}
        if model == "logreg":
            if dataset == "housing":
                return [
                    dbc.Label("Alpha (Regularização Ridge)"),
                    dcc.Slider(id="param-alpha", min=0.01, max=100.0, step=0.1, value=1.0),
                    html.Div(id="param-penalty",    style=_hidden),
                    html.Div(id="param-C",          style=_hidden),
                    html.Div(id="param-n",          style=_hidden),
                    html.Div(id="param-d",          style=_hidden),
                    html.Div(id="param-lr",         style=_hidden),
                    html.Div(id="param-subsample",  style=_hidden),
                ]
            return [
                dbc.Label("C (Regularização)"),
                dcc.Slider(id="param-C", min=0.01, max=10.0, step=0.01, value=1.0),
                dbc.Label("Penalty (Pena)"),
                dcc.RadioItems(
                    id="param-penalty",
                    options=[
                        {"label": " L2 (Ridge)", "value": "l2"},
                        {"label": " L1 (Lasso)", "value": "l1"},
                    ],
                    value="l2",
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "block", "fontSize": "0.85rem"},
                ),
                html.Div(id="param-n",         style=_hidden),
                html.Div(id="param-d",         style=_hidden),
                html.Div(id="param-lr",        style=_hidden),
                html.Div(id="param-subsample", style=_hidden),
            ]
        if model == "xgb":
            return [
                dbc.Label("n_estimators (Nº de árvores)"),
                dcc.Slider(id="param-n",         min=10,  max=500, step=10,  value=100),
                dbc.Label("max_depth (Profundidade máxima)"),
                dcc.Slider(id="param-d",         min=1,   max=12,  step=1,   value=6),
                dbc.Label("learning_rate (Taxa de aprendizado)"),
                dcc.Slider(id="param-lr",        min=0.01,max=0.5, step=0.01,value=0.1),
                dbc.Label("subsample (Amostragem linhas)"),
                dcc.Slider(id="param-subsample", min=0.5, max=1.0, step=0.1, value=1.0),
                html.Div(id="param-C",       style=_hidden),
                html.Div(id="param-penalty", style=_hidden),
            ]
        # Random Forest
        return [
            dbc.Label("n_estimators (Nº de árvores)"),
            dcc.Slider(id="param-n",         min=10, max=500, step=10, value=100),
            dbc.Label("max_depth (Profundidade máxima)"),
            dcc.Slider(id="param-d",         min=1,  max=30,  step=1,  value=6),
            dbc.Label("min_samples_split"),
            dcc.Slider(id="param-min-split", min=2,  max=20,  step=1,  value=2),
            html.Div(id="param-C",          style=_hidden),
            html.Div(id="param-lr",         style=_hidden),
            html.Div(id="param-penalty",    style=_hidden),
            html.Div(id="param-subsample",  style=_hidden),
        ]

    # ── Run / Delete experiments ──────────────────────────────────────────────
    @app.callback(
        Output("run-status", "children"),
        Output("runs-data-store", "data"),
        Input("run-btn", "n_clicks"),
        Input("delete-btn", "n_clicks"),
        State("runs-data-store", "data"),
        State("dataset-select", "value"),
        State("feature-select", "value"),
        State("scaling", "value"),
        State("class-weight", "value"),
        State("poly-features", "value"),
        State("test-size-slider", "value"),
        State("model-select", "value"),
        State("run-name", "value"),
        State("hyperparams-area", "children"),
        State("cv-enabled", "value"),
        State("cv-folds", "value"),
        prevent_initial_call=True,
    )
    def handle_experiment_actions(
        _run_clicks, _delete_clicks, current_data,
        dataset, features, scaling, class_weight, poly_features,
        test_size, model, run_name, hyper_children, cv_enabled, cv_folds_val,
    ):
        triggered_id = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered else None
        )
        if not triggered_id:
            return no_update, no_update

        data = current_data or []

        if triggered_id == "delete-btn":
            return html.Div("All runs deleted from view."), []

        # Extract hyperparams from DOM children
        params: dict = {}
        if model == "logreg":
            if dataset == "housing":
                alpha = _extract_val(hyper_children, "param-alpha", 1.0)
                params["alpha"] = float(alpha)
            else:
                c_val   = _extract_val(hyper_children, "param-C",       1.0)
                penalty = _extract_val(hyper_children, "param-penalty",  "l2")
                params["C"]       = float(c_val if c_val is not None else 1.0)
                params["penalty"] = "l1" if penalty == "l1" else "l2"
                if params["penalty"] == "l1":
                    params["solver"] = "liblinear"
        elif model == "xgb":
            params["n_estimators"]  = int(_extract_val(hyper_children, "param-n",        100))
            params["max_depth"]     = int(_extract_val(hyper_children, "param-d",          6))
            params["learning_rate"] = float(_extract_val(hyper_children, "param-lr",      0.1))
            params["subsample"]     = float(_extract_val(hyper_children, "param-subsample",1.0))
        else:  # rf
            params["n_estimators"]    = int(_extract_val(hyper_children, "param-n",        100))
            params["max_depth"]       = int(_extract_val(hyper_children, "param-d",          6))
            params["min_samples_split"]= int(_extract_val(hyper_children, "param-min-split",  2))

        cv_k = int(cv_folds_val or 5) if "enabled" in (cv_enabled or []) else 0
        poly_enabled = "enabled" in (poly_features or [])

        run_id, run_result = run_experiment_and_log(
            features or [],
            scaling,
            model,
            params,
            run_name=run_name,
            test_size=float(test_size) / 100.0,
            class_weight=class_weight,
            poly_features=poly_enabled,
            cv_folds=cv_k,
            dataset=dataset,
        )

        if isinstance(run_result, dict):
            data.append(run_result)

        return html.Div(f"Last run: {run_id}"), data

    # ── Chart rendering ───────────────────────────────────────────────────────
    @app.callback(
        Output("bottom-chart", "figure"),
        Input("runs-data-store", "data"),
        Input("runs-table", "selected_rows"),
        Input("chart-select-bottom", "value"),
        Input("chart-data-mode", "value"),
        Input("runs-table-tabs", "active_tab"),
        Input("shap-feature-select", "value"),
    )
    def update_chart(
        data, selected_rows: List[int],
        chart_type: str, chart_data_mode: str,
        _active_tab: str, shap_feature: str,
    ):
        from app.plots import (  # noqa: PLC0415
            _empty_fig,
            plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
            plot_feature_importance, plot_calibration_curve,
            plot_metric_distribution, plot_shap_summary, plot_shap_dependence,
            plot_bar_f1, plot_radar,
        )

        df = pd.DataFrame(data or [])
        if df.empty:
            return px.line_polar()

        if not selected_rows:
            return _empty_fig("Selecione ao menos um experimento na tabela para ver o gráfico")

        valid_rows = [i for i in selected_rows if i < len(df)]
        if not valid_rows:
            return _empty_fig("Selecione ao menos um experimento na tabela para ver o gráfico")

        sel = df.iloc[valid_rows]
        runs_records = sel.to_dict("records")
        cdm = chart_data_mode or "test"

        dispatch = {
            "roc":              lambda: plot_roc_curve(runs_records, data_split=cdm),
            "pr_curve":         lambda: plot_pr_curve(runs_records, data_split=cdm),
            "confusion_matrix": lambda: plot_confusion_matrix(runs_records, data_split=cdm),
            "feature_importance": lambda: plot_feature_importance(runs_records),
            "calibration":      lambda: plot_calibration_curve(runs_records, data_split=cdm),
            "metric_dist":      lambda: plot_metric_distribution(runs_records, data_split=cdm),
            "shap_summary":     lambda: plot_shap_summary(runs_records, data_split=cdm),
            "shap_dependence":  lambda: plot_shap_dependence(runs_records, shap_feature, data_split=cdm),
            "bar_f1":           lambda: plot_bar_f1(runs_records, data_split=cdm),
            "radar":            lambda: plot_radar(runs_records, data_split=cdm),
        }

        chart = chart_type or "radar"
        handler = dispatch.get(chart)
        if handler:
            return handler()

        # pred_error not in dispatch above (regression only, accessed via chart_type)
        if chart == "pred_error":
            from app.plots import plot_prediction_error  # noqa: PLC0415
            return plot_prediction_error(runs_records, data_split=cdm)

        return px.line_polar()

    # ── Chart description text ────────────────────────────────────────────────
    @app.callback(
        Output("chart-info-text", "children"),
        Input("chart-select-bottom", "value"),
    )
    def update_chart_info(chart_type: str) -> str:
        return PLOT_DESCRIPTIONS.get(chart_type or "radar", "")

    # ── SHAP feature selector ─────────────────────────────────────────────────
    @app.callback(
        Output("shap-feature-select", "options"),
        Output("shap-feature-select", "style"),
        Input("chart-select-bottom", "value"),
        Input("runs-data-store", "data"),
        Input("runs-table", "selected_rows"),
    )
    def update_shap_selector(chart_type, data, selected_rows):
        base    = {"width": "190px", "fontSize": "0.82rem"}
        hidden  = {**base, "display": "none"}
        visible = {**base, "display": "inline-block"}

        if chart_type != "shap_dependence" or not selected_rows or not data:
            return [], hidden

        df = pd.DataFrame(data)
        row = df.iloc[selected_rows[0]].to_dict()
        features_str = str(row.get("features", "") or "")
        features = [f.strip() for f in features_str.split(",") if f.strip()]
        if not features:
            features = ["pclass", "sex", "age", "sibsp", "parch", "fare"]

        return [{"label": f, "value": f} for f in features], visible

    # ── LLM Insights Assistant ────────────────────────────────────────────────
    @app.callback(
        Output("assistant-messages", "children"),
        Output("assistant-input", "value"),
        Output("assistant-input", "disabled"),
        Output("assistant-send", "disabled"),
        Input("assistant-send", "n_clicks"),
        Input("assistant-input", "n_submit"),
        State("assistant-input", "value"),
        State("runs-data-store", "data"),
        State("runs-table", "selected_rows"),
        State("assistant-messages", "children"),
        prevent_initial_call=True,
    )
    def generate_llm_insight(
        _n_clicks, _n_submit, question, table_data, selected_rows,
        current_messages,
    ):
        if not callback_context.triggered:
            return no_update, no_update, no_update, no_update
        if not question or not question.strip():
            return no_update, "", False, False

        history = current_messages if isinstance(current_messages, list) else (
            [current_messages] if current_messages else []
        )

        if not selected_rows:
            notice = html.Div(
                "Selecione ao menos um experimento na tabela para obter insights.",
                style={"fontSize": "0.82rem", "fontStyle": "italic", "color": "#adb5bd"},
            )
            return history + [notice], "", False, False

        try:
            from llm.service import generate_insight  # noqa: PLC0415
            runs_df    = pd.DataFrame(table_data or [])
            context_df = runs_df.iloc[selected_rows] if not runs_df.empty else runs_df
            answer, source = generate_insight(question.strip(), context_df)
            label = "LLM" if source == "llm" else "Local"
            label_color = "#8e44ad" if source == "llm" else "#27ae60"

            new_exchange = [
                html.Hr(style={"margin": "6px 0", "borderColor": "#dee2e6"}),
                html.Div(
                    [html.Strong("Você: ", style={"color": "#2980b9"}), question.strip()],
                    className="mb-1",
                    style={"fontSize": "0.82rem"},
                ),
                html.Div(
                    [html.Strong(f"Assistente ({label}): ", style={"color": label_color}), answer],
                    style={"fontSize": "0.82rem", "whiteSpace": "pre-wrap"},
                ),
            ]
            return history + new_exchange, "", False, False
        except Exception as e:
            error_msg = html.Div(
                f"Erro ao processar insight: {str(e)}",
                style={"fontSize": "0.82rem", "color": "red"},
            )
            return history + [error_msg], "", False, False

    # ── Auto-scroll assistant box to the bottom on new message ────────────────
    app.clientside_callback(
        """
        function(children) {
            var el = document.getElementById('assistant-scroll');
            if (el) { el.scrollTop = el.scrollHeight; }
            return null;
        }
        """,
        Output("dummy-output", "children"),
        Input("assistant-messages", "children"),
    )

from __future__ import annotations

import os
from typing import List
import sys
from pathlib import Path

# Ensure project root is on sys.path so sibling packages (experiments, ml) import correctly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dash
from dash import Input, Output, State, dcc, html, dash_table, callback_context, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from experiments.tracker import list_runs, run_experiment_and_log


APP_TITLE = "ML Playground"


def serve_app() -> dash.Dash:
    """Create and return the Dash app."""
    app = dash.Dash(
        __name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True, title=APP_TITLE
    )
    server = app.server

    # Sidebar removed per user request (visualizations list removed)
    sidebar = html.Div()

    # prepare dataset features for the features dropdown at layout creation
    import seaborn as sns
    _df = sns.load_dataset("titanic")
    _features = [c for c in _df.columns if c not in ("survived",)]
    _feature_opts = [{"label": f, "value": f} for f in _features]
    _feature_default = [f for f in ["age", "sex", "pclass"] if f in _features]

    controls = dbc.Card(
        [
            dbc.CardHeader("Pipeline Configuration"),
            dbc.CardBody(
                [
                    dbc.Label("Features (auto-filled from dataset)"),
                    dcc.Dropdown(
                        id="feature-select", 
                        multi=True, 
                        options=_feature_opts, 
                        value=_feature_default,
                        # Removido minHeight do style para usar o comportamento padrão do Dash
                        # que expande conforme os itens são selecionados.
                    ),
                    html.Hr(),
                    dbc.Label("Scaling Strategy"),
                    dcc.RadioItems(
                        id="scaling", options=[
                            {"label": "None", "value": "none"},
                            {"label": "StandardScaler", "value": "standard"},
                        ], value="standard",
                    ),
                    html.Hr(),
                    dbc.Label("Class Balancing"),
                    dcc.RadioItems(
                        id="class-weight",
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "Balanced", "value": "balanced"}
                        ],
                        value="none",
                    ),
                    html.Hr(),
                    dbc.Label("Polynomial Features"),
                    dcc.Checklist(
                        id="poly-features",
                        options=[{"label": "Degree 2 Interactions", "value": "enabled"}],
                        value=[]
                    ),
                    html.Hr(),
                    dbc.Label("Train/Test Split (%)"),
                    dcc.Slider(id="test-size-slider", min=10, max=50, step=5, value=20, marks={10: "10%", 20: "20%", 30: "30%", 40: "40%", 50: "50%"}),
                    html.Hr(),
                    dbc.Label("Model"),
                    dcc.RadioItems(
                        id="model-select",
                        options=[
                            {"label": "Logistic Regression", "value": "logreg"},
                            {"label": "Random Forest", "value": "rf"},
                        ],
                        value="logreg",
                    ),
                    html.Hr(),
                    # initial hyperparams children (show sliders immediately)
                    html.Div(id="hyperparams-area" , children=(
                        [dbc.Label("C"), dcc.Slider(id="param-C", min=0.01, max=10.0, step=0.01, value=1.0)]
                    )),
                    html.Hr(),
                    dbc.Label("Run name (optional)"),
                    dcc.Input(id="run-name", placeholder="optional run name", type="text", style={"width": "100%"}),
                    html.Br(), html.Br(),
                    dbc.Row([
                        dbc.Col(dbc.Button("Run Experiment", id="run-btn", color="primary", class_name="w-100")),
                        dbc.Col(dbc.Button("Delete All Runs", id="delete-btn", color="danger", class_name="w-100")),
                    ]),
                    html.Div(id="run-status", className="mt-2"),
                ],
                style={"overflowY": "auto", "flex": "1"}
            ),
        ], style={"height": "100%", "display": "flex", "flexDirection": "column"}
    )

    # initialize runs table data from MLflow (if available)
    try:
        runs_df = list_runs()
        initial_runs = runs_df.to_dict("records")
    except Exception:
        initial_runs = []

    runs_table = dash_table.DataTable(
        id="runs-table",
        columns=[{"name": (c.replace("metric_", "").replace("_", " ").title() if "metric_" in c else c.replace("_", " ").title()), "id": c, "type": "numeric", "format": {"specifier": ".2f"} if "metric_" in c else None} 
                 for c in ["run_name", "model", "n_features", "C", "n_estimators", "max_depth", "scaling", "class_weight", "poly_features", "test_size",
                           "metric_accuracy", "metric_precision", "metric_recall", "metric_f1", "metric_roc_auc"]],
        data=initial_runs if initial_runs else [],
        row_selectable="multi",
        selected_rows=[],
        style_table={"overflowX": "auto", "minWidth": "100%"},
        style_cell={
            "textAlign": "center",
            "minWidth": "80px", "width": "80px", "maxWidth": "80px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_header={
            "backgroundColor": "rgb(230, 230, 230)",
            "fontWeight": "bold",
            "textAlign": "center"
        },
    )

    # placeholder for chart (now rendered in bottom-chart)

    # Header - Solid, Clean, Modern
    header = html.Div(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        html.H3("ML PLAYGROUND", style={"color": "white", "margin": 0, "fontWeight": "800", "letterSpacing": "2px"}),
                        width="auto"
                    ),
                    dbc.Col(
                        html.Div("Titanic Experiment Discovery", style={"color": "white", "opacity": "0.7", "fontSize": "0.9rem", "marginLeft": "15px"}),
                        width="auto",
                        className="align-self-end pb-1"
                    ),
                    dbc.Col(
                        dbc.Badge("v1.0 MVP", color="light", text_color="primary", className="ms-auto px-3"),
                        width="auto",
                        className="ms-auto"
                    ),
                ],
                align="center",
                className="h-100"
            ),
            fluid=True,
            style={"height": "100%"}
        ),
        style={
            "backgroundColor": "#1a2a3a",
            "height": "60px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "zIndex": "1000"
        }
    )

    # Footer - Simple, Fixed, Professional
    footer = html.Div(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(html.Small("© 2026 ML Playground. Built with Dash & MLflow.", className="text-muted")),
                    dbc.Col(
                        html.Div(
                            [
                                html.Small("Backend: ", className="text-muted me-2"),
                                dbc.Badge("Live", color="success", pill=True, style={"fontSize": "0.6rem"}),
                            ],
                            className="text-end"
                        ),
                        width="auto"
                    ),
                ],
                align="center"
            ),
            fluid=True
        ),
        style={
            "height": "35px",
            "backgroundColor": "#f8f9fa",
            "borderTop": "1px solid #dee2e6",
            "display": "flex",
            "alignItems": "center",
            "position": "fixed",
            "bottom": "0",
            "width": "100%",
            "zIndex": "1000"
        }
    )

    # Layout: Dashboard View (100% height, internal scrolls only)
    app.layout = html.Div(
        [
            # CSS customizado para forçar altura do dropdown de features
            html.Div(
                children=[
                    # Usando uma string literal no children para injetar o CSS
                    # ou uma div invisível com style, preferimos a abordagem mais simples de layout
                ],
                style={"display": "none"}
            ),
            # CSS para silenciar o warning do Sklearn no frontend Dash (se houver) e outros ajustes
            html.Div(id="dummy-output", style={"display": "none"}),
            header,
            dbc.Container(
                [
                    dbc.Row(
                        [
                            # Left: Config panel (increased slightly)
                            dbc.Col(
                                html.Div(controls, style={"height": "100%", "display": "flex", "flexDirection": "column"}),
                                width=3,
                                style={"height": "calc(100vh - 110px)"}
                            ),
                            # Right: Results area (stays the same)
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Experiment Runs", style={"fontWeight": "600"}),
                                            dbc.CardBody(runs_table, style={"padding": "0", "height": "250px", "overflowY": "auto"}),
                                        ],
                                        className="mb-3"
                                    ),
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Visualizations", style={"fontWeight": "600"}),
                                            dbc.CardBody(
                                                [
                                                    dcc.Dropdown(
                                                        id="chart-select-bottom",
                                                        options=[
                                                            {"label": "Radar (metrics)", "value": "radar"},
                                                            {"label": "Bar — F1 by run", "value": "bar_f1"},
                                                        ],
                                                        value="radar",
                                                        clearable=False,
                                                        className="mb-1",
                                                        style={"width": "300px", "fontSize": "0.9rem"}
                                                    ),
                                                    dcc.Graph(
                                                        id="bottom-chart", 
                                                        style={"height": "calc(100vh - 540px)"},
                                                        config={"displayModeBar": False}
                                                    ),
                                                ],
                                                style={"display": "flex", "flexDirection": "column", "padding": "10px"}
                                            ),
                                        ],
                                        style={"flex": "1", "minHeight": "0"}
                                    ),
                                ],
                                width=7,
                                style={"height": "calc(100vh - 110px)", "display": "flex", "flexDirection": "column"}
                            ),
                            # Far Right: AI Assistant Placeholder (decreased by the same amount)
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("AI Insights Assistant", style={"fontWeight": "600"}),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "Ask me anything about your experiment results or model performance.",
                                                            style={"fontSize": "0.85rem", "color": "#6c757d", "marginBottom": "15px"}
                                                        ),
                                                        # Placeholder for future chat messages
                                                        html.Div(
                                                            style={
                                                                "flex": "1", 
                                                                "backgroundColor": "#f8f9fa", 
                                                                "borderRadius": "5px", 
                                                                "border": "1px solid #dee2e6",
                                                                "padding": "10px",
                                                                "marginBottom": "15px",
                                                                "overflowY": "auto"
                                                            },
                                                            children=[
                                                                html.Div("System: Assistant is ready. (Module pending implementation)", 
                                                                         style={"fontSize": "0.8rem", "fontStyle": "italic", "color": "#adb5bd"})
                                                            ]
                                                        ),
                                                    ],
                                                    style={"display": "flex", "flexDirection": "column", "height": "calc(100% - 70px)"}
                                                ),
                                                dbc.InputGroup(
                                                    [
                                                        dbc.Input(placeholder="Ask about your runs...", type="text", disabled=True),
                                                        dbc.Button("Send", color="primary", disabled=True),
                                                    ]
                                                )
                                            ],
                                            style={"display": "flex", "flexDirection": "column", "height": "100%", "padding": "15px"}
                                        ),
                                    ],
                                    style={"height": "100%", "display": "flex", "flexDirection": "column"}
                                ),
                                width=2,
                                style={"height": "calc(100vh - 110px)"}
                            ),
                        ],
                        className="g-4"
                    )
                ],
                fluid=True,
                style={"paddingTop": "20px", "height": "calc(100vh - 95px)"}
            ),
            footer
        ],
        style={
            "height": "100vh",
            "backgroundColor": "#f0f2f5",
            "overflow": "hidden"
        }
    )

    @app.callback(
        Output("feature-select", "options"),
        Output("feature-select", "value"),
        Input("feature-select", "id"),
    )
    def populate_features(_):
        # kept for compatibility: return the same options/value already set at layout
        return _feature_opts, _feature_default

    @app.callback(Output("hyperparams-area", "children"), Input("model-select", "value"))
    def render_hyperparams(model: str):
        if model == "logreg":
            return [dbc.Label("C"), dcc.Slider(id="param-C", min=0.01, max=10.0, step=0.01, value=1.0)]
        return [
            dbc.Label("n_estimators"),
            dcc.Slider(id="param-n", min=10, max=500, step=10, value=100),
            dbc.Label("max_depth"),
            dcc.Slider(id="param-d", min=1, max=30, step=1, value=6),
        ]

    @app.callback(
        Output("run-status", "children"),
        Output("runs-table", "data"),
        Input("run-btn", "n_clicks"),
        Input("delete-btn", "n_clicks"),
        State("feature-select", "value"),
        State("scaling", "value"),
        State("class-weight", "value"),
        State("poly-features", "value"),
        State("test-size-slider", "value"),
        State("model-select", "value"),
        State("run-name", "value"),
        State("hyperparams-area", "children"),
        prevent_initial_call=True,
    )
    def handle_experiment_actions(run_clicks: int, delete_clicks: int, features: List[str], scaling: str, class_weight: str, poly_features: List[str], test_size: int, model: str, run_name: str | None, hyper_children):
        """Unified callback for all experiment actions (Run/Delete) to avoid duplicate output conflicts."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update
            
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if triggered_id == "delete-btn":
            from experiments.tracker import delete_all_runs
            try:
                count = delete_all_runs()
                runs = list_runs()
                return html.Div([f"Deleted {count} runs."]), runs.to_dict("records")
            except Exception as e:
                return html.Div([f"Delete failed: {e}"], style={"color": "red"}), []
                
        # Handle Run Action
        params = {}
        poly_enabled = "enabled" in (poly_features or [])
        test_size_float = float(test_size) / 100.0

        # helper to extract slider value from children (recursive)
        def _extract_val(children, target_id, default=None):
            if not children:
                return default
            items = children if isinstance(children, list) else [children]
            for it in items:
                try:
                    # Dash component dict structure
                    props = it.get("props", {})
                    if props.get("id") == target_id:
                        return props.get("value", default)
                    # check nested children
                    nested = props.get("children")
                    if nested:
                        v = _extract_val(nested, target_id, None)
                        if v is not None:
                            return v
                except Exception:
                    continue
            return default

        if model == "logreg":
            c_val = _extract_val(hyper_children, "param-C", 1.0)
            params["C"] = float(c_val if c_val is not None else 1.0)
        else:
            n_val = _extract_val(hyper_children, "param-n", 100)
            d_val = _extract_val(hyper_children, "param-d", 6)
            params["n_estimators"] = int(n_val if n_val is not None else 100)
            params["max_depth"] = int(d_val if d_val is not None else 6)

        run_id, metrics = run_experiment_and_log(
            features or [], 
            scaling, 
            model, 
            params, 
            run_name=run_name,
            test_size=test_size_float,
            class_weight=class_weight,
            poly_features=poly_enabled
        )
        try:
            data = list_runs().to_dict("records")
        except Exception:
            data = []
        return html.Div([f"Last run: {run_id}"]), data


    @app.callback(
        Output("bottom-chart", "figure"),
        Input("runs-table", "data"),
        Input("runs-table", "selected_rows"),
        Input("chart-select-bottom", "value"),
    )
    def update_chart(data, selected_rows: List[int], chart_select_bottom: str):
        df = pd.DataFrame(data or [])
        if df.empty:
            return px.line_polar()

        # prefer bottom selector
        chart_type = chart_select_bottom or "radar"

        # List of internal metric column names matching the MLflow data structure
        internal_metrics = ["metric_accuracy", "metric_precision", "metric_recall", "metric_f1", "metric_roc_auc"]
        # Human readable names for the chart axes
        display_metrics = ["Accuracy", "Precision", "Recall", "F1", "Roc Auc"]

        # filter for selected rows IF they exist, otherwise show top 3
        if selected_rows:
            sel = df.iloc[selected_rows]
        else:
            sort_col = "metric_f1" if "metric_f1" in df.columns else (df.columns[0] if not df.empty else None)
            if sort_col and not df.empty:
                sel = df.nlargest(min(3, len(df)), sort_col)
            else:
                sel = df.head(min(3, len(df)))

        if chart_type == "bar_f1":
            # safe bar: if f1 missing, use 0.0
            y_col = "metric_f1" if "metric_f1" in df.columns else "f1"
            if y_col not in df.columns:
                df[y_col] = 0.0
            
            plot_df = sel if selected_rows else df
            fig = px.bar(plot_df.sort_values(y_col, ascending=False) if y_col in plot_df.columns else plot_df, 
                         x="name" if "name" in plot_df.columns else plot_df.index, 
                         y=y_col, color="model" if "model" in plot_df.columns else None, 
                         title="F1 by run")
            return fig

        # default: radar chart
        available_internal = [m for m in internal_metrics if m in df.columns]
        available_display = [display_metrics[internal_metrics.index(m)] for m in available_internal]
        
        if not available_internal:
            return px.line_polar()

        fig = px.line_polar()
        for _, r in sel.iterrows():
            values = [float(r.get(m, 0) or 0) for m in available_internal]
            # Use name if available for legend
            label = r.get("name") or r.get("run_name") or r.get("run_id") or "Run"
            fig.add_scatterpolar(r=values + [values[0]], theta=available_display + [available_display[0]], 
                                 name=str(label), fill="toself")
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Model Comparison (Radar)"
        )
        return fig

    return app


if __name__ == "__main__":
    dash_app = serve_app()
    # use `run` (newer Dash) instead of deprecated `run_server`
    dash_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=True)
"""ML Experiment Playground — ponto de entrada da aplicação Dash.

Execute a partir da raiz do projeto::

    python app/app.py

Depois abra http://localhost:8050 no navegador.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mlflow
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from dash import Input, Output, State, dash_table, dcc, html

from app.components import sidebar
from experiments.tracker import (
    delete_all_runs,
    list_runs,
    load_roc_data,
    log_artifact_dict,
    log_metrics,
    log_model,
    log_params,
    start_run,
)
from ml.pipeline import build_pipeline
from ml.train import get_available_features, load_titanic, train_and_evaluate

# ── Dataset carregado uma vez na inicialização ────────────────────────────────
DF = load_titanic()
FEATURES = get_available_features(DF)

# ── Inicialização do app ───────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="ML Playground",
)

# ── Componentes estáticos ─────────────────────────────────────────────────────

_HEADER = html.Header(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            "Playground de Experimentos ML",
                            className="mb-0 fw-bold text-white",
                        ),
                        html.Small(
                            "Compare modelos de classificação · Dataset Titanic · MLflow local",
                            className="text-white-50",
                        ),
                    ],
                    width=8,
                ),
                dbc.Col(
                    [
                        dbc.Badge("scikit-learn", color="light", text_color="dark", className="me-1"),
                        dbc.Badge("MLflow", color="light", text_color="dark", className="me-1"),
                        dbc.Badge("Dash + Plotly", color="light", text_color="dark"),
                    ],
                    width=4,
                    className="d-flex align-items-center justify-content-end",
                ),
            ],
            align="center",
        ),
        fluid=True,
    ),
    className="py-3",
    style={"background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"},
)

_FOOTER = html.Footer(
    dbc.Container(
        [
            html.Hr(className="mb-2"),
            html.P(
                "© 2026 ML Playground · Construído com Dash, Plotly, scikit-learn e MLflow",
                className="text-center text-muted small mb-0",
            ),
        ],
        fluid=True,
    ),
    className="py-3 bg-light mt-4",
)

# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    [
        _HEADER,

        # Conteúdo principal: sidebar + painel de resultados
        dbc.Row(
            [
                sidebar(FEATURES),

                dbc.Col(
                    [
                        # ── Tabela de corridas ────────────────────────────
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H5("Corridas de Experimento"),
                                    width="auto",
                                ),
                                dbc.Col(
                                    html.Small(
                                        "Selecione linhas para comparar nos gráficos.",
                                        className="text-muted",
                                    ),
                                    width="auto",
                                    className="align-self-center ms-2",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Apagar Todas as Corridas",
                                        id="delete-btn",
                                        color="danger",
                                        outline=True,
                                        size="sm",
                                        n_clicks=0,
                                    ),
                                    width="auto",
                                    className="ms-auto",
                                ),
                            ],
                            className="mb-2 align-items-center",
                        ),
                        dash_table.DataTable(
                            id="runs-table",
                            columns=[],
                            data=[],
                            row_selectable="multi",
                            selected_rows=[],
                            page_size=10,
                            filter_action="native",
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "fontSize": 12,
                                "padding": "6px 10px",
                                "whiteSpace": "normal",
                                "textAlign": "left",
                            },
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor": "#f8f9fa",
                                "textAlign": "left",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "#fafafa",
                                },
                                {
                                    "if": {"state": "selected"},
                                    "backgroundColor": "#e8f0fe",
                                    "border": "1px solid #4285f4",
                                },
                            ],
                        ),

                        # ── Seção de gráficos ─────────────────────────────
                        html.H5("Comparação de Corridas", className="mt-4 mb-2"),
                        dbc.Row(
                            [
                                # 20 % — seletor de tipo de gráfico
                                dbc.Col(
                                    [
                                        html.P(
                                            "Tipo de Gráfico",
                                            className="fw-semibold mb-2 small text-uppercase text-muted",
                                        ),
                                        dbc.RadioItems(
                                            id="chart-type",
                                            options=[
                                                {
                                                    "label": html.Span(
                                                        ["Radar", html.Br(),
                                                         html.Small("visão holística", className="text-muted")],
                                                    ),
                                                    "value": "radar",
                                                },
                                                {
                                                    "label": html.Span(
                                                        ["Barras", html.Br(),
                                                         html.Small("métricas lado a lado", className="text-muted")],
                                                    ),
                                                    "value": "barras",
                                                },
                                                {
                                                    "label": html.Span(
                                                        ["Curva ROC", html.Br(),
                                                         html.Small("TPR × FPR", className="text-muted")],
                                                    ),
                                                    "value": "roc",
                                                },
                                            ],
                                            value="radar",
                                            className="chart-selector",
                                        ),
                                    ],
                                    width=2,
                                    className="border-end pe-3",
                                ),

                                # 80 % — área do gráfico
                                dbc.Col(
                                    dcc.Graph(
                                        id="main-chart",
                                        style={"height": "420px"},
                                        config={"displayModeBar": True},
                                    ),
                                    width=10,
                                ),
                            ],
                            className="mt-1",
                        ),
                    ],
                    width=9,
                    className="p-3",
                ),
            ]
        ),

        _FOOTER,

        # ── Componentes ocultos ───────────────────────────────────────────
        dcc.Store(id="experiment-store", data=0),
        dcc.ConfirmDialog(
            id="confirm-delete",
            message="Apagar todas as corridas? Esta ação não pode ser desfeita.",
        ),
    ],
    fluid=True,
    className="px-0",
)

# ── Configuração das colunas da tabela ────────────────────────────────────────
_DISPLAY_COLS = [
    "run_name", "start_time",
    "model", "scaler", "n_features",
    "C", "n_estimators", "max_depth",
    "metric_accuracy", "metric_precision", "metric_recall",
    "metric_f1", "metric_roc_auc",
]
_COL_LABELS: dict[str, str] = {
    "run_name": "Corrida",
    "start_time": "Hora",
    "model": "Modelo",
    "scaler": "Normaliz.",
    "n_features": "# Var.",
    "C": "C",
    "n_estimators": "Árvores",
    "max_depth": "Profund.",
    "metric_accuracy": "Acurácia",
    "metric_precision": "Precisão",
    "metric_recall": "Recall",
    "metric_f1": "F1",
    "metric_roc_auc": "ROC AUC",
}
_METRIC_COLS = [
    "metric_accuracy", "metric_precision",
    "metric_recall", "metric_f1", "metric_roc_auc",
]
_METRIC_LABELS_PT = ["Acurácia", "Precisão", "Recall", "F1", "ROC AUC"]


def _format_table(runs_df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Converte o DataFrame de corridas em registros compatíveis com DataTable.

    O campo ``run_id`` é incluído nos dados (mas não nas colunas exibidas)
    para que callbacks possam carregar artefatos por ID de corrida.
    """
    if runs_df.empty:
        return [], []

    available = [c for c in _DISPLAY_COLS if c in runs_df.columns]
    display = runs_df[available].copy()

    # Inclui run_id nos dados sem exibi-lo na tabela
    if "run_id" in runs_df.columns:
        display["run_id"] = runs_df["run_id"].values

    if "start_time" in display.columns:
        display["start_time"] = display["start_time"].dt.strftime("%d/%m %H:%M")

    for col in _METRIC_COLS:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(4)

    columns = [{"name": _COL_LABELS.get(c, c), "id": c} for c in available]
    return display.to_dict("records"), columns


def _empty_chart(message: str) -> go.Figure:
    """Retorna uma figura vazia com mensagem centralizada."""
    fig = go.Figure()
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1])),
        annotations=[{
            "text": message,
            "showarrow": False,
            "font": {"size": 14, "color": "#888"},
            "x": 0.5, "y": 0.5, "xref": "paper", "yref": "paper",
        }],
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


# ── Callbacks ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("lr-params", "style"),
    Output("rf-params", "style"),
    Input("model-selector", "value"),
)
def toggle_hyperparams(model: str) -> tuple[dict, dict]:
    """Exibe/oculta o painel de hiperparâmetros do modelo selecionado."""
    show: dict = {}
    hide: dict = {"display": "none"}
    return (show, hide) if model == "logistic_regression" else (hide, show)


@app.callback(
    Output("experiment-store", "data"),
    Output("run-status", "children"),
    Input("run-btn", "n_clicks"),
    State("feature-selector", "value"),
    State("scaler-selector", "value"),
    State("model-selector", "value"),
    State("lr-C", "value"),
    State("rf-n-estimators", "value"),
    State("rf-max-depth", "value"),
    State("run-name", "value"),
    State("experiment-store", "data"),
    prevent_initial_call=True,
)
def run_experiment(
    _n_clicks: int,
    features: list[str] | None,
    scaler: str,
    model: str,
    lr_C: float,
    rf_n_estimators: int,
    rf_max_depth: int,
    run_name: str | None,
    store_count: int,
) -> tuple[int, object]:
    """Constrói e avalia um pipeline, depois registra a corrida no MLflow."""
    if not features:
        return store_count, dbc.Alert(
            "Selecione pelo menos uma variável antes de executar.",
            color="warning", className="py-1 mt-1",
        )

    try:
        pipeline = build_pipeline(
            model_type=model,
            scaler_type=scaler,
            lr_C=float(lr_C or 1.0),
            rf_n_estimators=int(rf_n_estimators or 100),
            rf_max_depth=int(rf_max_depth or 0),
        )

        metrics, fitted_pipeline, roc_data = train_and_evaluate(pipeline, DF, features)

        params: dict = {
            "model": model,
            "scaler": scaler,
            "features": ",".join(features),
            "n_features": len(features),
        }
        if model == "logistic_regression":
            params["C"] = lr_C
        else:
            params["n_estimators"] = rf_n_estimators
            params["max_depth"] = (
                "ilimitada" if int(rf_max_depth or 0) <= 0 else rf_max_depth
            )

        with start_run(run_name=run_name or None):
            log_params(params)
            log_metrics(metrics)
            log_model(fitted_pipeline)
            log_artifact_dict(roc_data, "roc_data.json")

        status = dbc.Alert(
            [
                html.Strong("Corrida concluída! "),
                f"Acurácia={metrics['accuracy']:.4f}  "
                f"F1={metrics['f1']:.4f}  "
                f"ROC AUC={metrics['roc_auc']:.4f}",
            ],
            color="success", className="py-1 mt-1",
        )
        return (store_count or 0) + 1, status

    except Exception as exc:  # noqa: BLE001
        return store_count, dbc.Alert(
            f"Erro: {exc}", color="danger", className="py-1 mt-1",
        )


@app.callback(
    Output("runs-table", "data"),
    Output("runs-table", "columns"),
    Input("experiment-store", "data"),
)
def refresh_table(_store_count: int) -> tuple[list[dict], list[dict]]:
    """Recarrega a tabela do MLflow sempre que o store de experimentos muda.

    Também dispara no carregamento da página para exibir corridas existentes.
    """
    return _format_table(list_runs())


@app.callback(
    Output("main-chart", "figure"),
    Input("chart-type", "value"),
    Input("runs-table", "selected_rows"),
    State("runs-table", "data"),
)
def update_chart(
    chart_type: str,
    selected_rows: list[int],
    table_data: list[dict],
) -> go.Figure:
    """Renderiza o gráfico selecionado para as corridas marcadas na tabela."""
    if not selected_rows or not table_data:
        return _empty_chart("Selecione corridas na tabela acima para comparar")

    if chart_type == "radar":
        return _build_radar(selected_rows, table_data)
    if chart_type == "barras":
        return _build_bar(selected_rows, table_data)
    if chart_type == "roc":
        return _build_roc(selected_rows, table_data)
    return _empty_chart("Tipo de gráfico desconhecido")


def _build_radar(selected_rows: list[int], table_data: list[dict]) -> go.Figure:
    """Gráfico de radar — uma teia por corrida selecionada."""
    fig = go.Figure()
    for idx in selected_rows:
        row = table_data[idx]
        values = [float(row.get(m) or 0) for m in _METRIC_COLS]
        closed = values + [values[0]]
        labels = _METRIC_LABELS_PT + [_METRIC_LABELS_PT[0]]
        fig.add_trace(go.Scatterpolar(
            r=closed,
            theta=labels,
            fill="toself",
            name=row.get("run_name", f"run-{idx}"),
            opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], visible=True, tickformat=".2f")),
        showlegend=True,
        legend=dict(x=1.05, y=1.0),
        margin=dict(l=40, r=120, t=40, b=40),
    )
    return fig


def _build_bar(selected_rows: list[int], table_data: list[dict]) -> go.Figure:
    """Gráfico de barras agrupadas — métricas × corridas."""
    fig = go.Figure()
    for idx in selected_rows:
        row = table_data[idx]
        values = [float(row.get(m) or 0) for m in _METRIC_COLS]
        fig.add_trace(go.Bar(
            name=row.get("run_name", f"run-{idx}"),
            x=_METRIC_LABELS_PT,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.12], title="Valor"),
        xaxis_title="Métrica",
        showlegend=True,
        legend=dict(x=1.0, y=1.0),
        margin=dict(l=40, r=120, t=40, b=40),
    )
    return fig


def _build_roc(selected_rows: list[int], table_data: list[dict]) -> go.Figure:
    """Curva ROC — TPR × FPR por corrida, com AUC na legenda."""
    fig = go.Figure()
    any_data = False

    for idx in selected_rows:
        row = table_data[idx]
        run_id = row.get("run_id")
        if not run_id:
            continue
        roc = load_roc_data(run_id)
        if roc is None:
            continue
        auc_val = row.get("metric_roc_auc", "?")
        fig.add_trace(go.Scatter(
            x=roc["fpr"],
            y=roc["tpr"],
            mode="lines",
            name=f"{row.get('run_name', f'run-{idx}')}  (AUC={auc_val})",
        ))
        any_data = True

    if not any_data:
        return _empty_chart(
            "Curva ROC não disponível para as corridas selecionadas.\n"
            "Execute um novo experimento para gerar os dados."
        )

    # Linha de referência aleatória
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Aleatório",
        showlegend=False,
    ))
    fig.update_layout(
        xaxis=dict(title="Taxa de Falso Positivo (FPR)", range=[0, 1]),
        yaxis=dict(title="Taxa de Verdadeiro Positivo (TPR)", range=[0, 1.02]),
        showlegend=True,
        legend=dict(x=0.55, y=0.08),
        margin=dict(l=50, r=40, t=40, b=50),
    )
    return fig


@app.callback(
    Output("confirm-delete", "displayed"),
    Input("delete-btn", "n_clicks"),
    prevent_initial_call=True,
)
def show_delete_confirm(_n_clicks: int) -> bool:
    """Abre o diálogo de confirmação ao clicar em 'Apagar Todas as Corridas'."""
    return True


@app.callback(
    Output("experiment-store", "data", allow_duplicate=True),
    Output("run-status", "children", allow_duplicate=True),
    Input("confirm-delete", "submit_n_clicks"),
    State("experiment-store", "data"),
    prevent_initial_call=True,
)
def handle_delete(submit_n_clicks: int | None, store_count: int) -> tuple[int, object]:
    """Apaga todas as corridas do MLflow após confirmação do usuário."""
    if not submit_n_clicks:
        return dash.no_update, dash.no_update

    n_deleted = delete_all_runs()
    return (store_count or 0) + 1, dbc.Alert(
        f"{n_deleted} corrida(s) apagada(s).",
        color="info", className="py-1 mt-1",
    )


# ── Ponto de entrada ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)

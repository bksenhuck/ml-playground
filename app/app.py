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
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, dash_table
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

    controls = dbc.Card(
        [
            dbc.CardHeader("Pipeline Configuration"),
            dbc.CardBody(
                [
                    dbc.Label("Features (auto-filled from dataset)"),
                    dcc.Dropdown(id="feature-select", multi=True),
                    html.Hr(),
                    dbc.Label("Scaling"),
                    dcc.RadioItems(
                        id="scaling", options=[
                            {"label": "None", "value": "none"},
                            {"label": "StandardScaler", "value": "standard"},
                        ], value="standard",
                    ),
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
                    dbc.Label("Chart"),
                    dcc.Dropdown(
                        id="chart-select",
                        options=[
                            {"label": "Radar (metrics)", "value": "radar"},
                            {"label": "Bar — F1 by run", "value": "bar_f1"},
                        ],
                        value="radar",
                    ),
                    html.Div(id="hyperparams-area"),
                    html.Hr(),
                    dbc.Label("Run name (optional)"),
                    dcc.Input(id="run-name", placeholder="optional run name", type="text", style={"width": "100%"}),
                    html.Br(), html.Br(),
                    dbc.Row([
                        dbc.Col(dbc.Button("Run Experiment", id="run-btn", color="primary", class_name="w-100")),
                        dbc.Col(dbc.Button("Delete All Runs", id="delete-btn", color="danger", class_name="w-100")),
                    ]),
                    html.Div(id="run-status", className="mt-2"),
                ]
            ),
        ], class_name="mb-3",
    )

    # initialize runs table data from MLflow (if available)
    try:
        runs_df = list_runs()
        initial_runs = runs_df.to_dict("records")
    except Exception:
        initial_runs = []

    runs_table = dash_table.DataTable(
        id="runs-table",
        columns=[{"name": c, "id": c} for c in ["run_id", "name", "model", "accuracy", "f1"]],
        data=initial_runs,
        row_selectable="multi",
        selected_rows=[],
        style_table={"overflowX": "auto"},
    )

    # placeholder for chart (now rendered in bottom-chart)

    footer = dbc.Container(
        html.Footer(
            [
                html.Div("ML Playground — lightweight experiment UI.", style={"fontWeight": "600"}),
                html.Div("Built with Dash, scikit-learn and MLflow. "),
            ],
            style={"padding": "12px", "textAlign": "center", "color": "#666", "backgroundColor": "#f8f9fa"},
        ),
        fluid=True,
    )

    # Header / top bar
    header = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="", height="28px")),
                        dbc.Col(html.H4("ML Playground", style={"margin": "0 0 0 8px"})),
                    ], align="center", class_name="g-0",
                ),
                dbc.Container(html.Div("Run and compare classification experiments — Titanic dataset"), fluid=True),
            ]
        ),
        color="#0d6efd",
        dark=True,
        class_name="mb-3",
    )

    # Layout: header, then main two-column area (left controls, right table+visuals), footer
    app.layout = dbc.Container(
        [
            header,
            dbc.Row(
                [
                    # left column: sidebar + controls stacked
                    dbc.Col(
                        [
                            sidebar,
                            html.Div(controls, style={"marginTop": "12px"}),
                        ],
                        width=3,
                    ),

                    # right column: runs table on top, visualizations below
                    dbc.Col(
                        [
                            dbc.Card([
                                dbc.CardHeader("Experiment Runs"),
                                dbc.CardBody(runs_table),
                            ]),
                            html.Div(
                                [
                                    dbc.Card([
                                        dbc.CardHeader("Visualizations"),
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
                                                    style={"width": "50%"},
                                                ),
                                                dcc.Graph(id="bottom-chart", figure={}),
                                            ]
                                        ),
                                    ])
                                ],
                                style={"marginTop": "12px"},
                            ),
                        ],
                        width=9,
                    ),
                ],
                align="start",
            ),
            html.Hr(),
            footer,
        ],
        fluid=True,
    )

    @app.callback(
        Output("feature-select", "options"),
        Output("feature-select", "value"),
        Input("feature-select", "id"),
    )
    def populate_features(_):
        import seaborn as sns

        df = sns.load_dataset("titanic")
        features = [c for c in df.columns if c not in ("survived",)]
        opts = [{"label": f, "value": f} for f in features]
        # default pick a few
        default = [f for f in ["age", "sex", "pclass"] if f in features]
        return opts, default

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
        State("feature-select", "value"),
        State("scaling", "value"),
        State("model-select", "value"),
        State("hyperparams-area", "children"),
        State("run-name", "value"),
        prevent_initial_call=True,
    )
    def on_run(n_clicks: int, features: List[str], scaling: str, model: str, hyper_children, run_name: str | None):
        """Run experiment using hyperparameters read from `hyperparams-area` children.

        This avoids referencing slider IDs that may not exist in the layout initially.
        """
        params = {"scaling": scaling, "model": model}

        # helper to extract slider value from children (list of component dicts)
        def _extract_val(children, target_id, default=None):
            if not children:
                return default
            # children can be a list or a single component
            items = children if isinstance(children, list) else [children]
            for it in items:
                try:
                    props = it.get("props", {})
                    comp_id = props.get("id")
                    if comp_id == target_id:
                        return props.get("value", default)
                    # nested children
                    nested = props.get("children")
                    if nested:
                        v = _extract_val(nested, target_id, default)
                        if v is not None:
                            return v
                except Exception:
                    continue
            return default

        if model == "logreg":
            c = _extract_val(hyper_children, "param-C", 1.0)
            params["C"] = float(c or 1.0)
        else:
            n = _extract_val(hyper_children, "param-n", 100)
            d = _extract_val(hyper_children, "param-d", 6)
            params["n_estimators"] = int(n or 100)
            params["max_depth"] = int(d or 6)

        run_id, metrics = run_experiment_and_log(features or [], scaling, model, params, run_name=run_name)
        try:
            runs = list_runs()
            data = runs.to_dict("records")
        except Exception:
            data = []
        return html.Div([f"Last run: {run_id}" ]), data


    @app.callback(
        Output("run-status", "children"),
        Output("runs-table", "data"),
        Input("delete-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_delete_all(n_clicks: int):
        from experiments.tracker import delete_all_runs

        try:
            count = delete_all_runs()
            runs = list_runs()
            data = runs.to_dict("records")
            return html.Div([f"Deleted {count} runs."]), data
        except Exception as e:
            return html.Div([f"Delete failed: {e}"], style={"color": "red"}), []

    @app.callback(
        Output("bottom-chart", "figure"),
        Input("runs-table", "data"),
        Input("runs-table", "selected_rows"),
        Input("chart-select", "value"),
        Input("chart-select-bottom", "value"),
    )
    def update_chart(data, selected_rows, chart_select_top, chart_select_bottom):
        df = pd.DataFrame(data or [])
        if df.empty:
            return px.line_polar()

        # prefer bottom selector if present, else top selector
        chart_type = chart_select_bottom or chart_select_top or "radar"

        if chart_type == "bar_f1":
            bar = px.bar(df.sort_values("f1", ascending=False), x="run_id", y="f1", color="model", title="F1 by run")
            return bar

            if selected_rows:
                sel = df.iloc[selected_rows]
            else:
                if "f1" in df.columns and not df["f1"].isna().all():
                    sel = df.nlargest(3, "f1")
                else:
                    sel = df.head(3)
            metrics = [m for m in ("accuracy", "precision", "recall", "f1", "roc_auc") if m in df.columns]
        fig = px.line_polar()
        for _, r in sel.iterrows():
            values = [r.get(m, 0) or 0 for m in metrics]
            fig.add_scatterpolar(r=values + [values[0]], theta=metrics + [metrics[0]], name=str(r.get("run_id")))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), showlegend=True)
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

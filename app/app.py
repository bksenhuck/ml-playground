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

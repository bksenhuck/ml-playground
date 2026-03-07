"""Experiments page layout (the main pipeline + results view)."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from app.components import section_header
from app.layout import make_controls, make_runs_table


def experiments_layout() -> dbc.Container:
    controls = make_controls()
    runs_table = make_runs_table()

    return dbc.Container(
        [
            dbc.Row(
                [
                    # ── Left: Config panel ────────────────────────────────────
                    dbc.Col(
                        html.Div(controls, style={"height": "100%", "display": "flex", "flexDirection": "column"}),
                        width=3,
                        style={"height": "calc(100vh - 135px)"},
                    ),

                    # ── Centre: Results ───────────────────────────────────────
                    dbc.Col(
                        [
                            # Runs table card
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.Div([
                                            html.Div(
                                                section_header(
                                                    "Experimentos", "tooltip-experiments",
                                                    "Histórico de runs registrados. Selecione linhas para visualizar "
                                                    "métricas nos gráficos abaixo. Use as abas para alternar entre "
                                                    "métricas de teste, treino, overfitting e validação cruzada.",
                                                ),
                                                className="mb-1",
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Tabs(
                                                        [
                                                            dbc.Tab(label="Teste",       tab_id="tab-test"),
                                                            dbc.Tab(label="Treino",      tab_id="tab-train"),
                                                            dbc.Tab(label="Overfitting", tab_id="tab-overfit"),
                                                            dbc.Tab(label="Cross-Val",   tab_id="tab-cv"),
                                                        ],
                                                        id="runs-table-tabs",
                                                        active_tab="tab-test",
                                                        style={"marginBottom": "-1px"},
                                                    ),
                                                    dbc.Button(
                                                        "Selecionar todos",
                                                        id="select-all-btn",
                                                        size="sm",
                                                        color="link",
                                                        className="ms-auto align-self-center",
                                                        style={"fontSize": "0.75rem", "padding": "0 4px"},
                                                    ),
                                                ],
                                                className="d-flex align-items-end",
                                            ),
                                        ]),
                                        style={"paddingBottom": "0"},
                                    ),
                                    dbc.CardBody(
                                        runs_table,
                                        style={"padding": "0", "height": "200px", "overflowY": "auto"},
                                    ),
                                ],
                                className="mb-3",
                            ),

                            # Visualizations card
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        section_header(
                                            "Visualizations", "tooltip-viz",
                                            "Gráficos interativos gerados a partir dos runs selecionados na tabela "
                                            "acima. Troque o tipo de visualização pelo dropdown.",
                                        ),
                                    ),
                                    dbc.CardBody(
                                        [
                                            # Controls row: dropdown + mode + info + SHAP selector
                                            html.Div(
                                                [
                                                    dcc.Dropdown(
                                                        id="chart-select-bottom",
                                                        options=[
                                                            {"label": "Radar (metrics)",        "value": "radar"},
                                                            {"label": "Bar — F1 by run",        "value": "bar_f1"},
                                                            {"label": "ROC Curve",              "value": "roc"},
                                                            {"label": "Precision-Recall Curve", "value": "pr_curve"},
                                                            {"label": "Confusion Matrix",       "value": "confusion_matrix"},
                                                            {"label": "Feature Importance",     "value": "feature_importance"},
                                                            {"label": "Calibration Curve",      "value": "calibration"},
                                                            {"label": "Metric Distribution",    "value": "metric_dist"},
                                                            {"label": "SHAP Summary",           "value": "shap_summary"},
                                                            {"label": "SHAP Dependence",        "value": "shap_dependence"},
                                                        ],
                                                        value="radar",
                                                        clearable=False,
                                                        style={"width": "220px", "fontSize": "0.9rem", "flexShrink": "0"},
                                                    ),
                                                    dcc.RadioItems(
                                                        id="chart-data-mode",
                                                        options=[
                                                            {"label": "Teste",  "value": "test"},
                                                            {"label": "Treino", "value": "train"},
                                                            {"label": "Ambos",  "value": "both"},
                                                        ],
                                                        value="test",
                                                        inline=True,
                                                        inputStyle={"marginRight": "4px"},
                                                        labelStyle={"marginRight": "12px", "fontSize": "0.85rem"},
                                                        style={"flexShrink": "0", "paddingLeft": "10px", "paddingRight": "4px"},
                                                    ),
                                                    html.Div(
                                                        id="chart-info-text",
                                                        style={
                                                            "fontSize": "0.85rem",
                                                            "color": "#6c757d",
                                                            "fontStyle": "italic",
                                                            "paddingLeft": "12px",
                                                            "flex": "1",
                                                            "minWidth": "0",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="shap-feature-select",
                                                        placeholder="Feature para SHAP...",
                                                        clearable=False,
                                                        style={
                                                            "width": "190px",
                                                            "fontSize": "0.82rem",
                                                            "display": "none",
                                                            "flexShrink": "0",
                                                        },
                                                    ),
                                                ],
                                                className="d-flex align-items-center mb-2",
                                            ),
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="bottom-chart",
                                                    style={"height": "calc(100vh - 537px)"},
                                                    config={"displayModeBar": False},
                                                ),
                                                type="circle",
                                                color="#2c7bb6",
                                            ),
                                        ],
                                        style={"display": "flex", "flexDirection": "column", "padding": "10px"},
                                    ),
                                ],
                                style={"flex": "1", "minHeight": "0"},
                            ),
                        ],
                        width=7,
                        style={"height": "calc(100vh - 135px)", "display": "flex", "flexDirection": "column"},
                    ),

                    # ── Right: AI Assistant ───────────────────────────────────
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    section_header(
                                        "AI Insights Assistant", "tooltip-ai",
                                        "Assistente de IA que analisa os resultados dos seus experimentos. "
                                        "Selecione runs na tabela e faça perguntas sobre métricas, overfitting, "
                                        "comparação de modelos e muito mais.",
                                        placement="left",
                                    ),
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            dcc.Loading(
                                                html.Div(
                                                    id="assistant-messages",
                                                    style={"padding": "10px"},
                                                    children=[
                                                        html.Div(
                                                            "Assistente pronto. Selecione runs na tabela e faça uma pergunta.",
                                                            style={"fontSize": "0.8rem", "fontStyle": "italic", "color": "#adb5bd"},
                                                        )
                                                    ],
                                                ),
                                                type="circle",
                                                color="#27ae60",
                                            ),
                                            id="assistant-scroll",
                                            style={
                                                "flex": "1",
                                                "minHeight": "0",
                                                "backgroundColor": "#f8f9fa",
                                                "borderRadius": "5px",
                                                "border": "1px solid #dee2e6",
                                                "marginBottom": "10px",
                                                "overflowY": "scroll",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                dcc.Loading(
                                                    dbc.Textarea(
                                                        id="assistant-input", 
                                                        placeholder="Pergunte sobre seus experimentos...", 
                                                        rows=3,
                                                        style={
                                                            "resize": "none", 
                                                            "borderRadius": "5px",
                                                            "marginBottom": "10px",
                                                            "border": "1px solid #dee2e6"
                                                        }
                                                    ),
                                                    type="circle",
                                                    color="#2980b9",
                                                    style={"transform": "scale(0.5)"},
                                                    overlay_style={"visibility":"visible", "filter": "blur(1px)"},
                                                ),
                                                dcc.Loading(
                                                    dbc.Button(
                                                        "Enviar Pergunta", 
                                                        id="assistant-send", 
                                                        color="primary", 
                                                        className="w-100 fw-bold",
                                                        style={"borderRadius": "5px", "padding": "10px"}
                                                    ),
                                                    type="dot",
                                                    color="white",
                                                    style={"transform": "scale(0.3)"}
                                                ),
                                            ], 
                                            style={"display": "flex", "flexDirection": "column"}
                                        ),
                                    ],
                                    style={"display": "flex", "flexDirection": "column", "height": "100%", "padding": "15px", "overflow": "hidden"},
                                ),
                            ],
                            style={"height": "100%", "display": "flex", "flexDirection": "column"},
                        ),
                        width=2,
                        style={"height": "calc(100vh - 135px)"},
                    ),
                ],
                className="g-4",
            )
        ],
        fluid=True,
        style={"paddingTop": "20px", "paddingBottom": "20px", "height": "calc(100vh - 95px)", "overflow": "hidden"},
    )

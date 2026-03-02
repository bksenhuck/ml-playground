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

from experiments.tracker import run_experiment_and_log


APP_TITLE = "ML Playground"


def serve_app() -> dash.Dash:
    """Create and return the Dash app."""
    app = dash.Dash(
        __name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True, title=APP_TITLE
    )
    server = app.server

    # prepare dataset features for the features dropdown at layout creation
    import seaborn as sns
    _df = sns.load_dataset("titanic")
    # Only exclude the direct target ("survived") and its string alias ("alive")
    # Let the user decide if they want to use redundant features (like 'class', 'who', 'alone')
    _EXCLUDED = {"survived", "alive"}
    _features = [c for c in _df.columns if c not in _EXCLUDED]
    _feature_opts = [{"label": f, "value": f} for f in _features]
    _feature_default = [f for f in ["age", "sex", "pclass", "fare", "embarked"] if f in _features]

    controls = dbc.Card(
        [
            dbc.CardHeader(
                html.Div(
                    [
                        html.Span("Pipeline Configuration", style={"fontWeight": "600", "fontSize": "0.95rem", "letterSpacing": "0.5px"}),
                        html.Span("?", id="tooltip-pipeline", style={"cursor": "help", "color": "#6c757d", "fontSize": "0.7rem", "border": "1px solid #adb5bd", "borderRadius": "50%", "width": "15px", "height": "15px", "display": "inline-flex", "alignItems": "center", "justifyContent": "center", "marginLeft": "6px", "flexShrink": "0"}),
                        dbc.Tooltip("Configure as features de entrada, transformações (scaling, polynomial) e hiperparâmetros do modelo. Clique em 'Run Experiment' para treinar e registrar um novo run.", target="tooltip-pipeline", placement="right"),
                    ],
                    className="d-flex align-items-center",
                ),
            ),
            dbc.CardBody(
                dbc.Accordion(
                    [
                        # ── Features ──────────────────────────────────────
                        dbc.AccordionItem(
                            [
                                # Sub-seção: Seleção Principal
                                html.Div([
                                    dbc.Label("Seleção de Features", className="fw-bold small mb-1"),
                                    dcc.Dropdown(
                                        id="feature-select",
                                        multi=True,
                                        options=_feature_opts,
                                        value=_feature_default,
                                        placeholder="Selecione as features de entrada...",
                                        style={"fontSize": "0.85rem"},
                                    ),
                                ], className="mb-3"),

                                # Sub-seção: Ações Rápidas (Exclusão/Limpeza)
                                html.Div([
                                    dbc.Label("Limpeza e Filtros", className="fw-bold small mb-1"),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Button("Remover Redundantes", id="remove-redundant-btn", outline=True, color="secondary", size="sm", class_name="w-100", style={"fontSize": "0.7rem"}),
                                            width=6,
                                        ),
                                        dbc.Col(
                                            dbc.Button("Limpar Tudo", id="clear-features-btn", outline=True, color="danger", size="sm", class_name="w-100", style={"fontSize": "0.7rem"}),
                                            width=6,
                                        ),
                                    ], className="g-1"),
                                    html.Small("Remove colunas como 'alive', 'class', 'alone' que duplicam informação.", className="text-muted", style={"fontSize": "0.65rem", "display": "block", "marginTop": "4px"}),
                                ], className="mb-3"),

                                # Sub-seção: Engenharia / Adição
                                html.Div([
                                    dbc.Label("Transformações", className="fw-bold small mb-1"),
                                    dcc.Checklist(
                                        id="poly-features",
                                        options=[{"label": " Adicionar Polynomial (grau 2)", "value": "enabled"}],
                                        value=[],
                                        inputStyle={"marginRight": "6px"},
                                        style={"fontSize": "0.85rem"},
                                    ),
                                ]),
                            ],
                            title="Features",
                            item_id="features",
                        ),
                        # ── Tratamentos ────────────────────────────────────
                        dbc.AccordionItem(
                            [
                                dbc.Label("Scaling Strategy", className="fw-semibold mb-1"),
                                dcc.RadioItems(
                                    id="scaling",
                                    options=[
                                        {"label": " Nenhum", "value": "none"},
                                        {"label": " StandardScaler", "value": "standard"},
                                    ],
                                    value="standard",
                                    inputStyle={"marginRight": "6px"},
                                    labelStyle={"display": "block", "marginBottom": "4px"},
                                ),
                                html.Hr(className="my-2"),
                                dbc.Label("Class Balancing", className="fw-semibold mb-1"),
                                dcc.RadioItems(
                                    id="class-weight",
                                    options=[
                                        {"label": " Nenhum", "value": "none"},
                                        {"label": " Balanced", "value": "balanced"},
                                    ],
                                    value="none",
                                    inputStyle={"marginRight": "6px"},
                                    labelStyle={"display": "block", "marginBottom": "4px"},
                                ),
                            ],
                            title="Tratamentos",
                            item_id="tratamentos",
                        ),
                        # ── Treinamento ────────────────────────────────────
                        dbc.AccordionItem(
                            [
                                dbc.Label("Train/Test Split (%)", className="fw-semibold mb-1"),
                                dcc.Slider(
                                    id="test-size-slider",
                                    min=10, max=50, step=5, value=20,
                                    marks={10: "10%", 20: "20%", 30: "30%", 40: "40%", 50: "50%"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                                html.Hr(className="my-2"),
                                dbc.Label("Modelo", className="fw-semibold mb-1"),
                                dcc.RadioItems(
                                    id="model-select",
                                    options=[
                                        {"label": " Logistic Regression", "value": "logreg"},
                                        {"label": " Random Forest", "value": "rf"},
                                        {"label": " XGBoost", "value": "xgb"},
                                    ],
                                    value="logreg",
                                    inputStyle={"marginRight": "6px"},
                                    labelStyle={"display": "block", "marginBottom": "4px"},
                                ),
                                html.Hr(className="my-2"),
                                dbc.Label("Cross-Validation", className="fw-semibold mb-1"),
                                dcc.Checklist(
                                    id="cv-enabled",
                                    options=[{"label": " K-Fold CV", "value": "enabled"}],
                                    value=[],
                                    inputStyle={"marginRight": "6px"},
                                ),
                                dbc.Label("k folds", className="small text-muted mt-1 mb-0"),
                                dcc.Slider(
                                    id="cv-folds",
                                    min=3, max=10, step=1, value=5,
                                    marks={3: "3", 5: "5", 10: "10"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                            title="Treinamento",
                            item_id="treinamento",
                        ),
                        # ── Parâmetros ─────────────────────────────────────
                        dbc.AccordionItem(
                            [
                                html.Div(
                                    id="hyperparams-area",
                                    children=[
                                        dbc.Label("C"),
                                        dcc.Slider(id="param-C", min=0.01, max=10.0, step=0.01, value=1.0),
                                    ],
                                ),
                            ],
                            title="Parâmetros",
                            item_id="parametros",
                        ),
                        # ── General ────────────────────────────────────────
                        dbc.AccordionItem(
                            [
                                dbc.Label("Nome do experimento", className="fw-semibold mb-1"),
                                dcc.Input(
                                    id="run-name",
                                    placeholder="opcional…",
                                    type="text",
                                    style={"width": "100%"},
                                    className="form-control form-control-sm",
                                ),
                            ],
                            title="General",
                            item_id="general",
                        ),
                    ],
                    id="pipeline-accordion",
                    always_open=True,
                    active_item=["features", "treinamento", "parametros"],
                    flush=True,
                    className="border-0",
                ),
                style={"overflowY": "auto", "flex": "1", "padding": "0"},
            ),
            dbc.CardFooter(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button("Run Experiment", id="run-btn", color="primary", class_name="w-100", size="sm"),
                            ),
                            dbc.Col(
                                dbc.Button("Delete All Experiments", id="delete-btn", color="danger", class_name="w-100", size="sm"),
                            ),
                        ],
                        className="g-2",
                    ),
                    html.Div(id="run-status", style={"display": "none"}),
                ],
                style={"backgroundColor": "#f8f9fa", "borderTop": "1px solid #dee2e6", "padding": "0.75rem"},
            ),
        ],
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )

    # initialize runs table data (empty by default for in-memory session)
    initial_runs = []

    runs_table = dash_table.DataTable(
        id="runs-table",
        columns=[{"name": (c.replace("metric_", "").replace("_", " ").title() if "metric_" in c else c.replace("_", " ").title()), "id": c, "type": "numeric", "format": {"specifier": ".2f"} if "metric_" in c else None}
                 for c in ["run_name", "model", "n_features", "C", "n_estimators", "max_depth", "scaling", "class_weight", "poly_features", "test_size",
                           "metric_accuracy", "metric_precision", "metric_recall", "metric_f1", "metric_roc_auc"]],
        data=[],
        row_selectable="multi",
        selected_rows=[],
        style_table={"overflowX": "auto", "minWidth": "100%"},
        style_cell={
            "textAlign": "center",
            "minWidth": "80px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_header={
            "backgroundColor": "rgb(230, 230, 230)",
            "fontWeight": "bold",
            "textAlign": "center"
        },
    )

    # ── Navbar ─────────────────────────────────────────────────────────────────
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    "ML PLAYGROUND",
                    href="/",
                    style={"fontWeight": "800", "letterSpacing": "2px", "fontSize": "1.1rem"},
                ),
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")),
                        dbc.NavItem(dbc.NavLink("Experimentos", href="/experimentos", active="partial")),
                        dbc.NavItem(dbc.NavLink("Datasets", href="/datasets", active="partial")),
                        dbc.NavItem(dbc.NavLink("Modelos", href="/modelos", active="partial")),
                        dbc.NavItem(dbc.NavLink("Sobre", href="/sobre", active="partial")),
                    ],
                    navbar=True,
                    className="ms-4",
                ),
                dbc.Col(
                    dbc.Badge("v1.0 MVP", color="light", text_color="primary", className="ms-auto px-3"),
                    width="auto",
                    className="ms-auto",
                ),
            ],
            fluid=True,
        ),
        dark=True,
        style={
            "backgroundColor": "#1a2a3a",
            "height": "60px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "zIndex": "1000",
        },
    )

    # ── Footer ─────────────────────────────────────────────────────────────────
    footer = html.Div(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(html.Small("© 2026 ML Playground. Built with Dash & Scikit-Learn.", className="text-muted")),
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

    # ── Page layout functions ──────────────────────────────────────────────────

    def welcome_layout():
        _card_style = {
            "border": "none",
            "boxShadow": "0 2px 12px rgba(0,0,0,0.10)",
            "cursor": "pointer",
            "height": "100%",
        }
        _cards = [
            {
                "title": "Experimentos",
                "icon": None,
                "desc": (
                    "Configure features, modelos e hiperparâmetros. "
                    "Execute experimentos e compare resultados com MLflow."
                ),
                "href": "/experimentos",
                "color": "#2980b9",
            },
            {
                "title": "Datasets",
                "icon": None,
                "desc": (
                    "Explore os datasets disponíveis, suas variáveis e "
                    "características antes de iniciar os experimentos."
                ),
                "href": "/datasets",
                "color": "#27ae60",
            },
            {
                "title": "Modelos",
                "icon": None,
                "desc": (
                    "Visualize e compare métricas dos modelos treinados: "
                    "accuracy, F1, ROC AUC, SHAP e muito mais."
                ),
                "href": "/modelos",
                "color": "#8e44ad",
            },
            {
                "title": "Sobre",
                "icon": None,
                "desc": (
                    "Saiba mais sobre o ML Playground, a stack tecnologica "
                    "e o roadmap de desenvolvimento."
                ),
                "href": "/sobre",
                "color": "#e67e22",
            },
        ]
        return dbc.Container(
            [
                dbc.Row(
                    dbc.Col([
                        html.H2(
                            "Bem-vindo ao ML Playground",
                            className="fw-bold mt-4 mb-1",
                        ),
                        html.P(
                            "Uma plataforma interativa para exploração e "
                            "experimentação com Machine Learning.",
                            className="text-muted mb-4 fs-5",
                        ),
                    ])
                ),
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4(
                                    "O que é o ML Playground?",
                                    className="mb-3",
                                ),
                                html.P(
                                    "Configure, execute e compare experimentos de Machine Learning "
                                    "de forma visual e interativa — sem escrever código. "
                                    "Escolha um módulo abaixo para começar.",
                                    className="lead mb-0",
                                ),
                            ]),
                            style={
                                "border": "none",
                                "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                                "borderLeft": "4px solid #2980b9",
                            },
                            className="mb-4",
                        )
                    )
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Link(
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H5(
                                            info["title"],
                                            className="fw-bold mb-2",
                                        ),
                                        html.P(
                                            info["desc"],
                                            className="text-muted mb-0",
                                            style={"fontSize": "0.9rem"},
                                        ),
                                        html.Div(
                                            "Acessar →",
                                            style={
                                                "marginTop": "1rem",
                                                "color": info["color"],
                                                "fontWeight": "600",
                                                "fontSize": "0.9rem",
                                            },
                                        ),
                                    ], style={"textAlign": "center", "padding": "1.5rem"}),
                                    style={
                                        **_card_style,
                                        "borderTop": f"4px solid {info['color']}",
                                    },
                                    className="h-100",
                                ),
                                href=info["href"],
                                style={"textDecoration": "none"},
                            ),
                            md=3,
                            className="mb-4",
                        )
                        for info in _cards
                    ],
                    className="mb-4",
                ),
            ],
            fluid=True,
            style={"paddingTop": "20px", "paddingBottom": "60px"},
        )

    def home_layout():
        return dbc.Container(
            [
                dbc.Row(
                    [
                        # Left: Config panel
                        dbc.Col(
                            html.Div(controls, style={"height": "100%", "display": "flex", "flexDirection": "column"}),
                            width=3,
                            style={"height": "calc(100vh - 135px)"}
                        ),
                        # Right: Results area
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Span("Experimentos", style={"fontWeight": "600", "fontSize": "0.9rem"}),
                                                            html.Span("?", id="tooltip-experiments", style={"cursor": "help", "color": "#6c757d", "fontSize": "0.7rem", "border": "1px solid #adb5bd", "borderRadius": "50%", "width": "15px", "height": "15px", "display": "inline-flex", "alignItems": "center", "justifyContent": "center", "marginLeft": "6px", "flexShrink": "0"}),
                                                            dbc.Tooltip("Histórico de runs registrados. Selecione linhas para visualizar métricas nos gráficos abaixo. Use as abas para alternar entre métricas de teste, treino, overfitting e validação cruzada.", target="tooltip-experiments", placement="right"),
                                                        ],
                                                        className="d-flex align-items-center mb-1",
                                                    ),
                                                    html.Div(
                                                        [
                                                            dbc.Tabs(
                                                                [
                                                                    dbc.Tab(label="Teste",         tab_id="tab-test"),
                                                                    dbc.Tab(label="Treino",        tab_id="tab-train"),
                                                                    dbc.Tab(label="Overfitting ↓", tab_id="tab-overfit"),
                                                                    dbc.Tab(label="Cross-Val",     tab_id="tab-cv"),
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
                                                ],
                                            ),
                                            style={"paddingBottom": "0"},
                                        ),
                                        dbc.CardBody(
                                            runs_table,
                                            style={"padding": "0", "height": "200px", "overflowY": "auto"},
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.Div(
                                                [
                                                    html.Span("Visualizations", style={"fontWeight": "600"}),
                                                    html.Span("?", id="tooltip-viz", style={"cursor": "help", "color": "#6c757d", "fontSize": "0.7rem", "border": "1px solid #adb5bd", "borderRadius": "50%", "width": "15px", "height": "15px", "display": "inline-flex", "alignItems": "center", "justifyContent": "center", "marginLeft": "6px", "flexShrink": "0"}),
                                                    dbc.Tooltip("Gráficos interativos gerados a partir dos runs selecionados na tabela acima. Troque o tipo de visualização pelo dropdown.", target="tooltip-viz", placement="right"),
                                                ],
                                                className="d-flex align-items-center",
                                            ),
                                        ),
                                        dbc.CardBody(
                                            [
                                                # ── Controls row: dropdown + inline description + SHAP selector ──
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
                                                        config={"displayModeBar": False}
                                                    ),
                                                    type="circle",
                                                    color="#2c7bb6",
                                                ),
                                            ],
                                            style={"display": "flex", "flexDirection": "column", "padding": "10px"}
                                        ),
                                    ],
                                    style={"flex": "1", "minHeight": "0"}
                                ),
                            ],
                            width=7,
                            style={"height": "calc(100vh - 135px)", "display": "flex", "flexDirection": "column"}
                        ),
                        # Far Right: AI Assistant Placeholder
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.Div(
                                            [
                                                html.Span("AI Insights Assistant", style={"fontWeight": "600"}),
                                                html.Span("?", id="tooltip-ai", style={"cursor": "help", "color": "#6c757d", "fontSize": "0.7rem", "border": "1px solid #adb5bd", "borderRadius": "50%", "width": "15px", "height": "15px", "display": "inline-flex", "alignItems": "center", "justifyContent": "center", "marginLeft": "6px", "flexShrink": "0"}),
                                                dbc.Tooltip("Assistente de IA que analisa os resultados dos seus experimentos. Selecione runs na tabela e faça perguntas sobre métricas, overfitting, comparação de modelos e muito mais.", target="tooltip-ai", placement="left"),
                                            ],
                                            className="d-flex align-items-center",
                                        ),
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Ask me anything about your experiment results or model performance.",
                                                        style={"fontSize": "0.85rem", "color": "#6c757d", "marginBottom": "15px"}
                                                    ),
                                                    dcc.Loading(
                                                        html.Div(
                                                            id="assistant-messages",
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
                                                                html.Div("Assistente pronto. Selecione runs na tabela e faça uma pergunta.",
                                                                         style={"fontSize": "0.8rem", "fontStyle": "italic", "color": "#adb5bd"})
                                                            ]
                                                        ),
                                                        type="circle",
                                                        color="#27ae60",
                                                    ),
                                                ],
                                                style={"display": "flex", "flexDirection": "column", "height": "calc(100% - 70px)"}
                                            ),
                                            dbc.InputGroup(
                                                [
                                                    dbc.Input(id="assistant-input", placeholder="Pergunte sobre seus experimentos...", type="text"),
                                                    dbc.Button("Enviar", id="assistant-send", color="primary"),
                                                ]
                                            )
                                        ],
                                        style={"display": "flex", "flexDirection": "column", "height": "100%", "padding": "15px"}
                                    ),
                                ],
                                style={"height": "100%", "display": "flex", "flexDirection": "column"}
                            ),
                            width=2,
                            style={"height": "calc(100vh - 135px)"}
                        ),
                    ],
                    className="g-4"
                )
            ],
            fluid=True,
            style={"paddingTop": "20px", "paddingBottom": "20px", "height": "calc(100vh - 95px)", "overflow": "hidden"}
        )

    def datasets_layout():
        card_style = {"border": "none", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}

        def var_table(rows):
            """Build a compact Bootstrap table for variable descriptions."""
            return dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Variável", style={"width": "15%"}),
                        html.Th("Tipo", style={"width": "15%"}),
                        html.Th("Descrição"),
                        html.Th("Valores / Range", style={"width": "25%"}),
                    ])),
                    html.Tbody([
                        html.Tr([html.Td(html.Code(r[0])), html.Td(r[1]), html.Td(r[2]), html.Td(r[3])])
                        for r in rows
                    ]),
                ],
                bordered=True, hover=True, responsive=True, size="sm",
                style={"fontSize": "0.82rem"},
            )

        titanic_vars = [
            ("pclass",      "Categórica", "Classe do bilhete",                      "1 = 1ª · 2 = 2ª · 3 = 3ª classe"),
            ("sex",         "Categórica", "Gênero do passageiro",                   "male, female"),
            ("age",         "Numérica",   "Idade em anos (contém nulos)",            "0.42 – 80"),
            ("sibsp",       "Numérica",   "Nº de irmãos / cônjuges a bordo",        "0 – 8"),
            ("parch",       "Numérica",   "Nº de pais / filhos a bordo",            "0 – 6"),
            ("fare",        "Numérica",   "Tarifa paga pela passagem (£)",           "0 – 512"),
            ("embarked",    "Categórica", "Porto de embarque",                       "C = Cherbourg · Q = Queenstown · S = Southampton"),
            ("class",       "Categórica", "Classe como string (alias de pclass)",   "First, Second, Third"),
            ("who",         "Categórica", "Categoria do passageiro",                "man, woman, child"),
            ("adult_male",  "Booleana",   "Adulto do sexo masculino",               "True, False"),
            ("deck",        "Categórica", "Deck do camarote (muitos nulos)",        "A, B, C, D, E, F, G"),
            ("embark_town", "Categórica", "Cidade de embarque (alias de embarked)", "Cherbourg, Queenstown, Southampton"),
            ("alone",       "Booleana",   "Passageiro viajou sozinho",              "True, False"),
        ]

        housing_vars = [
            ("MedInc",      "Numérica", "Renda mediana dos residentes do bloco",      "0.5 – 15.0  (×$10 k)"),
            ("HouseAge",    "Numérica", "Idade mediana das casas no bloco",           "1 – 52 anos"),
            ("AveRooms",    "Numérica", "Média de cômodos por residência",            "0.8 – 141"),
            ("AveBedrms",   "Numérica", "Média de quartos por residência",            "0.3 – 34"),
            ("Population",  "Numérica", "Total de pessoas no bloco",                 "3 – 35 682"),
            ("AveOccup",    "Numérica", "Média de ocupantes por residência",          "0.7 – 1 243"),
            ("Latitude",    "Numérica", "Latitude geográfica do bloco",              "32.6 – 41.9 °N"),
            ("Longitude",   "Numérica", "Longitude geográfica do bloco",             "−124.3 – −114.3 °W"),
        ]

        diabetes_vars = [
            ("age",  "Numérica", "Idade do paciente (normalizada)",                  "−0.11 – 0.11"),
            ("sex",  "Numérica", "Gênero (normalizado)",                             "−0.04 – 0.06"),
            ("bmi",  "Numérica", "Índice de Massa Corporal (normalizado)",           "−0.09 – 0.18"),
            ("bp",   "Numérica", "Pressão arterial média (normalizada)",             "−0.11 – 0.13"),
            ("s1",   "Numérica", "Colesterol total — tc (normalizado)",              "−0.13 – 0.15"),
            ("s2",   "Numérica", "LDL colesterol — ldl (normalizado)",               "−0.12 – 0.20"),
            ("s3",   "Numérica", "HDL colesterol — hdl (normalizado)",               "−0.10 – 0.18"),
            ("s4",   "Numérica", "Colesterol total / HDL — tch (normalizado)",       "−0.08 – 0.19"),
            ("s5",   "Numérica", "Log do nível sérico — ltg (normalizado)",          "−0.13 – 0.13"),
            ("s6",   "Numérica", "Glicose sérica — glu (normalizado)",               "−0.11 – 0.13"),
        ]

        return dbc.Container(
            [
                dbc.Row(
                    dbc.Col([
                        html.H2("Datasets", className="fw-bold mt-4 mb-1"),
                        html.P(
                            "Conheça os datasets disponíveis e planejados na plataforma — "
                            "com a descrição de cada variável, target e métricas de avaliação.",
                            className="text-muted mb-4",
                        ),
                    ])
                ),

                # ── Classificação ────────────────────────────────────────
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        dbc.Badge("Disponível", color="success", className="me-2"),
                                        html.Span("Classificação — Titanic", className="fw-bold fs-5"),
                                    ],
                                    style={"backgroundColor": "#eaf4fb", "borderBottom": "2px solid #2980b9"}
                                ),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H6("O que é?", className="text-primary fw-bold"),
                                            html.P(
                                                "Prever se um passageiro sobreviveu ao naufrágio do Titanic "
                                                "(variável binária). Ideal para explorar classificação supervisionada.",
                                            ),
                                            html.H6("Resumo do dataset", className="text-primary fw-bold mt-3"),
                                            dbc.ListGroup([
                                                dbc.ListGroupItem([html.Strong("Registros: "), "891 passageiros"]),
                                                dbc.ListGroupItem([html.Strong("Target: "), html.Code("survived"), " — 0 = não sobreviveu · 1 = sobreviveu"]),
                                                dbc.ListGroupItem([html.Strong("Origem: "), "seaborn.load_dataset('titanic')"]),
                                            ], flush=True, className="mb-3"),
                                        ], md=4),
                                        dbc.Col([
                                            html.H6("Modelos Suportados", className="text-primary fw-bold"),
                                            dbc.Row([
                                                dbc.Col(dbc.Card(dbc.CardBody("Logistic Regression"), color="primary", outline=True, className="text-center mb-2")),
                                                dbc.Col(dbc.Card(dbc.CardBody("Random Forest"), color="primary", outline=True, className="text-center mb-2")),
                                                dbc.Col(dbc.Card(dbc.CardBody("XGBoost"), color="primary", outline=True, className="text-center mb-2")),
                                            ], className="mb-3"),
                                            html.H6("Métricas de Avaliação", className="text-primary fw-bold"),
                                            html.Div([
                                                dbc.Badge("Accuracy", color="primary", className="me-1 mb-1"),
                                                dbc.Badge("Precision", color="primary", className="me-1 mb-1"),
                                                dbc.Badge("Recall", color="primary", className="me-1 mb-1"),
                                                dbc.Badge("F1-Score", color="primary", className="me-1 mb-1"),
                                                dbc.Badge("ROC AUC", color="primary", className="me-1 mb-1"),
                                            ]),
                                        ], md=8),
                                    ], className="mb-3"),
                                    html.H6("Variáveis (features disponíveis para seleção)", className="text-primary fw-bold"),
                                    var_table(titanic_vars),
                                ]),
                            ],
                            style=card_style, className="mb-4",
                        )
                    )
                ),

                # ── Regressão ────────────────────────────────────────────
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        dbc.Badge("Em breve", color="warning", text_color="dark", className="me-2"),
                                        html.Span("Regressão", className="fw-bold fs-5"),
                                    ],
                                    style={"backgroundColor": "#fef9ec", "borderBottom": "2px solid #f39c12"}
                                ),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H6("O que é?", className="text-warning fw-bold"),
                                            html.P(
                                                "Prever um valor numérico contínuo. "
                                                "O modelo aprende a estimar quantidades a partir das features.",
                                            ),
                                        ], md=4),
                                        dbc.Col([
                                            html.H6("Modelos Planejados", className="text-warning fw-bold"),
                                            dbc.Row([
                                                dbc.Col(dbc.Card(dbc.CardBody("Linear Regression"), color="warning", outline=True, className="text-center mb-2")),
                                                dbc.Col(dbc.Card(dbc.CardBody("RF Regressor"), color="warning", outline=True, className="text-center mb-2")),
                                            ], className="mb-2"),
                                            html.H6("Métricas de Avaliação", className="text-warning fw-bold"),
                                            html.Div([
                                                dbc.Badge("MAE", color="warning", text_color="dark", className="me-1 mb-1"),
                                                dbc.Badge("MSE", color="warning", text_color="dark", className="me-1 mb-1"),
                                                dbc.Badge("RMSE", color="warning", text_color="dark", className="me-1 mb-1"),
                                                dbc.Badge("R²", color="warning", text_color="dark", className="me-1 mb-1"),
                                            ]),
                                        ], md=8),
                                    ], className="mb-4"),

                                    # California Housing
                                    html.H6(
                                        [dbc.Badge("Planejado", color="secondary", className="me-2"), "California Housing — Previsão de preço de imóveis"],
                                        className="fw-bold mb-1",
                                    ),
                                    html.P([
                                        html.Strong("Target: "), html.Code("MedHouseVal"), " — valor mediano das casas do bloco (×$100 k)",
                                        html.Span(" · ", className="text-muted"),
                                        html.Strong("Registros: "), "20 640",
                                    ], className="text-muted small mb-2"),
                                    var_table(housing_vars),

                                    html.Hr(),

                                    # Diabetes
                                    html.H6(
                                        [dbc.Badge("Planejado", color="secondary", className="me-2"), "Diabetes — Progressão da doença"],
                                        className="fw-bold mb-1 mt-3",
                                    ),
                                    html.P([
                                        html.Strong("Target: "), "progressão da diabetes após 1 ano (0 – 346)",
                                        html.Span(" · ", className="text-muted"),
                                        html.Strong("Registros: "), "442 · todas as variáveis já normalizadas",
                                    ], className="text-muted small mb-2"),
                                    var_table(diabetes_vars),
                                ]),
                            ],
                            style=card_style, className="mb-5",
                        )
                    )
                ),
            ],
            fluid=True,
            style={"paddingTop": "20px", "paddingBottom": "60px"},
        )

    def models_layout():
        card_style = {"border": "none", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}

        def param_table(rows):
            return dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Parâmetro", style={"width": "18%"}),
                        html.Th("Padrão", style={"width": "12%"}),
                        html.Th("Range / Valores"),
                        html.Th("O que faz"),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(html.Code(r[0])),
                            html.Td(html.Code(str(r[1]))),
                            html.Td(r[2]),
                            html.Td(r[3]),
                        ])
                        for r in rows
                    ]),
                ],
                bordered=True, hover=True, responsive=True, size="sm",
                style={"fontSize": "0.82rem"},
            )

        logreg_params = [
            (
                "C",
                1.0,
                "0.01 – 10.0",
                "Inverso da força de regularização. C alto = menos regularização. "
                "C baixo = mais regularização (modelo mais simples).",
            ),
            (
                "penalty",
                "l2",
                "l1 (Lasso) ou l2 (Ridge)",
                "Tipo de penalização aplicada. L1 pode zerar coeficientes (seleção de features), "
                "enquanto L2 apenas os diminui.",
            ),
        ]
        rf_params = [
            (
                "n_estimators",
                100,
                "10 – 500",
                "Número de árvores na floresta. Mais árvores costumam melhorar o modelo, "
                "mas aumentam o tempo de treino.",
            ),
            (
                "max_depth",
                6,
                "1 – 30",
                "Profundidade máxima de cada árvore. Árvores muito profundas "
                "podem causar overfitting.",
            ),
            (
                "min_samples_split",
                2,
                "2 – 20",
                "Número mínimo de amostras necessárias para dividir um nó interno.",
            ),
        ]
        xgb_params = [
            (
                "n_estimators",
                100,
                "10 – 500",
                "Número de rounds de boosting (árvores). Mais rounds aumentam a complexidade.",
            ),
            (
                "max_depth",
                6,
                "1 – 12",
                "Profundidade máxima das árvores. No boosting, profundidades menores (3-6) "
                "são comuns.",
            ),
            (
                "learning_rate",
                0.1,
                "0.01 – 0.5",
                "Passo de aprendizado. Valores baixos exigem mais n_estimators.",
            ),
            (
                "subsample",
                1.0,
                "0.5 – 1.0",
                "Fração de amostras usadas para treinar cada árvore. Ajuda a evitar overfitting.",
            ),
        ]

        def model_card(
            title, badge_text, badge_color, accent, summary, algo_html, params
        ):
            return dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            dbc.Badge(
                                badge_text, color=badge_color, className="me-2"
                            ),
                            html.Span(title, className="fw-bold fs-5"),
                        ],
                        style={
                            "borderBottom": f"2px solid {accent}",
                            "backgroundColor": "#f8f9fa",
                        },
                    ),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Como funciona?",
                                        className="fw-bold",
                                        style={"color": accent}),
                                html.P(summary, className="mb-3"),
                                algo_html,
                            ], md=4),
                            dbc.Col([
                                html.H6("Hiperparâmetros disponíveis",
                                        className="fw-bold",
                                        style={"color": accent}),
                                param_table(params),
                            ], md=8),
                        ]),
                    ]),
                ],
                style=card_style,
                className="mb-4",
            )

        logreg_algo = html.Div([
            dbc.ListGroup([
                dbc.ListGroupItem([html.Strong("Tipo: "), "Classificação linear"]),
                dbc.ListGroupItem([html.Strong("Interpretável: "), "Sim — coeficientes por feature"]),
                dbc.ListGroupItem([html.Strong("Regularização: "), "L2 (Ridge) por padrão"]),
                dbc.ListGroupItem([html.Strong("Probabilidades: "), "Sim (sigmoid)"]),
            ], flush=True, className="mb-2"),
            html.Small(
                "Ideal como baseline. Funciona bem com features normalizadas. "
                "Sensível a features irrelevantes ou correlacionadas.",
                className="text-muted",
            ),
        ])

        rf_algo = html.Div([
            dbc.ListGroup([
                dbc.ListGroupItem([html.Strong("Tipo: "), "Ensemble de árvores (Bagging)"]),
                dbc.ListGroupItem([html.Strong("Interpretável: "), "Parcialmente — feature importance"]),
                dbc.ListGroupItem([html.Strong("Regularização: "), "Via max_depth e min_samples"]),
                dbc.ListGroupItem([html.Strong("Probabilidades: "), "Sim (média das árvores)"]),
            ], flush=True, className="mb-2"),
            html.Small(
                "Robusto a outliers e features não normalizadas. "
                "Lida bem com features de diferentes escalas. "
                "Paralelizável.",
                className="text-muted",
            ),
        ])

        xgb_algo = html.Div([
            dbc.ListGroup([
                dbc.ListGroupItem([html.Strong("Tipo: "), "Ensemble de árvores (Boosting)"]),
                dbc.ListGroupItem([html.Strong("Interpretável: "), "Parcialmente — SHAP nativo"]),
                dbc.ListGroupItem([html.Strong("Regularização: "), "L1 + L2 integradas"]),
                dbc.ListGroupItem([html.Strong("Probabilidades: "), "Sim (sigmoid na saída)"]),
            ], flush=True, className="mb-2"),
            html.Small(
                "Geralmente supera o Random Forest em benchmarks tabulares. "
                "Constrói árvores sequencialmente, cada uma corrigindo os erros "
                "da anterior. Suporta dados faltantes nativamente.",
                className="text-muted",
            ),
        ])

        return dbc.Container(
            [
                dbc.Row(
                    dbc.Col([
                        html.H2("Modelos", className="fw-bold mt-4 mb-1"),
                        html.P(
                            "Referência dos algoritmos disponíveis na plataforma — "
                            "como funcionam, quando usar e o que cada hiperparâmetro controla.",
                            className="text-muted mb-4",
                        ),
                    ])
                ),
                model_card(
                    "Logistic Regression", "Classificação", "primary", "#2980b9",
                    "Aprende um hiperplano linear que separa as classes. "
                    "A saída passa por uma função sigmoid, produzindo probabilidades entre 0 e 1. "
                    "Simples, rápido e muito interpretável — ótimo ponto de partida.",
                    logreg_algo, logreg_params,
                ),
                model_card(
                    "Random Forest", "Classificação", "success", "#27ae60",
                    "Treina centenas de árvores de decisão em subconjuntos aleatórios dos dados "
                    "(bootstrap) e features. A predição final é a média das probabilidades de "
                    "todas as árvores, reduzindo variância sem aumentar viés.",
                    rf_algo, rf_params,
                ),
                model_card(
                    "XGBoost", "Classificação", "warning", "#e67e22",
                    "Implementa Gradient Boosting com árvores de decisão. "
                    "Cada árvore é treinada para corrigir os erros residuais das anteriores, "
                    "usando gradiente descendente no espaço de funções. "
                    "Inclui regularização L1/L2 nativa e é altamente eficiente.",
                    xgb_algo, xgb_params,
                ),
            ],
            fluid=True,
            style={"paddingTop": "20px", "paddingBottom": "60px"},
        )

    def about_layout():
        step_card_style = {"border": "none", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "textAlign": "center"}
        return dbc.Container(
            [
                dbc.Row(
                    dbc.Col([
                        html.H2("Sobre o ML Playground", className="fw-bold mt-4 mb-1"),
                        html.P(
                            "Uma plataforma interativa para exploração e experimentação com Machine Learning.",
                            className="text-muted mb-4 fs-5",
                        ),
                    ])
                ),
                # Hero description
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H4("O que é o ML Playground?", className="mb-3"),
                                html.P(
                                    "O ML Playground é uma plataforma web que permite configurar, executar e comparar "
                                    "experimentos de Machine Learning de forma visual e interativa — sem escrever código. "
                                    "O objetivo é acelerar o entendimento de como diferentes modelos, hiperparâmetros e "
                                    "estratégias de pré-processamento impactam os resultados.",
                                    className="lead mb-0",
                                ),
                            ]),
                            style={"border": "none", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "borderLeft": "4px solid #2980b9"},
                            className="mb-4",
                        )
                    )
                ),
                # How it works — 4 steps
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(dbc.CardBody([
                                html.H5("1. Configure", className="fw-bold"),
                                html.P("Selecione features, modelo, hiperparâmetros, scaling e proporção de treino/teste no painel lateral."),
                            ]), style=step_card_style),
                            md=3, className="mb-3",
                        ),
                        dbc.Col(
                            dbc.Card(dbc.CardBody([
                                html.H5("2. Execute", className="fw-bold"),
                                html.P("Clique em 'Run Experiment'. O pipeline é montado, treinado e avaliado automaticamente."),
                            ]), style=step_card_style),
                            md=3, className="mb-3",
                        ),
                        dbc.Col(
                            dbc.Card(dbc.CardBody([
                                html.H5("3. Rastreie", className="fw-bold"),
                                html.P("Cada experimento é registrado em memória com parâmetros, métricas e o modelo treinado."),
                            ]), style=step_card_style),
                            md=3, className="mb-3",
                        ),
                        dbc.Col(
                            dbc.Card(dbc.CardBody([
                                html.H5("4. Compare", className="fw-bold"),
                                html.P("Selecione runs na tabela e visualize comparações em gráficos de radar e barras."),
                            ]), style=step_card_style),
                            md=3, className="mb-3",
                        ),
                    ],
                    className="mb-4",
                ),
                # Tech stack
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.Span("Stack Tecnológico", className="fw-bold")),
                                dbc.CardBody(
                                    dbc.Row([
                                        dbc.Col([
                                            html.H6("Frontend & Visualização", className="text-muted fw-bold"),
                                            dbc.ListGroup([
                                                dbc.ListGroupItem("Dash (Python web framework)"),
                                                dbc.ListGroupItem("Dash Bootstrap Components"),
                                                dbc.ListGroupItem("Plotly (gráficos interativos)"),
                                            ], flush=True),
                                        ], md=4),
                                        dbc.Col([
                                            html.H6("Machine Learning", className="text-muted fw-bold"),
                                            dbc.ListGroup([
                                                dbc.ListGroupItem("scikit-learn (modelos e pipelines)"),
                                                dbc.ListGroupItem("pandas & numpy (dados)"),
                                                dbc.ListGroupItem("seaborn (datasets)"),
                                            ], flush=True),
                                        ], md=4),
                                        dbc.Col([
                                            html.H6("Rastreamento & Deploy", className="text-muted fw-bold"),
                                            dbc.ListGroup([
                                                dbc.ListGroupItem("In-memory Session Tracking"),
                                                dbc.ListGroupItem("GCS / Cloud Run ready"),
                                                dbc.ListGroupItem("Docker (containerização)"),
                                            ], flush=True),
                                        ], md=4),
                                    ])
                                ),
                            ],
                            style={"border": "none", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"},
                            className="mb-4",
                        )
                    )
                ),
                # Roadmap
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.Span("Roadmap", className="fw-bold")),
                                dbc.CardBody(
                                    dbc.Row([
                                        dbc.Col([
                                            html.H6("Em desenvolvimento", className="text-muted fw-bold"),
                                            dbc.ListGroup([
                                                dbc.ListGroupItem([dbc.Badge("soon", color="warning", className="me-2"), "Dataset de Regressão"]),
                                                dbc.ListGroupItem([dbc.Badge("soon", color="warning", className="me-2"), "AI Insights Assistant"]),
                                                dbc.ListGroupItem([dbc.Badge("soon", color="warning", className="me-2"), "Upload de CSV customizado"]),
                                            ], flush=True),
                                        ], md=6),
                                        dbc.Col([
                                            html.H6("Planejado", className="text-muted fw-bold"),
                                            dbc.ListGroup([
                                                dbc.ListGroupItem([dbc.Badge("v2", color="secondary", className="me-2"), "Mais modelos (XGBoost, SVM)"]),
                                                dbc.ListGroupItem([dbc.Badge("v2", color="secondary", className="me-2"), "Exportação de relatórios"]),
                                                dbc.ListGroupItem([dbc.Badge("v2", color="secondary", className="me-2"), "AutoML básico"]),
                                            ], flush=True),
                                        ], md=6),
                                    ])
                                ),
                            ],
                            style={"border": "none", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"},
                            className="mb-5",
                        )
                    )
                ),
            ],
            fluid=True,
            style={"paddingTop": "20px", "paddingBottom": "60px"},
        )

    # ── App layout (multi-page) ────────────────────────────────────────────────
    app.layout = html.Div(
        [
            html.Div(id="dummy-output", style={"display": "none"}),
            dcc.Store(id="runs-data-store", data=initial_runs),
            dcc.Location(id="url", refresh=False),
            navbar,
            html.Div(id="page-content"),
            footer,
        ],
        style={
            "backgroundColor": "#f0f2f5",
            "minHeight": "100vh",
        }
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def render_page_content(pathname):
        if pathname == "/experimentos":
            return home_layout()
        elif pathname == "/datasets":
            return datasets_layout()
        elif pathname == "/modelos":
            return models_layout()
        elif pathname == "/sobre":
            return about_layout()
        return welcome_layout()

    @app.callback(
        Output("feature-select", "value"),
        Input("remove-redundant-btn", "n_clicks"),
        Input("clear-features-btn", "n_clicks"),
        Input("feature-select", "id"),
        State("feature-select", "value"),
    )
    def manage_features(remove_clicks, clear_clicks, _, current_features):
        ctx = callback_context
        if not ctx.triggered:
            return _feature_default

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == "feature-select":
            return _feature_default

        if triggered_id == "clear-features-btn":
            return []

        if triggered_id == "remove-redundant-btn":
            redundant = {"alive", "class", "who", "adult_male", "embark_town", "alone"}
            current = current_features or []
            return [f for f in current if f not in redundant]

        return no_update

    @app.callback(
        Output("feature-select", "options"),
        Input("feature-select", "id"),
    )
    def populate_feature_options(_):
        return _feature_opts

    # ── Runs table view: switch columns + enrich data based on active tab ──────
    _METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    _INFO_COLS = ["run_name", "model", "n_features", "scaling"]

    @app.callback(
        Output("runs-table", "data"),
        Output("runs-table", "columns"),
        Output("runs-table", "style_data_conditional"),
        Input("runs-table-tabs", "active_tab"),
        Input("runs-data-store", "data"),
    )
    def render_runs_table(active_tab, raw_data):
        def _col(c):
            base = c.replace("_", " ")
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
                name = base.title()
            is_num = "metric_" in c or "delta_" in c
            return {
                "name": name, "id": c,
                "type": "numeric" if is_num else "text",
                "format": {"specifier": ".3f"} if is_num else None,
            }

        data = list(raw_data or [])

        if active_tab == "tab-train":
            cols = _INFO_COLS + [f"metric_train_{m}" for m in _METRICS]
            return data, [_col(c) for c in cols], []

        if active_tab == "tab-overfit":
            for row in data:
                for m in _METRICS:
                    t  = float(row.get(f"metric_{m}", 0) or 0)
                    tr = float(row.get(f"metric_train_{m}", 0) or 0)
                    row[f"delta_{m}"] = round(tr - t, 4)
            cols = ["run_name", "model"] + [f"delta_{m}" for m in _METRICS]
            _RED    = {"backgroundColor": "#f8d7da", "color": "#721c24"}
            _YELLOW = {"backgroundColor": "#fff3cd", "color": "#856404"}
            _GREEN  = {"backgroundColor": "#d4edda", "color": "#155724"}
            style_cond = []
            for col in [f"delta_{m}" for m in _METRICS]:
                style_cond += [
                    {"if": {"filter_query": f"{{{col}}} >= 0.15",
                            "column_id": col}, **_RED},
                    {"if": {"filter_query": f"{{{col}}} >= 0.05 && {{{col}}} < 0.15",
                            "column_id": col}, **_YELLOW},
                    {"if": {"filter_query": f"{{{col}}} < 0.05",
                            "column_id": col}, **_GREEN},
                ]
            return data, [_col(c) for c in cols], style_cond

        if active_tab == "tab-cv":
            cols = (
                ["run_name", "model"]
                + [f"metric_cv_mean_{m}" for m in _METRICS]
                + [f"metric_cv_std_{m}" for m in _METRICS]
            )
            return data, [_col(c) for c in cols], []

        # Default: tab-test
        cols = (
            ["run_name", "model", "n_features", "C", "n_estimators",
             "max_depth", "scaling", "class_weight", "poly_features", "test_size"]
            + [f"metric_{m}" for m in _METRICS]
        )
        return data, [_col(c) for c in cols], []

    @app.callback(
        Output("runs-table", "selected_rows"),
        Input("select-all-btn", "n_clicks"),
        State("runs-data-store", "data"),
        State("runs-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def toggle_select_all(_, data, current):
        n = len(data or [])
        if n == 0:
            return []
        if len(current or []) == n:
            return []
        return list(range(n))

    @app.callback(Output("hyperparams-area", "children"), Input("model-select", "value"))
    def render_hyperparams(model: str):
        if model == "logreg":
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
                html.Div(id="param-n", style={"display": "none"}),
                html.Div(id="param-d", style={"display": "none"}),
                html.Div(id="param-lr", style={"display": "none"}),
                html.Div(id="param-subsample", style={"display": "none"}),
            ]
        if model == "xgb":
            return [
                dbc.Label("n_estimators (Nº de árvores)"),
                dcc.Slider(id="param-n", min=10, max=500, step=10, value=100),
                dbc.Label("max_depth (Profundidade máxima)"),
                dcc.Slider(id="param-d", min=1, max=12, step=1, value=6),
                dbc.Label("learning_rate (Taxa de aprendizado)"),
                dcc.Slider(id="param-lr", min=0.01, max=0.5, step=0.01, value=0.1),
                dbc.Label("subsample (Amostragem linhas)"),
                dcc.Slider(id="param-subsample", min=0.5, max=1.0, step=0.1, value=1.0),
                html.Div(id="param-C", style={"display": "none"}),
                html.Div(id="param-penalty", style={"display": "none"}),
            ]
        # Random Forest
        return [
            dbc.Label("n_estimators (Nº de árvores)"),
            dcc.Slider(id="param-n", min=10, max=500, step=10, value=100),
            dbc.Label("max_depth (Profundidade máxima)"),
            dcc.Slider(id="param-d", min=1, max=30, step=1, value=6),
            dbc.Label("min_samples_split"),
            dcc.Slider(id="param-min-split", min=2, max=20, step=1, value=2),
            html.Div(id="param-C", style={"display": "none"}),
            html.Div(id="param-lr", style={"display": "none"}),
            html.Div(id="param-penalty", style={"display": "none"}),
            html.Div(id="param-subsample", style={"display": "none"}),
        ]

    @app.callback(
        Output("run-status", "children"),
        Output("runs-data-store", "data"),
        Input("run-btn", "n_clicks"),
        Input("delete-btn", "n_clicks"),
        State("runs-data-store", "data"),
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
    def handle_experiment_actions(run_clicks: int, delete_clicks: int, current_data: List[dict], features: List[str], scaling: str, class_weight: str, poly_features: List[str], test_size: int, model: str, run_name: str | None, hyper_children, cv_enabled, cv_folds_val):
        """Unified callback for all experiment actions (Run/Delete) to avoid duplicate output conflicts."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        data = current_data or []

        if triggered_id == "delete-btn":
            return html.Div(["All runs deleted from view."]), []

        # Handle Run Action
        params = {}
        poly_enabled = "enabled" in (poly_features or [])
        test_size_float = float(test_size) / 100.0

        def _extract_val(children, target_id, default=None):
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

        if model == "logreg":
            c_val = _extract_val(hyper_children, "param-C", 1.0)
            penalty = _extract_val(hyper_children, "param-penalty", "l2")
            params["C"] = float(c_val if c_val is not None else 1.0)
            params["penalty"] = "l1" if penalty == "l1" else "l2"
            if params["penalty"] == "l1":
                params["solver"] = "liblinear" # L1 needs liblinear or saga
        elif model == "xgb":
            n_val = _extract_val(hyper_children, "param-n", 100)
            d_val = _extract_val(hyper_children, "param-d", 6)
            lr_val = _extract_val(hyper_children, "param-lr", 0.1)
            sub_val = _extract_val(hyper_children, "param-subsample", 1.0)
            params["n_estimators"] = int(n_val if n_val is not None else 100)
            params["max_depth"] = int(d_val if d_val is not None else 6)
            params["learning_rate"] = float(lr_val if lr_val is not None else 0.1)
            params["subsample"] = float(sub_val if sub_val is not None else 1.0)
        else:
            n_val = _extract_val(hyper_children, "param-n", 100)
            d_val = _extract_val(hyper_children, "param-d", 6)
            min_split = _extract_val(hyper_children, "param-min-split", 2)
            params["n_estimators"] = int(n_val if n_val is not None else 100)
            params["max_depth"] = int(d_val if d_val is not None else 6)
            params["min_samples_split"] = int(min_split if min_split is not None else 2)

        cv_k = int(cv_folds_val or 5) if "enabled" in (cv_enabled or []) else 0

        # run_experiment_and_log returns (run_id, row_dict)
        run_id, run_result = run_experiment_and_log(
            features or [],
            scaling,
            model,
            params,
            run_name=run_name,
            test_size=test_size_float,
            class_weight=class_weight,
            poly_features=poly_enabled,
            cv_folds=cv_k,
        )

        if isinstance(run_result, dict):
            data.append(run_result)
        else:
            # Reconstruct if it was just metrics (should not happen with updated tracker)
            pass

        return html.Div([f"Last run: {run_id}"]), data

    @app.callback(
        Output("bottom-chart", "figure"),
        Input("runs-data-store", "data"),
        Input("runs-table", "selected_rows"),
        Input("chart-select-bottom", "value"),
        Input("runs-table-tabs", "active_tab"),
        Input("shap-feature-select", "value"),
    )
    def update_chart(
        data,
        selected_rows: List[int],
        chart_select_bottom: str,
        active_tab: str,
        shap_feature: str,
    ):
        from app.plots import _empty_fig  # noqa: PLC0415

        df = pd.DataFrame(data or [])
        if df.empty:
            return px.line_polar()

        chart_type = chart_select_bottom or "radar"
        mode = "train" if active_tab == "tab-train" else "test"

        # ── Metric column selection (test vs train) ────────────────────────────
        if mode == "train":
            internal_metrics = [
                "metric_train_accuracy", "metric_train_precision",
                "metric_train_recall", "metric_train_f1", "metric_train_roc_auc",
            ]
            display_metrics = [
                "Train Acc", "Train Prec", "Train Recall", "Train F1", "Train AUC",
            ]
        else:
            internal_metrics = [
                "metric_accuracy", "metric_precision",
                "metric_recall", "metric_f1", "metric_roc_auc",
            ]
            display_metrics = ["Accuracy", "Precision", "Recall", "F1", "Roc Auc"]

        # Always require explicit row selection — no automatic fallback.
        if not selected_rows:
            return _empty_fig(
                "Selecione ao menos um experimento na tabela para ver o gráfico"
            )
        # Guard against stale indices when store is refreshed or tabs switch
        valid_rows = [i for i in selected_rows if i < len(df)]
        if not valid_rows:
            return _empty_fig(
                "Selecione ao menos um experimento na tabela para ver o gráfico"
            )
        sel = df.iloc[valid_rows]

        # ── Bar F1 (respects metric-mode) ─────────────────────────────────────
        if chart_type == "bar_f1":
            y_col = (
                "metric_train_f1" if mode == "train" else "metric_f1"
            )
            if y_col not in df.columns:
                y_col = "metric_f1" if "metric_f1" in df.columns else "f1"
            if y_col not in df.columns:
                df[y_col] = 0.0
            label = "Train F1" if mode == "train" else "F1"
            fig = px.bar(
                sel.sort_values(y_col, ascending=False) if y_col in sel.columns else sel,
                x="run_name" if "run_name" in sel.columns else sel.index,
                y=y_col,
                color="model" if "model" in sel.columns else None,
            )
            fig.update_layout(
                legend={"orientation": "v", "x": 1.02, "xanchor": "left", "y": 1, "yanchor": "top"},
                margin={"t": 15, "b": 40, "l": 50, "r": 150},
            )
            return fig

        # ── Existing chart types (delegated to app/plots.py) ──────────────────
        if chart_type in (
            "roc", "pr_curve", "confusion_matrix",
            "feature_importance", "calibration", "metric_dist",
        ):
            from app.plots import (  # noqa: PLC0415
                plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
                plot_feature_importance, plot_calibration_curve,
                plot_metric_distribution,
            )
            # metric_dist: all visible rows; model-heavy plots: selected only
            runs_records = (
                df.to_dict("records")
                if chart_type == "metric_dist"
                else sel.to_dict("records")
            )
            if chart_type == "roc":
                return plot_roc_curve(runs_records)
            if chart_type == "pr_curve":
                return plot_pr_curve(runs_records)
            if chart_type == "confusion_matrix":
                return plot_confusion_matrix(runs_records)
            if chart_type == "feature_importance":
                return plot_feature_importance(runs_records)
            if chart_type == "calibration":
                return plot_calibration_curve(runs_records)
            if chart_type == "metric_dist":
                return plot_metric_distribution(runs_records)

        # ── SHAP chart types ───────────────────────────────────────────────────
        if chart_type in ("shap_summary", "shap_dependence"):
            from app.plots import plot_shap_summary, plot_shap_dependence  # noqa: PLC0415
            runs_records = sel.to_dict("records")
            if chart_type == "shap_summary":
                return plot_shap_summary(runs_records)
            return plot_shap_dependence(runs_records, shap_feature)

        # ── Default: radar chart (existing logic — respects metric-mode) ───────
        available_internal = [m for m in internal_metrics if m in df.columns]
        available_display = [
            display_metrics[internal_metrics.index(m)] for m in available_internal
        ]

        if not available_internal:
            return px.line_polar()

        fig = px.line_polar()
        for _, r in sel.iterrows():
            values = [float(r.get(m, 0) or 0) for m in available_internal]
            label = r.get("run_name") or r.get("name") or r.get("run_id") or "Run"
            theta = available_display + [available_display[0]]
            fig.add_scatterpolar(
                r=values + [values[0]], theta=theta,
                name=str(label), fill="toself",
            )

        suffix = " (Train)" if mode == "train" else ""
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            legend={"orientation": "v", "x": 1.02, "xanchor": "left", "y": 1, "yanchor": "top"},
            margin={"t": 15, "b": 20, "l": 20, "r": 150},
        )
        return fig

    # ── chart description — update text when chart type changes ────────────────
    @app.callback(
        Output("chart-info-text", "children"),
        Input("chart-select-bottom", "value"),
    )
    def update_chart_info(chart_type: str) -> str:
        from app.plot_help import PLOT_DESCRIPTIONS  # noqa: PLC0415
        return PLOT_DESCRIPTIONS.get(chart_type or "radar", "")

    # ── SHAP feature selector — show & populate when chart = shap_dependence ──
    @app.callback(
        Output("shap-feature-select", "options"),
        Output("shap-feature-select", "style"),
        Input("chart-select-bottom", "value"),
        Input("runs-data-store", "data"),
        Input("runs-table", "selected_rows"),
    )
    def update_shap_selector(chart_type, data, selected_rows):
        base = {"width": "190px", "fontSize": "0.82rem"}
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

        opts = [{"label": f, "value": f} for f in features]
        return opts, visible

    # ── LLM Insights Assistant callback ───────────────────────────────────────
    @app.callback(
        Output("assistant-messages", "children"),
        Output("assistant-input", "value"),
        Input("assistant-send", "n_clicks"),
        State("assistant-input", "value"),
        State("runs-data-store", "data"),
        State("runs-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def generate_llm_insight(n_clicks, question, table_data, selected_rows):
        """Hybrid LLM assistant: Gemini when available, local engine as fallback."""
        if not question or not question.strip():
            return no_update, no_update

        # Require explicit row selection — guide the user to pick experiments
        if not selected_rows:
            tip = html.Div(
                "Selecione ao menos um experimento na tabela para obter insights.",
                style={"fontSize": "0.82rem", "fontStyle": "italic", "color": "#adb5bd"},
            )
            return tip, no_update

        runs_df    = pd.DataFrame(table_data or [])
        context_df = runs_df.iloc[selected_rows] if not runs_df.empty else runs_df

        from llm.service import generate_insight
        answer, source = generate_insight(question.strip(), context_df)
        label = "Assistente(LLM)" if source == "llm" else "Assistente(Local)"

        messages = [
            html.Div(
                [html.Strong("Você: ", style={"color": "#2980b9"}), question.strip()],
                className="mb-2",
                style={"fontSize": "0.82rem"},
            ),
            html.Div(
                [html.Strong(f"{label}: ", style={"color": "#27ae60"}), answer],
                className="mb-1",
                style={"fontSize": "0.82rem", "whiteSpace": "pre-wrap"},
            ),
        ]
        return messages, ""  # clear input field after sending

    return app


# Module-level initialisation — required for gunicorn entry point (app.app:server)
_dash_app = serve_app()
server = _dash_app.server  # gunicorn target

if __name__ == "__main__":
    _dash_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)

"""Shared layout pieces: navbar, footer, pipeline-controls card, runs DataTable."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from app.components import section_header, tooltip_icon
from app.data_config import TITANIC_DEFAULT, TITANIC_OPTS


# ── Navbar ────────────────────────────────────────────────────────────────────

def make_navbar() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    "ML PLAYGROUND",
                    href="/",
                    style={"fontWeight": "800", "letterSpacing": "2px", "fontSize": "1.1rem"},
                ),
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Home",         href="/",            active="exact")),
                        dbc.NavItem(dbc.NavLink("Experimentos", href="/experimentos", active="partial")),
                        dbc.NavItem(dbc.NavLink("Datasets",     href="/datasets",     active="partial")),
                        dbc.NavItem(dbc.NavLink("Modelos",      href="/modelos",      active="partial")),
                        dbc.NavItem(dbc.NavLink("Sobre",        href="/sobre",        active="partial")),
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
        sticky="top",
        style={
            "backgroundColor": "#1a2a3a",
            "height": "60px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "zIndex": "1030",
        },
    )


# ── Footer ────────────────────────────────────────────────────────────────────

def make_footer() -> html.Div:
    return html.Div(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(html.Small(
                        "© 2026 ML Playground. Built with Dash & Scikit-Learn.",
                        className="text-muted",
                    )),
                    dbc.Col(
                        html.Div(
                            [
                                html.Small("Backend: ", className="text-muted me-2"),
                                dbc.Badge("Live", color="success", pill=True, style={"fontSize": "0.6rem"}),
                            ],
                            className="text-end",
                        ),
                        width="auto",
                    ),
                ],
                align="center",
            ),
            fluid=True,
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
            "zIndex": "1000",
        },
    )


# ── Pipeline Controls card (left sidebar) ─────────────────────────────────────

def make_controls() -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(
                html.Div(
                    [
                        html.Span(
                            "Pipeline Configuration",
                            style={"fontWeight": "600", "fontSize": "0.95rem", "letterSpacing": "0.5px"},
                        ),
                    ]
                    + tooltip_icon(
                        "tooltip-pipeline",
                        "Configure as features de entrada, transformações (scaling, polynomial) e "
                        "hiperparâmetros do modelo. Clique em 'Run Experiment' para treinar e "
                        "registrar um novo run.",
                    ),
                    className="d-flex align-items-center",
                ),
            ),
            dbc.CardBody(
                dbc.Accordion(
                    [
                        # Dataset
                        dbc.AccordionItem(
                            [
                                html.Div([
                                    dbc.Label("Dataset", className="fw-bold small mb-1"),
                                    dcc.Dropdown(
                                        id="dataset-select",
                                        options=[
                                            {"label": "🚢 Titanic (Classificação)", "value": "titanic"},
                                            {"label": "🏠 California Housing (Regressão)", "value": "housing"},
                                        ],
                                        value="titanic",
                                        clearable=False,
                                        style={"fontSize": "0.85rem"},
                                    ),
                                ]),
                            ],
                            title="Dataset",
                            item_id="dataset-accordion",
                        ),

                        # Análise Exploratória
                        dbc.AccordionItem(
                            [
                                html.Div([
                                    dbc.Label("Dataset Visualization Options", className="fw-bold small mb-1"),
                                    html.P(
                                        "Configure as visualizações de análise exploratória aqui.",
                                        className="text-muted small",
                                    ),
                                ]),
                            ],
                            title="Análise Exploratória",
                            item_id="viz-accordion",
                        ),

                        # Features
                        dbc.AccordionItem(
                            [
                                html.Div([
                                    dbc.Label("Seleção de Features", className="fw-bold small mb-1"),
                                    dcc.Dropdown(
                                        id="feature-select",
                                        multi=True,
                                        options=TITANIC_OPTS,
                                        value=TITANIC_DEFAULT,
                                        placeholder="Selecione as features de entrada...",
                                        style={"fontSize": "0.85rem"},
                                    ),
                                ], className="mb-3"),
                                html.Div([
                                    dbc.Label("Limpeza e Filtros", className="fw-bold small mb-1"),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Button(
                                                "Remover Redundantes",
                                                id="remove-redundant-btn",
                                                outline=True, color="secondary", size="sm",
                                                class_name="w-100", style={"fontSize": "0.7rem"},
                                            ),
                                            width=6,
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                "Limpar Tudo",
                                                id="clear-features-btn",
                                                outline=True, color="danger", size="sm",
                                                class_name="w-100", style={"fontSize": "0.7rem"},
                                            ),
                                            width=6,
                                        ),
                                    ], className="g-1"),
                                    html.Small(
                                        "Remove colunas como 'alive', 'class', 'alone' que duplicam informação.",
                                        className="text-muted",
                                        style={"fontSize": "0.65rem", "display": "block", "marginTop": "4px"},
                                    ),
                                ], className="mb-3"),
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

                        # Tratamentos
                        dbc.AccordionItem(
                            [
                                dbc.Label("Scaling Strategy", className="fw-semibold mb-1"),
                                dcc.RadioItems(
                                    id="scaling",
                                    options=[
                                        {"label": " Nenhum",          "value": "none"},
                                        {"label": " StandardScaler",  "value": "standard"},
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
                                        {"label": " Nenhum",   "value": "none"},
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

                        # Treinamento
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
                                        {"label": " Random Forest",       "value": "rf"},
                                        {"label": " XGBoost",             "value": "xgb"},
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

                        # Parâmetros
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

                        # General
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
                            dbc.Col(dbc.Button(
                                "Run Experiment",
                                id="run-btn", color="primary", class_name="w-100", size="sm",
                            )),
                            dbc.Col(dbc.Button(
                                "Delete All Experiments",
                                id="delete-btn", color="danger", class_name="w-100", size="sm",
                            )),
                        ],
                        className="g-2",
                    ),
                    html.Div(id="run-status", style={"display": "none"}),
                ],
                style={
                    "backgroundColor": "#f8f9fa",
                    "borderTop": "1px solid #dee2e6",
                    "padding": "0.75rem",
                },
            ),
        ],
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )


# ── Runs DataTable ─────────────────────────────────────────────────────────────

def make_runs_table() -> dash_table.DataTable:
    _all_cols = [
        "run_name", "model", "n_features", "C", "n_estimators", "max_depth",
        "scaling", "class_weight", "poly_features", "test_size",
        "metric_accuracy", "metric_precision", "metric_recall",
        "metric_f1", "metric_roc_auc",
    ]

    def _col_def(c: str) -> dict:
        if "metric_" in c:
            name = c.replace("metric_", "").replace("_", " ").title()
        else:
            name = c.replace("_", " ").title()
        is_num = "metric_" in c
        return {
            "name": name, "id": c,
            "type": "numeric" if is_num else "text",
            "format": {"specifier": ".2f"} if is_num else None,
        }

    return dash_table.DataTable(
        id="runs-table",
        columns=[_col_def(c) for c in _all_cols],
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
            "textAlign": "center",
        },
    )

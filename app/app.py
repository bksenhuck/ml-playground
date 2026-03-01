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
                    html.Div(id="hyperparams-area", children=(
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
                        dbc.NavItem(dbc.NavLink("Experimentos", href="/", active="exact")),
                        dbc.NavItem(dbc.NavLink("Datasets", href="/datasets", active="partial")),
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

    # ── Page layout functions ──────────────────────────────────────────────────

    def home_layout():
        return dbc.Container(
            [
                dbc.Row(
                    [
                        # Left: Config panel
                        dbc.Col(
                            html.Div(controls, style={"height": "100%", "display": "flex", "flexDirection": "column"}),
                            width=3,
                            style={"height": "calc(100vh - 110px)"}
                        ),
                        # Right: Results area
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
                        # Far Right: AI Assistant Placeholder
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
            style={"paddingTop": "20px", "height": "calc(100vh - 95px)", "overflow": "hidden"}
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
                                html.P("Cada experimento é salvo no MLflow com parâmetros, métricas e o modelo treinado."),
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
                                                dbc.ListGroupItem("MLflow (experiment tracking)"),
                                                dbc.ListGroupItem("SQLite (backend local)"),
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
        if pathname == "/datasets":
            return datasets_layout()
        elif pathname == "/sobre":
            return about_layout()
        return home_layout()

    @app.callback(
        Output("feature-select", "options"),
        Output("feature-select", "value"),
        Input("feature-select", "id"),
    )
    def populate_features(_):
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

        chart_type = chart_select_bottom or "radar"

        internal_metrics = ["metric_accuracy", "metric_precision", "metric_recall", "metric_f1", "metric_roc_auc"]
        display_metrics = ["Accuracy", "Precision", "Recall", "F1", "Roc Auc"]

        if selected_rows:
            sel = df.iloc[selected_rows]
        else:
            sort_col = "metric_f1" if "metric_f1" in df.columns else (df.columns[0] if not df.empty else None)
            if sort_col and not df.empty:
                sel = df.nlargest(min(3, len(df)), sort_col)
            else:
                sel = df.head(min(3, len(df)))

        if chart_type == "bar_f1":
            y_col = "metric_f1" if "metric_f1" in df.columns else "f1"
            if y_col not in df.columns:
                df[y_col] = 0.0

            plot_df = sel if selected_rows else df
            fig = px.bar(plot_df.sort_values(y_col, ascending=False) if y_col in plot_df.columns else plot_df,
                         x="name" if "name" in plot_df.columns else plot_df.index,
                         y=y_col, color="model" if "model" in plot_df.columns else None,
                         title="F1 by run")
            return fig

        available_internal = [m for m in internal_metrics if m in df.columns]
        available_display = [display_metrics[internal_metrics.index(m)] for m in available_internal]

        if not available_internal:
            return px.line_polar()

        fig = px.line_polar()
        for _, r in sel.iterrows():
            values = [float(r.get(m, 0) or 0) for m in available_internal]
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
    dash_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=True)

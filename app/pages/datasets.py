"""Datasets reference page layout."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

_CARD_STYLE = {"border": "none", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}


def _var_table(rows: list) -> dbc.Table:
    """Build a compact Bootstrap table for variable descriptions."""
    return dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("Variável",        style={"width": "15%"}),
                html.Th("Tipo",            style={"width": "15%"}),
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


_TITANIC_VARS = [
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

_HOUSING_VARS = [
    ("MedInc",     "Numérica", "Renda mediana dos residentes do bloco",    "0.5 – 15.0  (×$10 k)"),
    ("HouseAge",   "Numérica", "Idade mediana das casas no bloco",         "1 – 52 anos"),
    ("AveRooms",   "Numérica", "Média de cômodos por residência",          "0.8 – 141"),
    ("AveBedrms",  "Numérica", "Média de quartos por residência",          "0.3 – 34"),
    ("Population", "Numérica", "Total de pessoas no bloco",               "3 – 35 682"),
    ("AveOccup",   "Numérica", "Média de ocupantes por residência",        "0.7 – 1 243"),
    ("Latitude",   "Numérica", "Latitude geográfica do bloco",            "32.6 – 41.9 °N"),
    ("Longitude",  "Numérica", "Longitude geográfica do bloco",           "−124.3 – −114.3 °W"),
]

_DIABETES_VARS = [
    ("age", "Numérica", "Idade do paciente (normalizada)",               "−0.11 – 0.11"),
    ("sex", "Numérica", "Gênero (normalizado)",                          "−0.04 – 0.06"),
    ("bmi", "Numérica", "Índice de Massa Corporal (normalizado)",        "−0.09 – 0.18"),
    ("bp",  "Numérica", "Pressão arterial média (normalizada)",          "−0.11 – 0.13"),
    ("s1",  "Numérica", "Colesterol total — tc (normalizado)",           "−0.13 – 0.15"),
    ("s2",  "Numérica", "LDL colesterol — ldl (normalizado)",            "−0.12 – 0.20"),
    ("s3",  "Numérica", "HDL colesterol — hdl (normalizado)",            "−0.10 – 0.18"),
    ("s4",  "Numérica", "Colesterol total / HDL — tch (normalizado)",    "−0.08 – 0.19"),
    ("s5",  "Numérica", "Log do nível sérico — ltg (normalizado)",       "−0.13 – 0.13"),
    ("s6",  "Numérica", "Glicose sérica — glu (normalizado)",            "−0.11 – 0.13"),
]


def datasets_layout() -> dbc.Container:
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

            # ── Classificação ────────────────────────────────────────────────
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    dbc.Badge("Disponível", color="success", className="me-2"),
                                    html.Span("Classificação — Titanic", className="fw-bold fs-5"),
                                ],
                                style={"backgroundColor": "#eaf4fb", "borderBottom": "2px solid #2980b9"},
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
                                            dbc.Col(dbc.Card(dbc.CardBody("Random Forest"),       color="primary", outline=True, className="text-center mb-2")),
                                            dbc.Col(dbc.Card(dbc.CardBody("XGBoost"),             color="primary", outline=True, className="text-center mb-2")),
                                        ], className="mb-3"),
                                        html.H6("Métricas de Avaliação", className="text-primary fw-bold"),
                                        html.Div([
                                            dbc.Badge("Accuracy",  color="primary", className="me-1 mb-1"),
                                            dbc.Badge("Precision", color="primary", className="me-1 mb-1"),
                                            dbc.Badge("Recall",    color="primary", className="me-1 mb-1"),
                                            dbc.Badge("F1-Score",  color="primary", className="me-1 mb-1"),
                                            dbc.Badge("ROC AUC",   color="primary", className="me-1 mb-1"),
                                        ]),
                                    ], md=8),
                                ], className="mb-3"),
                                html.H6("Variáveis (features disponíveis para seleção)", className="text-primary fw-bold"),
                                _var_table(_TITANIC_VARS),
                            ]),
                        ],
                        style=_CARD_STYLE, className="mb-4",
                    )
                )
            ),

            # ── Regressão ────────────────────────────────────────────────────
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    dbc.Badge("Em breve", color="warning", text_color="dark", className="me-2"),
                                    html.Span("Regressão", className="fw-bold fs-5"),
                                ],
                                style={"backgroundColor": "#fef9ec", "borderBottom": "2px solid #f39c12"},
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
                                            dbc.Col(dbc.Card(dbc.CardBody("RF Regressor"),      color="warning", outline=True, className="text-center mb-2")),
                                        ], className="mb-2"),
                                        html.H6("Métricas de Avaliação", className="text-warning fw-bold"),
                                        html.Div([
                                            dbc.Badge("MAE",  color="warning", text_color="dark", className="me-1 mb-1"),
                                            dbc.Badge("MSE",  color="warning", text_color="dark", className="me-1 mb-1"),
                                            dbc.Badge("RMSE", color="warning", text_color="dark", className="me-1 mb-1"),
                                            dbc.Badge("R²",   color="warning", text_color="dark", className="me-1 mb-1"),
                                        ]),
                                    ], md=8),
                                ], className="mb-4"),

                                html.H6(
                                    [dbc.Badge("Ativo", color="success", className="me-2"),
                                     "California Housing — Previsão de preço de imóveis"],
                                    className="fw-bold mb-1",
                                ),
                                html.P([
                                    html.Strong("Target: "), html.Code("MedHouseVal"),
                                    " — valor mediano das casas do bloco (×$100 k)",
                                    html.Span(" · ", className="text-muted"),
                                    html.Strong("Registros: "), "20 640",
                                ], className="text-muted small mb-2"),
                                _var_table(_HOUSING_VARS),

                                html.Hr(),

                                html.H6(
                                    [dbc.Badge("Planejado", color="secondary", className="me-2"),
                                     "Diabetes — Progressão da doença"],
                                    className="fw-bold mb-1 mt-3",
                                ),
                                html.P([
                                    html.Strong("Target: "), "progressão da diabetes após 1 ano (0 – 346)",
                                    html.Span(" · ", className="text-muted"),
                                    html.Strong("Registros: "), "442 · todas as variáveis já normalizadas",
                                ], className="text-muted small mb-2"),
                                _var_table(_DIABETES_VARS),
                            ]),
                        ],
                        style=_CARD_STYLE, className="mb-5",
                    )
                )
            ),
        ],
        fluid=True,
        style={"paddingTop": "20px", "paddingBottom": "60px"},
    )

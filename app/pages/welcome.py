"""Welcome / home page layout."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

_CARD_STYLE = {
    "border": "none",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.10)",
    "cursor": "pointer",
    "height": "100%",
}

_NAV_CARDS = [
    {
        "title": "Experimentos",
        "desc": (
            "Configure features, modelos e hiperparâmetros. "
            "Execute treinamentos e analise os resultados em tempo real."
        ),
        "href": "/experimentos",
        "color": "#2980b9",
    },
    {
        "title": "Datasets",
        "desc": (
            "Explore os datasets disponíveis, suas variáveis e "
            "características antes de iniciar os experimentos."
        ),
        "href": "/datasets",
        "color": "#27ae60",
    },
    {
        "title": "Modelos",
        "desc": (
            "Visualize e compare métricas dos modelos treinados: "
            "accuracy, F1, ROC AUC, SHAP e muito mais."
        ),
        "href": "/modelos",
        "color": "#8e44ad",
    },
    {
        "title": "Sobre",
        "desc": (
            "Saiba mais sobre o ML Playground, a stack tecnologica "
            "e o roadmap de desenvolvimento."
        ),
        "href": "/sobre",
        "color": "#e67e22",
    },
]


_STEPS_CARDS = [
    {
        "step": "1",
        "title": "Configure",
        "desc": "Escolha as colunas, o escalonamento e o algoritmo de classificação.",
        "icon": "bi bi-sliders",
        "color": "#3498db"
    },
    {
        "step": "2",
        "title": "Execute",
        "desc": "Ajuste os hiperparâmetros e inicie o treinamento do modelo.",
        "icon": "bi bi-play-fill",
        "color": "#e67e22"
    },
    {
        "step": "3",
        "title": "Compare",
        "desc": "Analise as métricas e artefatos gerados pelo treinamento.",
        "icon": "bi bi-bar-chart-fill",
        "color": "#27ae60"
    },
    {
        "step": "4",
        "title": "Interprete",
        "desc": "Use o SHAP e a IA para entender as decisões do seu modelo.",
        "icon": "bi bi-cpu-fill",
        "color": "#9b59b6"
    }
]

def welcome_layout() -> dbc.Container:
    return dbc.Container(
        [
            # --- SEÇÃO HERO ---
            dbc.Row(
                dbc.Col([
                    html.Div(
                        [
                            html.H1("Bem-vindo ao ML Playground", className="fw-bold mb-2 text-center", style={"color": "#1a2a3a", "fontSize": "2.8rem"}),
                            html.Div(style={"width": "100px", "height": "5px", "backgroundColor": "#2980b9", "margin": "0 auto 25px"}),
                            html.P(
                                "Ume ecossistema interativo projetado para acelerar a experimentação e "
                                "a compreensão técnica de Machine Learning.",
                                className="text-muted mb-5 fs-5 text-center px-lg-5 mx-lg-5",
                                style={"maxWidth": "800px", "marginLeft": "auto", "marginRight": "auto"}
                            ),
                        ],
                        className="mt-5 pb-4"
                    ),
                ])
            ),

            # --- SEÇÃO DE PASSOS (ESTILO TIMELINE/PROCESS) ---
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.Div(
                                    [
                                        html.Div(
                                            step["step"],
                                            style={
                                                "fontSize": "4rem",
                                                "fontWeight": "900",
                                                "color": "rgba(41, 128, 185, 0.05)",
                                                "position": "absolute",
                                                "top": "-10px",
                                                "right": "10px",
                                                "lineHeight": "1",
                                                "userSelect": "none"
                                            }
                                        ),
                                        html.H5(step["title"], className="fw-bold mb-3 d-flex align-items-center justify-content-center", style={"color": "#1a2a3a", "position": "relative"}),
                                        html.P(
                                            step["desc"],
                                            className="text-muted mb-0 text-center",
                                            style={"fontSize": "0.88rem", "position": "relative", "lineHeight": "1.5"},
                                        ),
                                    ],
                                    style={"position": "relative", "zIndex": "1"}
                                ),
                            ]),
                            style={
                                "border": "none",
                                "borderBottom": "3px solid #e9ecef",
                                "backgroundColor": "#fff",
                                "height": "100%",
                                "transition": "all 0.3s ease",
                                "boxShadow": "0 4px 12px rgba(0,0,0,0.03)"
                            },
                        ),
                        width=12, sm=6, lg=3,
                        className="mb-4"
                    ) for step in _STEPS_CARDS
                ],
                className="mb-4 g-4"
            ),

            # --- SEÇÃO DE CHAMADA PARA AÇÃO (CTA) ---
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H3("Comece sua Jornada", className="fw-bold mb-2"),
                            html.P("Escolha um módulo abaixo para explorar o conjunto de dados ou arquitetar novos modelos.", className="mb-0 text-muted"),
                        ],
                        className="text-center py-4 rounded-3 bg-light mb-5 border-0"
                    )
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Link(
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5(info["title"], className="fw-bold mb-2"),
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
                                style={**_CARD_STYLE, "borderTop": f"4px solid {info['color']}"},
                                className="h-100",
                            ),
                            href=info["href"],
                            style={"textDecoration": "none"},
                        ),
                        md=3,
                        className="mb-4",
                    )
                    for info in _NAV_CARDS
                ],
                className="mb-4",
            ),
        ],
        fluid=True,
        style={"paddingTop": "20px", "paddingBottom": "60px"},
    )

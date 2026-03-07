"""Models reference page layout."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

_CARD_STYLE = {"border": "none", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}


def _param_table(rows: list) -> dbc.Table:
    return dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("Parâmetro",                   style={"width": "14%"}),
                html.Th("Padrão",                      style={"width": "9%"}),
                html.Th("Range / Valores",             style={"width": "14%"}),
                html.Th("O que faz",                   style={"width": "30%"}),
                html.Th("Expectativa ao mudar o valor",style={"width": "33%"}),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(html.Code(r[0])),
                    html.Td(html.Code(str(r[1]))),
                    html.Td(r[2]),
                    html.Td(r[3]),
                    html.Td(r[4], style={"color": "#5a6474", "fontStyle": "italic"}),
                ])
                for r in rows
            ]),
        ],
        bordered=True, hover=True, responsive=True, size="sm",
        style={"fontSize": "0.82rem"},
    )


_LOGREG_PARAMS = [
    (
        "C", 1.0, "0.01 – 10.0",
        "Inverso da força de regularização. C alto = menos regularização. "
        "C baixo = mais regularização (modelo mais simples).",
        "Valores altos reduzem a penalização e aumentam o risco de overfitting. "
        "Valores baixos simplificam o modelo e podem causar underfitting.",
    ),
    (
        "penalty", "l2", "l1 · l2",
        "Tipo de penalização. L1 pode zerar coeficientes (seleção de features), "
        "L2 apenas os diminui.",
        "L1 gera esparsidade: alguns coeficientes vão a zero, selecionando features automaticamente. "
        "L2 mantém todos os coeficientes ativos, mas os reduz proporcionalmente.",
    ),
]

_RF_PARAMS = [
    (
        "n_estimators", 100, "10 – 500",
        "Número de árvores na floresta. Mais árvores costumam melhorar o modelo, "
        "mas aumentam o tempo de treino.",
        "Valores maiores melhoram a generalização e estabilidade das predições. "
        "Dificilmente causa overfitting, mas o treino fica mais lento.",
    ),
    (
        "max_depth", 6, "1 – 30",
        "Profundidade máxima de cada árvore. Árvores muito profundas "
        "podem causar overfitting.",
        "Valores altos deixam as árvores mais complexas e aumentam o risco de overfitting. "
        "Valores baixos simplificam demais e podem causar underfitting.",
    ),
    (
        "min_samples_split", 2, "2 – 20",
        "Mínimo de amostras para dividir um nó interno.",
        "Valores maiores tornam os nós mais conservadores, resultando em árvores mais rasas "
        "e menos propensas a overfitting.",
    ),
]

_XGB_PARAMS = [
    (
        "n_estimators", 100, "10 – 500",
        "Número de rounds de boosting. Cada round adiciona uma árvore para corrigir erros.",
        "Valores maiores aumentam a capacidade do modelo. Sem early stopping, "
        "valores muito altos podem causar overfitting.",
    ),
    (
        "max_depth", 6, "1 – 12",
        "Profundidade máxima das árvores. No boosting, valores menores (3–6) "
        "são recomendados.",
        "Valores altos tornam as árvores mais expressivas e aumentam o risco de overfitting. "
        "Valores entre 3 e 5 costumam ser o ponto ideal.",
    ),
    (
        "learning_rate", 0.1, "0.01 – 0.5",
        "Passo de aprendizado (shrinkage). Escala a contribuição de cada árvore.",
        "Valores baixos tornam o aprendizado mais cauteloso e estável, "
        "mas exigem mais n_estimators para compensar. "
        "Valores altos aprendem mais rápido, mas com maior risco de instabilidade.",
    ),
    (
        "subsample", 1.0, "0.5 – 1.0",
        "Fração de amostras usadas por árvore (sem reposição).",
        "Valores menores introduzem aleatoriedade e funcionam como regularização, "
        "reduzindo overfitting. Valores muito baixos aumentam a variância das predições.",
    ),
]

_LOGREG_ALGO = html.Div([
    dbc.ListGroup([
        dbc.ListGroupItem([html.Strong("Tipo: "),          "Classificação linear"]),
        dbc.ListGroupItem([html.Strong("Interpretável: "), "Sim — coeficientes por feature"]),
        dbc.ListGroupItem([html.Strong("Regularização: "), "L2 (Ridge) por padrão"]),
        dbc.ListGroupItem([html.Strong("Probabilidades: "),"Sim (sigmoid)"]),
    ], flush=True, className="mb-2"),
    html.Small(
        "Ideal como baseline. Funciona bem com features normalizadas. "
        "Sensível a features irrelevantes ou correlacionadas.",
        className="text-muted",
    ),
])

_RF_ALGO = html.Div([
    dbc.ListGroup([
        dbc.ListGroupItem([html.Strong("Tipo: "),          "Ensemble de árvores (Bagging)"]),
        dbc.ListGroupItem([html.Strong("Interpretável: "), "Parcialmente — feature importance"]),
        dbc.ListGroupItem([html.Strong("Regularização: "), "Via max_depth e min_samples"]),
        dbc.ListGroupItem([html.Strong("Probabilidades: "),"Sim (média das árvores)"]),
    ], flush=True, className="mb-2"),
    html.Small(
        "Robusto a outliers e features não normalizadas. "
        "Lida bem com features de diferentes escalas. Paralelizável.",
        className="text-muted",
    ),
])

_XGB_ALGO = html.Div([
    dbc.ListGroup([
        dbc.ListGroupItem([html.Strong("Tipo: "),          "Ensemble de árvores (Boosting)"]),
        dbc.ListGroupItem([html.Strong("Interpretável: "), "Parcialmente — SHAP nativo"]),
        dbc.ListGroupItem([html.Strong("Regularização: "), "L1 + L2 integradas"]),
        dbc.ListGroupItem([html.Strong("Probabilidades: "),"Sim (sigmoid na saída)"]),
    ], flush=True, className="mb-2"),
    html.Small(
        "Geralmente supera o Random Forest em benchmarks tabulares. "
        "Constrói árvores sequencialmente, cada uma corrigindo os erros "
        "da anterior. Suporta dados faltantes nativamente.",
        className="text-muted",
    ),
])


def _model_card(title, badge_text, badge_color, accent, summary, algo_html, params):
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    dbc.Badge(badge_text, color=badge_color, className="me-2"),
                    html.Span(title, className="fw-bold fs-5"),
                ],
                style={"borderBottom": f"2px solid {accent}", "backgroundColor": "#f8f9fa"},
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Como funciona?", className="fw-bold", style={"color": accent}),
                        html.P(summary, className="mb-3"),
                        algo_html,
                    ], md=4),
                    dbc.Col([
                        html.H6("Hiperparâmetros disponíveis", className="fw-bold", style={"color": accent}),
                        _param_table(params),
                    ], md=8),
                ]),
            ]),
        ],
        style=_CARD_STYLE,
        className="mb-4",
    )


def models_layout() -> dbc.Container:
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
            _model_card(
                "Logistic Regression", "Classificação", "primary", "#2980b9",
                "Aprende um hiperplano linear que separa as classes. "
                "A saída passa por uma função sigmoid, produzindo probabilidades entre 0 e 1. "
                "Simples, rápido e muito interpretável — ótimo ponto de partida.",
                _LOGREG_ALGO, _LOGREG_PARAMS,
            ),
            _model_card(
                "Random Forest", "Classificação", "success", "#27ae60",
                "Treina centenas de árvores de decisão em subconjuntos aleatórios dos dados "
                "(bootstrap) e features. A predição final é a média das probabilidades de "
                "todas as árvores, reduzindo variância sem aumentar viés.",
                _RF_ALGO, _RF_PARAMS,
            ),
            _model_card(
                "XGBoost", "Classificação", "warning", "#e67e22",
                "Implementa Gradient Boosting com árvores de decisão. "
                "Cada árvore é treinada para corrigir os erros residuais das anteriores, "
                "usando gradiente descendente no espaço de funções. "
                "Inclui regularização L1/L2 nativa e é altamente eficiente.",
                _XGB_ALGO, _XGB_PARAMS,
            ),
        ],
        fluid=True,
        style={"paddingTop": "20px", "paddingBottom": "60px"},
    )

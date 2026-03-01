"""UI component helpers for the Dash app (kept small)."""
from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html


def card(title: str, body: Any):
    return dbc.Card([dbc.CardHeader(title), dbc.CardBody(body)])
"""Componentes de UI reutilizáveis — barra lateral do ML Playground."""

import dash_bootstrap_components as dbc
from dash import dcc, html


def sidebar(features: list[str]) -> dbc.Col:
    """Renderiza a barra lateral de configuração do pipeline.

    Inclui controles para:
    - Dropdown multi-seleção de variáveis
    - Toggle de normalização (Nenhuma / StandardScaler)
    - Seletor de modelo (Regressão Logística / Floresta Aleatória)
    - Sliders de hiperparâmetros por modelo
    - Campo de nome da corrida
    - Botão "Executar Experimento"

    Args:
        features: Lista de nomes de colunas para popular o dropdown.

    Returns:
        Um ``dbc.Col`` com todos os controles de configuração.
    """
    feature_options = [{"label": f, "value": f} for f in features]

    return dbc.Col(
        [
            html.H5("Configuração do Pipeline", className="fw-bold mb-3"),

            # ── Seleção de variáveis ──────────────────────────────────────
            html.Label("Variáveis", className="form-label fw-semibold"),
            dcc.Dropdown(
                id="feature-selector",
                options=feature_options,
                value=features,
                multi=True,
                placeholder="Selecione variáveis…",
                className="mb-3",
            ),

            # ── Normalização ──────────────────────────────────────────────
            html.Label("Normalização", className="form-label fw-semibold"),
            dbc.RadioItems(
                id="scaler-selector",
                options=[
                    {"label": "Nenhuma", "value": "none"},
                    {"label": "StandardScaler", "value": "standard"},
                ],
                value="none",
                className="mb-3",
            ),

            # ── Modelo ────────────────────────────────────────────────────
            html.Label("Modelo", className="form-label fw-semibold"),
            dbc.RadioItems(
                id="model-selector",
                options=[
                    {"label": "Regressão Logística", "value": "logistic_regression"},
                    {"label": "Floresta Aleatória", "value": "random_forest"},
                ],
                value="logistic_regression",
                className="mb-3",
            ),

            # ── Hiperparâmetros — Regressão Logística ─────────────────────
            html.Div(
                id="lr-params",
                children=[
                    html.Label(
                        "C  (força de regularização)",
                        className="form-label fw-semibold",
                    ),
                    dcc.Slider(
                        id="lr-C",
                        min=0.01,
                        max=10.0,
                        step=0.01,
                        value=1.0,
                        marks={0.01: "0,01", 1: "1", 5: "5", 10: "10"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                className="mb-3",
            ),

            # ── Hiperparâmetros — Floresta Aleatória ──────────────────────
            html.Div(
                id="rf-params",
                children=[
                    html.Label(
                        "n_estimadores",
                        className="form-label fw-semibold",
                    ),
                    dcc.Slider(
                        id="rf-n-estimators",
                        min=10,
                        max=300,
                        step=10,
                        value=100,
                        marks={10: "10", 100: "100", 200: "200", 300: "300"},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="mb-2",
                    ),
                    html.Label(
                        "profundidade máx.  (0 = ilimitada)",
                        className="form-label fw-semibold mt-2",
                    ),
                    dcc.Slider(
                        id="rf-max-depth",
                        min=0,
                        max=20,
                        step=1,
                        value=0,
                        marks={0: "∞", 5: "5", 10: "10", 20: "20"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                className="mb-3",
                style={"display": "none"},
            ),

            html.Hr(),

            # ── Nome da corrida ───────────────────────────────────────────
            html.Label("Nome da corrida  (opcional)", className="form-label fw-semibold"),
            dbc.Input(
                id="run-name",
                placeholder="ex.: rf-ajustado",
                type="text",
                size="sm",
                className="mb-3",
            ),

            # ── Botão de ação ─────────────────────────────────────────────
            dbc.Button(
                "Executar Experimento",
                id="run-btn",
                color="primary",
                className="w-100",
                n_clicks=0,
            ),

            # ── Mensagem de status ────────────────────────────────────────
            html.Div(id="run-status", className="mt-2"),
        ],
        width=3,
        className="p-3 border-end bg-light",
        style={"minHeight": "calc(100vh - 130px)"},
    )

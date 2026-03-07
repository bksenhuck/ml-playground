"""About page - Technical overview of the ML & LLM architecture."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

def about_layout() -> dbc.Container:
    return dbc.Container(
        [
            # --- SEÇÃO 1: VISÃO GERAL DO PROJETO ---
            dbc.Row(
                dbc.Col([
                    html.Div(
                        [
                            html.H2("Arquitetura do Sistema & Estratégia de IA", className="fw-bold mb-1"),
                            html.Div(style={"width": "60px", "height": "4px", "backgroundColor": "#2980b9", "marginBottom": "20px"}),
                        ],
                        className="mt-4"
                    ),
                    html.P(
                        "O ML Playground é uma plataforma interativa de experimentação para projetos de machine learning, "
                        "contando com um assistente de LLM dedicado para análise técnica.",
                        className="lead text-muted mb-4"
                    ),
                ])
            ),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("O Assistente de IA", className="mb-0 fw-bold"), className="bg-white border-bottom-0 pt-3"),
                        dbc.CardBody([
                            html.Ul([
                                html.Li("Analisa resultados de experimentos em tempo real."),
                                html.Li("Explica métricas complexas (F1-score, ROC AUC, SHAP)."),
                                html.Li("Discute a arquitetura do projeto e fluxos de trabalho."),
                                html.Li("Fornece insights técnicos para demonstrações de portfólio."),
                            ], className="mb-0 list-unstyled lh-lg"),
                        ])
                    ], className="border-0 shadow-sm h-100 rounded-3")
                ], width=12, md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Princípios de Design", className="mb-0 fw-bold"), className="bg-white border-bottom-0 pt-3"),
                        dbc.CardBody([
                            html.Ul([
                                html.Li("Domínio restrito (Apenas Machine Learning / IA)."),
                                html.Li("Respostas concisas, técnicas e diretas."),
                                html.Li("Otimizado para ambientes de baixos recursos."),
                                html.Li("Desenvolvido para cenários de entrevistas e demonstrações."),
                            ], className="mb-0 list-unstyled lh-lg"),
                        ])
                    ], className="border-0 shadow-sm h-100 rounded-3")
                ], width=12, md=6),
            ], className="mb-5 g-4"),

            # --- SEÇÃO 2: ARQUITETURA DO SISTEMA LLM ---
            dbc.Card([
                dbc.CardHeader(html.H4("Pipeline de Processamento LLM", className="fw-bold mb-0 py-2"), className="bg-light border-bottom"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div(
                                    [
                                        html.Div("Pergunta do Usuário", className="p-2 mb-2 rounded bg-secondary text-white text-center small"),
                                        html.Div("↓", className="text-center fw-bold mb-1"),
                                        html.Div("Guardrails de Entrada", className="p-2 mb-2 rounded bg-danger text-white text-center small"),
                                        html.Div("↓", className="text-center fw-bold mb-1"),
                                        html.Div("Construção do Prompt", className="p-2 mb-2 rounded bg-primary text-white text-center small"),
                                        html.Div("↓", className="text-center fw-bold mb-1"),
                                        html.Div("Inferência LLM (Qwen)", className="p-2 mb-2 rounded bg-dark text-white text-center small"),
                                        html.Div("↓", className="text-center fw-bold mb-1"),
                                        html.Div("Insight Final", className="p-2 rounded bg-success text-white text-center small"),
                                    ],
                                    className="d-flex flex-column align-items-stretch"
                                )
                            ], className="px-3 py-2 border rounded-3 bg-white")
                        ], width=12, lg=4, className="d-flex align-items-center mb-4 mb-lg-0"),
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.H6("Guardrails de Entrada", className="fw-bold text-danger"),
                                    html.P("Validação por Regex e Llama Guard 3 1B para bloquear consultas inseguras ou fora de tópico.", className="small"),
                                ], className="mb-3"),
                                html.Div([
                                    html.H6("Construção do Prompt", className="fw-bold text-primary"),
                                    html.P("Injeta metadados técnicos e parâmetros de treinamento em um prompt de sistema especializado.", className="small"),
                                ], className="mb-3"),
                                html.Div([
                                    html.H6("Inferência LLM", className="fw-bold text-dark"),
                                    html.P("Utiliza o Qwen 2.5 (0.5B Instruct) — modelo open-source otimizado para inferência em CPU.", className="small"),
                                ], className="mb-3"),
                                html.Div([
                                    html.H6("Guardrails de Saída", className="fw-bold text-success"),
                                    html.P("Sanitiza as respostas para garantir foco técnico e reduzir alucinações.", className="small"),
                                ]),
                            ])
                        ], width=12, lg=8, className="ps-lg-5")
                    ])
                ], className="p-4")
            ], className="border-0 shadow-sm mb-5 rounded-3"),

            # --- SEÇÃO 3: SEGURANÇA & ROADMAP ---
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Segurança & Proteção", className="mb-0 fw-bold"), className="bg-white border-bottom-0 pt-3"),
                        dbc.CardBody([
                            html.Ul([
                                html.Li([html.Strong("Restrição de Tópico: "), "Software, Ciência de Dados e IA apenas."]),
                                html.Li([html.Strong("Proteção contra Injeção: "), "Detecta e bloqueia tentativas de engenharia de prompt."]),
                                html.Li([html.Strong("Recusa Segura: "), "Fallback: 'Fui projetado para responder sobre projetos de ML.'"]),
                            ], className="mb-0 small lh-lg")
                        ])
                    ], className="border-0 shadow-sm h-100 rounded-3")
                ], width=12, md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Melhorias Futuras", className="mb-0 fw-bold"), className="bg-white border-bottom-0 pt-3"),
                        dbc.CardBody([
                            html.Ul([
                                html.Li([html.Strong("RAG: "), "Recuperação de conhecimento por documentação."]),
                                html.Li([html.Strong("Busca Vetorial: "), "Experimentos históricos via FAISS."]),
                                html.Li([html.Strong("Multi-Modelo: "), "Suporte para Llama 3 local ou APIs."]),
                            ], className="mb-0 small lh-lg")
                        ])
                    ], className="border-0 shadow-sm h-100 rounded-3")
                ], width=12, md=6),
            ], className="mb-5 g-4"),

            html.Hr(className="mt-4"),
            html.Footer(
                "Documentação Técnica — Arquitetura ML Playground v1.1",
                className="text-center text-muted small pb-5"
            )
        ],
        fluid=True,
        className="px-4"
    )



"""About page - Technical overview of the ML & LLM architecture."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def _badge(text: str, color: str) -> html.Span:
    return html.Span(
        text,
        className=f"badge rounded-pill bg-{color} me-1",
        style={"fontSize": "0.72rem"},
    )


def _pipeline_step(
    label: str,
    color: str,
    desc: str,
    detail: str | None = None,
    is_last: bool = False,
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.Span(label, className="fw-semibold small"),
                        className=f"p-2 px-3 rounded-3 bg-{color} text-white d-flex align-items-center",
                        style={"minWidth": "220px"},
                    ),
                    html.Div(
                        [
                            html.Span(desc, className="small text-muted"),
                            html.Span(
                                f" {detail}",
                                className="small text-secondary fst-italic",
                            ) if detail else None,
                        ],
                        className="ms-3 d-flex flex-column justify-content-center",
                    ),
                ],
                className="d-flex align-items-center",
            ),
            html.Div(
                "│",
                className=f"text-{color} fw-bold ms-4 my-0",
                style={"fontSize": "1.2rem", "lineHeight": "1"},
            ) if not is_last else None,
        ],
    )


def about_layout() -> dbc.Container:
    return dbc.Container(
        [
            # ── Header ────────────────────────────────────────────────────────
            dbc.Row(
                dbc.Col([
                    html.Div(
                        [
                            html.H2(
                                "Arquitetura do Sistema & Estratégia de IA",
                                className="fw-bold mb-1",
                            ),
                            html.Div(
                                style={
                                    "width": "60px", "height": "4px",
                                    "backgroundColor": "#2980b9",
                                    "marginBottom": "20px",
                                }
                            ),
                        ],
                        className="mt-4",
                    ),
                    html.P(
                        "O ML Playground é uma plataforma interativa de experimentação "
                        "para projetos de machine learning, com um assistente LLM "
                        "100% local para análise técnica — sem dependências de APIs externas.",
                        className="lead text-muted mb-4",
                    ),
                ])
            ),

            # ── Cards: Assistente + Princípios ────────────────────────────────
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H5("O Assistente de IA", className="mb-0 fw-bold"),
                            className="bg-white border-bottom-0 pt-3",
                        ),
                        dbc.CardBody([
                            html.Ul([
                                html.Li("Analisa resultados de experimentos em tempo real."),
                                html.Li("Explica métricas (F1-score, ROC AUC, SHAP, etc.)."),
                                html.Li("Compara modelos e sugere melhorias de hiperparâmetros."),
                                html.Li("Opera 100% offline — pesos carregados localmente na CPU."),
                            ], className="mb-0 list-unstyled lh-lg"),
                        ]),
                    ], className="border-0 shadow-sm h-100 rounded-3"),
                ], width=12, md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H5("Princípios de Design", className="mb-0 fw-bold"),
                            className="bg-white border-bottom-0 pt-3",
                        ),
                        dbc.CardBody([
                            html.Ul([
                                html.Li("Domínio restrito: somente Machine Learning / IA."),
                                html.Li("Respostas concisas, técnicas e em Português do Brasil."),
                                html.Li("Otimizado para CPU — sem GPU necessária."),
                                html.Li("Projetado para demonstrações e portfólio técnico."),
                            ], className="mb-0 list-unstyled lh-lg"),
                        ]),
                    ], className="border-0 shadow-sm h-100 rounded-3"),
                ], width=12, md=6),
            ], className="mb-5 g-4"),

            # ── Pipeline LLM ──────────────────────────────────────────────────
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col(
                            html.H4("Pipeline de Processamento LLM", className="fw-bold mb-0"),
                            width="auto",
                        ),
                        dbc.Col(
                            [
                                _badge("Open-Source", "success"),
                                _badge("CPU-only", "secondary"),
                                _badge("Offline", "dark"),
                            ],
                            className="d-flex align-items-center",
                        ),
                    ], align="center"),
                    className="bg-light border-bottom py-3",
                ),
                dbc.CardBody([
                    dbc.Row([
                        # ── Diagrama ─────────────────────────────────────────
                        dbc.Col([
                            _pipeline_step(
                                "Pergunta do Usuário", "secondary",
                                "Texto livre em PT-BR",
                            ),
                            _pipeline_step(
                                "Llama Guard 3 1B", "danger",
                                "Moderação neural (opcional)",
                                "— ativo quando carregado",
                            ),
                            _pipeline_step(
                                "Filtro Regex & Tópico", "warning",
                                "Regex + lista de palavras-chave ML",
                            ),
                            _pipeline_step(
                                "Construção do Prompt", "primary",
                                "Injeta contexto dos experimentos",
                            ),
                            _pipeline_step(
                                "Qwen 2.5 0.5B Instruct", "dark",
                                "Inferência local na CPU",
                            ),
                            _pipeline_step(
                                "Insight Final", "success",
                                "Resposta técnica em PT-BR",
                                is_last=True,
                            ),
                        ], width=12, lg=5, className="mb-4 mb-lg-0"),

                        # ── Descrições ───────────────────────────────────────
                        dbc.Col([
                            html.Div([
                                html.H6(
                                    "Guardrails de Entrada (2 camadas)",
                                    className="fw-bold text-danger",
                                ),
                                html.P(
                                    [
                                        html.Strong("Camada 1 — Llama Guard 3 1B: "),
                                        "Modelo de linguagem open-source da Meta especializado em "
                                        "moderação de conteúdo. Classifica a pergunta em categorias "
                                        "de segurança (violência, jailbreak, etc.). Carregado em "
                                        "background; se não disponível, o LLM ainda funciona.",
                                    ],
                                    className="small mb-1",
                                ),
                                html.P(
                                    [
                                        html.Strong("Camada 2 — Regex & Tópico: "),
                                        "Filtro rápido por expressões regulares bloqueia jailbreaks "
                                        "e tentativas de extração de credenciais. Filtro de tópico "
                                        "garante que apenas perguntas sobre ML, métricas e modelos "
                                        "cheguem ao LLM.",
                                    ],
                                    className="small",
                                ),
                            ], className="mb-4 pb-3 border-bottom"),

                            html.Div([
                                html.H6(
                                    "Construção do Prompt",
                                    className="fw-bold text-primary",
                                ),
                                html.P(
                                    "Monta o contexto com os dados dos experimentos selecionados "
                                    "(até 5 runs, colunas relevantes) e injeta o system prompt "
                                    "com instruções de papel, estilo e restrições de resposta.",
                                    className="small",
                                ),
                            ], className="mb-4 pb-3 border-bottom"),

                            html.Div([
                                html.H6(
                                    "Inferência — Qwen 2.5 0.5B Instruct",
                                    className="fw-bold",
                                ),
                                html.P(
                                    [
                                        "Modelo open-source da Alibaba Cloud, otimizado para "
                                        "instrução em múltiplos idiomas. Roda inteiramente na "
                                        "CPU do servidor (sem GPU). Pesos carregados uma vez e "
                                        "mantidos em memória para respostas rápidas. ",
                                        html.Br(),
                                        html.Span(
                                            "Fallback automático para motor de regras local "
                                            "se o modelo não estiver pronto.",
                                            className="fst-italic text-muted",
                                        ),
                                    ],
                                    className="small",
                                ),
                            ], className="mb-4 pb-3 border-bottom"),

                            html.Div([
                                html.H6(
                                    "Fonte da Resposta",
                                    className="fw-bold text-success",
                                ),
                                html.P(
                                    "Cada resposta exibe a fonte: ",
                                    className="small mb-1",
                                ),
                                html.Div([
                                    html.Span(
                                        "LLM",
                                        className="badge bg-success me-2",
                                    ),
                                    html.Span(
                                        "Qwen 2.5 respondeu",
                                        className="small text-muted me-3",
                                    ),
                                    html.Span(
                                        "Local",
                                        className="badge bg-secondary me-2",
                                    ),
                                    html.Span(
                                        "Motor de regras (Qwen carregando ou offline)",
                                        className="small text-muted",
                                    ),
                                ]),
                            ]),
                        ], width=12, lg=7, className="ps-lg-5"),
                    ]),
                ], className="p-4"),
            ], className="border-0 shadow-sm mb-5 rounded-3"),

            # ── Segurança & Roadmap ───────────────────────────────────────────
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H5("Segurança & Proteção", className="mb-0 fw-bold"),
                            className="bg-white border-bottom-0 pt-3",
                        ),
                        dbc.CardBody([
                            html.Ul([
                                html.Li([
                                    html.Strong("Moderação Neural: "),
                                    "Llama Guard 3 1B classifica riscos por categoria.",
                                ]),
                                html.Li([
                                    html.Strong("Restrição de Tópico: "),
                                    "Somente ML, Ciência de Dados e IA.",
                                ]),
                                html.Li([
                                    html.Strong("Anti-Jailbreak: "),
                                    "Regex detecta tentativas de bypass de instruções.",
                                ]),
                                html.Li([
                                    html.Strong("Sem APIs externas: "),
                                    "Nenhum dado sai do servidor.",
                                ]),
                            ], className="mb-0 small lh-lg"),
                        ]),
                    ], className="border-0 shadow-sm h-100 rounded-3"),
                ], width=12, md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H5("Melhorias Futuras", className="mb-0 fw-bold"),
                            className="bg-white border-bottom-0 pt-3",
                        ),
                        dbc.CardBody([
                            html.Ul([
                                html.Li([html.Strong("RAG: "), "Recuperação por documentação e histórico."]),
                                html.Li([html.Strong("Busca Vetorial: "), "Experimentos históricos via FAISS."]),
                                html.Li([html.Strong("Modelo maior: "), "Qwen 2.5 1.5B ou 3B com quantização."]),
                                html.Li([html.Strong("Streaming: "), "Respostas token a token na UI."]),
                            ], className="mb-0 small lh-lg"),
                        ]),
                    ], className="border-0 shadow-sm h-100 rounded-3"),
                ], width=12, md=6),
            ], className="mb-5 g-4"),

            html.Hr(className="mt-4"),
            html.Footer(
                "Documentação Técnica — ML Playground v1.2 · Qwen 2.5 0.5B · Llama Guard 3 1B",
                className="text-center text-muted small pb-5",
            ),
        ],
        fluid=True,
        className="px-4",
    )

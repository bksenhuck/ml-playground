"""ML Playground — Dash application entry point.

Gunicorn target: app.app:server
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so sibling packages import correctly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
import threading

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from app.callbacks import register_callbacks
from app.layout import make_footer, make_navbar
from app.pages.experiments import experiments_layout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,  # override any handlers Flask/Werkzeug may have added first
)
logger = logging.getLogger(__name__)

APP_TITLE = "ML Playground"

_INITIAL_RUNS = [
    {"run_name": "Teste Qwen", "model": "Manual", "metric_accuracy": 0.95}
]


def serve_app() -> dash.Dash:
    """Create, configure and return the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
        title=APP_TITLE,
    )

    app.layout = html.Div(
        [
            html.Div(id="dummy-output", style={"display": "none"}),
            dcc.Store(id="runs-data-store", data=_INITIAL_RUNS),
            dcc.Location(id="url", refresh=False),
            make_navbar(),
            html.Div(experiments_layout(), id="page-content"),
            make_footer(),
        ],
        style={"backgroundColor": "#f0f2f5", "minHeight": "100vh"},
    )

    register_callbacks(app)
    return app


def _preload_llm() -> None:
    """Warm up the Qwen model in the background at startup."""
    try:
        from llm.config import load_config
        from llm.client_qwen import QwenClient
        cfg = load_config()
        if cfg.llm_ready:
            QwenClient.get_or_create(cfg.model_path)
    except Exception as exc:
        logger.warning("[LLM] Background preload failed: %s", exc)


# Module-level initialisation — required for gunicorn (app.app:server)
_dash_app = serve_app()
server = _dash_app.server  # gunicorn target

# Preload Qwen weights in the background so the first question is instant
threading.Thread(target=_preload_llm, daemon=True, name="llm-preload").start()

if __name__ == "__main__":
    _dash_app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False,
    )

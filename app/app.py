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


def serve_app() -> dash.Dash:
    """Create, configure and return the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
        title=APP_TITLE,
    )

    def _layout():
        """Return a fresh layout for every new page load.

        Using a callable ensures each browser session/tab starts with an empty
        runs-data-store, preventing experiment data from leaking across sessions
        or persisting after a server restart.
        """
        return html.Div(
            [
                html.Div(id="dummy-output", style={"display": "none"}),
                dcc.Store(id="runs-data-store", data=[]),
                dcc.Location(id="url", refresh=False),
                make_navbar(),
                html.Div(experiments_layout(), id="page-content"),
                make_footer(),
            ],
            style={"backgroundColor": "#f0f2f5", "minHeight": "100vh"},
        )

    app.layout = _layout

    register_callbacks(app)
    return app


def _preload_llm() -> None:
    """Warm up Qwen AND Llama Guard sequentially in one background thread.

    Loading both here prevents concurrent torch imports from two threads,
    which causes a circular-import crash in torch.utils._pytree.
    """
    try:
        from llm.config import load_config
        from llm.client_qwen import QwenClient
        cfg = load_config()
        if cfg.llm_ready:
            QwenClient.get_or_create(cfg.model_path)
    except Exception as exc:
        logger.warning("[LLM] Qwen preload failed: %s", exc)

    # LlamaGuard is optional: loaded only after Qwen is ready, and only if
    # there is enough free memory.  Since it is not required for the LLM to
    # work (service.py degrades to regex-only guardrails when not ready),
    # we skip preloading it here to avoid OOM-killing the instance.
    # It can be enabled later by uncommenting the block below.
    # try:
    #     from llm.llama_guard import LlamaGuardClient
    #     LlamaGuardClient.get_or_create()
    # except Exception as exc:
    #     logger.warning("[LLM] LlamaGuard preload failed: %s", exc)


# Module-level initialisation — required for gunicorn (app.app:server)
_dash_app = serve_app()
server = _dash_app.server  # gunicorn target

# Pre-import torch BEFORE starting any threads.
# Python's import lock prevents concurrent imports, but only if the first import
# fully completes first.  Importing here ensures torch is cached in sys.modules
# before the background preload thread and any request threads run.
try:
    import torch  # noqa: F401
    logger.info("[startup] torch %s cached in sys.modules", torch.__version__)
except Exception as _torch_exc:
    logger.warning("[startup] torch not available at startup: %s", _torch_exc)

# Preload Qwen + LlamaGuard sequentially in background so first question is fast
threading.Thread(target=_preload_llm, daemon=True, name="llm-preload").start()

if __name__ == "__main__":
    _dash_app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False,
    )

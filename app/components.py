"""Reusable UI component helpers for the Dash app."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

# Style shared by all circular "?" info badges
_BADGE_STYLE = {
    "cursor": "help",
    "color": "#6c757d",
    "fontSize": "0.7rem",
    "border": "1px solid #adb5bd",
    "borderRadius": "50%",
    "width": "15px",
    "height": "15px",
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "marginLeft": "6px",
    "flexShrink": "0",
}


def tooltip_icon(tooltip_id: str, text: str, placement: str = "right") -> list:
    """Return [badge_span, dbc.Tooltip] for a hoverable ❓ info icon."""
    return [
        html.Span("?", id=tooltip_id, style=_BADGE_STYLE),
        dbc.Tooltip(text, target=tooltip_id, placement=placement),
    ]


def section_header(
    title: str,
    tooltip_id: str,
    tooltip_text: str,
    placement: str = "right",
    title_style: dict | None = None,
) -> html.Div:
    """Return a d-flex div with a bold title and a hoverable ❓ badge."""
    style = {"fontWeight": "600", "fontSize": "0.9rem"}
    if title_style:
        style.update(title_style)
    return html.Div(
        [html.Span(title, style=style)]
        + tooltip_icon(tooltip_id, tooltip_text, placement),
        className="d-flex align-items-center",
    )

"""Plot explanation helpers.

Provides:
  PLOT_DESCRIPTIONS  — concise, user-facing description of every chart type.
  create_plot_header — returns a Dash component: title + hoverable ❓ icon.

The ❓ tooltip relies on dbc.Tooltip whose *children* are updated dynamically
by the `update_chart_tooltip` callback in app.py, so the layout only needs a
stable target element (id="chart-info-icon") and an empty Tooltip shell.
create_plot_header() is exported for use in tests or future server-side rendering.
"""
from __future__ import annotations

from dash import html


# ── Descriptions ──────────────────────────────────────────────────────────────

PLOT_DESCRIPTIONS: dict[str, str] = {
    "radar": (
        "Radar chart comparing all metrics across selected runs. "
        "Larger filled area = better overall performance. "
        "Useful for spotting trade-offs, e.g. high precision at the cost of recall."
    ),
    "bar_f1": (
        "F1 score bar chart for selected runs. "
        "F1 is the harmonic mean of precision and recall — "
        "a balanced measure especially useful when classes are imbalanced."
    ),
    "roc": (
        "ROC curve: True Positive Rate vs False Positive Rate as the classification "
        "threshold varies. Higher AUC (area closer to top-left corner) = "
        "better discrimination. Dashed line = random classifier baseline."
    ),
    "pr_curve": (
        "Precision-Recall curve: trade-off between precision and recall as the "
        "threshold varies. Most informative when the positive class is rare "
        "and you care more about false negatives than false positives."
    ),
    "confusion_matrix": (
        "Confusion matrix for the first selected run. "
        "Rows = actual class, columns = predicted class. "
        "Values are normalised by row (row sums = 1). "
        "Diagonal = correct classifications; off-diagonal = errors."
    ),
    "feature_importance": (
        "Feature importance (Random Forest uses mean impurity decrease; "
        "Logistic Regression uses absolute coefficient magnitude). "
        "Shows which features most influence model predictions. Top 15 shown."
    ),
    "calibration": (
        "Calibration curve: how well predicted probabilities match observed "
        "positive rates. A perfectly calibrated model follows the dashed diagonal. "
        "Above diagonal = underconfident (probabilities too low); "
        "below = overconfident."
    ),
    "metric_dist": (
        "Box plot of each metric across all visible runs. "
        "Shows median, IQR, whiskers, and individual points. "
        "Useful for assessing result stability across different hyperparameter configurations."
    ),
    "shap_summary": (
        "SHAP summary: mean absolute SHAP value per feature across test samples. "
        "Higher bar = feature contributes more to predictions on average. "
        "Based on Shapley values — model-agnostic and theoretically sound. "
        "Requires the SHAP library (pip install shap)."
    ),
    "shap_dependence": (
        "SHAP dependence plot: how a single feature's SHAP contribution changes "
        "with its own (transformed) value. "
        "Positive SHAP → pushes prediction toward class 1 (survived). "
        "Colour encodes SHAP magnitude — red = high positive contribution."
    ),
}


# ── Component factory ─────────────────────────────────────────────────────────

def create_plot_header(title: str, plot_key: str) -> html.Div:
    """Return a Dash Div containing a title and a hoverable ❓ abbreviation.

    The abbreviation uses the HTML ``title`` attribute for a built-in browser
    tooltip — no Dash callbacks required, works in every environment.

    Args:
        title:    Display title shown in bold.
        plot_key: Key into PLOT_DESCRIPTIONS for the tooltip text.

    Returns:
        An ``html.Div`` with the title and an ``html.Abbr`` ❓ icon.
    """
    description = PLOT_DESCRIPTIONS.get(plot_key, "")
    return html.Div(
        [
            html.Span(title, style={"fontWeight": "600"}),
            html.Abbr(
                " ❓",
                title=description,
                style={"cursor": "help", "textDecoration": "none", "color": "#6c757d"},
            ),
        ],
        style={"display": "inline-flex", "alignItems": "center"},
    )

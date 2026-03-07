"""Centralized plot style configuration.

Import from here to keep experiment colors consistent across all charts.
Each run/experiment at index i gets PALETTE[i % len(PALETTE)] across all charts.
In "Ambos" (both) mode: test = solid line, train = dashed line (same color).
"""

# Reference / baseline lines (e.g., random diagonal in ROC)
COLOR_REF = "lightgray"

# Per-experiment palette — 8 clearly distinct colors.
# Run i gets PALETTE[i % len(PALETTE)] in every chart type.
PALETTE = [
    "#3498db",  # blue
    "#2ecc71",  # green
    "#e74c3c",  # red
    "#9b59b6",  # purple
    "#f39c12",  # amber
    "#1abc9c",  # teal
    "#e67e22",  # orange
    "#34495e",  # dark slate
]


def hex_rgba(h: str, a: float = 0.2) -> str:
    """Convert a hex colour string to an rgba() CSS string."""
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return f"rgba({r},{g},{b},{a})"

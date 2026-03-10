"""Shared dataset configurations — feature options for UI dropdowns.

Loaded once at import time so every module references the same objects.
"""
from __future__ import annotations

import seaborn as sns

# ── Titanic ───────────────────────────────────────────────────────────────────
_titanic_df = sns.load_dataset("titanic")
_TITANIC_EXCLUDED = {"survived", "alive"}

TITANIC_FEATURES: list[str] = [
    c for c in _titanic_df.columns if c not in _TITANIC_EXCLUDED
]
TITANIC_OPTS: list[dict] = [{"label": f, "value": f} for f in TITANIC_FEATURES]
TITANIC_DEFAULT: list[str] = [
    f for f in ["age", "sex", "pclass", "fare", "embarked"] if f in TITANIC_FEATURES
]
TITANIC_REDUNDANT = {"alive", "class", "who", "adult_male", "embark_town", "alone"}

# ── California Housing ────────────────────────────────────────────────────────
# Feature names are stable across sklearn versions — hardcoded to avoid a
# network fetch at import time (sklearn downloads from figshare on first use).
# The actual dataset is fetched lazily in plots.py and the ML runner.
HOUSING_FEATURES: list[str] = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
HOUSING_OPTS: list[dict] = [{"label": f, "value": f} for f in HOUSING_FEATURES]
HOUSING_DEFAULT: list[str] = HOUSING_FEATURES

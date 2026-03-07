"""Shared dataset configurations — feature options for UI dropdowns.

Loaded once at import time so every module references the same objects.
"""
from __future__ import annotations

import seaborn as sns
from sklearn.datasets import fetch_california_housing

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
_housing_data = fetch_california_housing()
HOUSING_FEATURES: list[str] = list(_housing_data.feature_names)
HOUSING_OPTS: list[dict] = [{"label": f, "value": f} for f in HOUSING_FEATURES]
HOUSING_DEFAULT: list[str] = HOUSING_FEATURES

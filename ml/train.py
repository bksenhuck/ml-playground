"""Dataset loading, preprocessing, and model evaluation utilities."""

import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

TARGET = "survived"

_NUMERIC_COLS = ["pclass", "age", "sibsp", "parch", "fare"]
_CATEGORICAL_COLS = ["sex", "embarked"]

# Stable ordinal encoding for the two categorical columns.
_SEX_MAP = {"female": 0, "male": 1}
_EMBARKED_MAP = {"C": 0, "Q": 1, "S": 2}


def load_titanic() -> pd.DataFrame:
    """Load and preprocess the Titanic dataset from seaborn.

    Preprocessing steps:
    - Keep ``pclass``, ``age``, ``sibsp``, ``parch``, ``fare``,
      ``sex``, ``embarked``, and ``survived``.
    - Fill missing numeric values with the column median.
    - Fill missing categorical values with the column mode.
    - Label-encode ``sex`` (female=0, male=1) and
      ``embarked`` (C=0, Q=1, S=2).
    - Drop any rows that still contain NaN after encoding.

    Returns:
        A clean DataFrame ready for feature selection and training.
        One row per passenger, with ``survived`` as the binary target.
    """
    raw = sns.load_dataset("titanic")
    df = raw[_NUMERIC_COLS + _CATEGORICAL_COLS + [TARGET]].copy()

    for col in _NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())

    for col in _CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["sex"] = df["sex"].map(_SEX_MAP)
    df["embarked"] = df["embarked"].map(_EMBARKED_MAP)

    return df.dropna().reset_index(drop=True)


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return all column names except the target column.

    Args:
        df: Preprocessed Titanic DataFrame produced by :func:`load_titanic`.

    Returns:
        Sorted list of feature column names.
    """
    return [c for c in df.columns if c != TARGET]


def train_and_evaluate(
    pipeline: Pipeline,
    df: pd.DataFrame,
    features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, float], Pipeline, dict]:
    """Fit a pipeline and compute classification metrics on the held-out set.

    Args:
        pipeline: An unfitted sklearn ``Pipeline``.
        df: Preprocessed DataFrame with feature and target columns.
        features: Column names to use as model inputs.
        test_size: Proportion of data reserved for evaluation.
        random_state: Seed for train/test split reproducibility.

    Returns:
        A tuple of:
        - ``metrics``: dict with keys ``accuracy``, ``precision``,
          ``recall``, ``f1``, ``roc_auc`` — all rounded to 4 decimal places.
        - ``pipeline``: The fitted pipeline.
    """
    X, y = df[features], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    metrics: dict[str, float] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    return metrics, pipeline, roc_data

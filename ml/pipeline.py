"""Build sklearn pipelines dynamically based on user selection."""
from __future__ import annotations

from typing import Dict, List

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_pipeline(
    feature_columns: List[str], scaling: str, model_name: str, hyperparams: Dict
) -> Pipeline:
    """Return an sklearn Pipeline configured with preprocessing and estimator.

    feature_columns is for the calling code's bookkeeping; preprocessing operates on whatever
    DataFrame columns are provided at fit time.
    """
    # numeric pipeline: impute then optional scaler
    numeric_transformers = [("imputer", SimpleImputer(strategy="median"))]
    if scaling == "standard":
        numeric_transformers.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_transformers)

    # categorical pipeline: impute then one-hot
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Use selectors so transformers are applied to appropriate dtypes at fit time.
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, make_column_selector(dtype_include=["number"])),
            ("cat", categorical_pipeline, make_column_selector(dtype_include=["object", "category"])),
        ], remainder="drop",
    )

    if model_name == "logreg":
        C = float(hyperparams.get("C", 1.0))
        clf = LogisticRegression(C=C, max_iter=1000)
    else:
        n = int(hyperparams.get("n_estimators", 100))
        d = hyperparams.get("max_depth")
        clf = RandomForestClassifier(n_estimators=n, max_depth=(int(d) if d else None))

    pipeline = Pipeline([("preproc", preprocessor), ("clf", clf)])
    return pipeline


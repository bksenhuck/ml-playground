"""Build sklearn pipelines dynamically based on user selection."""
from __future__ import annotations

from typing import Dict, List

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


def build_pipeline(
    feature_columns: List[str], 
    scaling: str, 
    model_name: str, 
    hyperparams: Dict,
    class_weight: str = "none",
    poly_features: bool = False,
    task: str = "classification"
) -> Pipeline:
    """Return an sklearn Pipeline configured with preprocessing and estimator."""
    # numeric pipeline: impute then optional scaler
    numeric_transformers = [("imputer", SimpleImputer(strategy="median"))]
    if poly_features:
        numeric_transformers.append(("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
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

    if task == "regression":
        if model_name == "logreg": # We'll use Ridge for linear regression baseline
            alpha = float(hyperparams.get("alpha", 1.0))
            clf = Ridge(alpha=alpha)
        elif model_name == "xgb" and _XGB_AVAILABLE:
            n = int(hyperparams.get("n_estimators", 100))
            d = int(hyperparams.get("max_depth", 6))
            lr = float(hyperparams.get("learning_rate", 0.1))
            clf = XGBRegressor(
                n_estimators=n,
                max_depth=d,
                learning_rate=lr,
                verbosity=0,
            )
        else:
            n = int(hyperparams.get("n_estimators", 100))
            d = hyperparams.get("max_depth")
            clf = RandomForestRegressor(n_estimators=n, max_depth=(int(d) if d else None))
    else:
        cw = "balanced" if class_weight == "balanced" else None
        if model_name == "logreg":
            C = float(hyperparams.get("C", 1.0))
            clf = LogisticRegression(C=C, max_iter=1000, class_weight=cw)
        elif model_name == "xgb" and _XGB_AVAILABLE:
            n = int(hyperparams.get("n_estimators", 100))
            d = int(hyperparams.get("max_depth", 6))
            lr = float(hyperparams.get("learning_rate", 0.1))
            clf = XGBClassifier(
                n_estimators=n,
                max_depth=d,
                learning_rate=lr,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            n = int(hyperparams.get("n_estimators", 100))
            d = hyperparams.get("max_depth")
            clf = RandomForestClassifier(n_estimators=n, max_depth=(int(d) if d else None), class_weight=cw)

    pipeline = Pipeline([("preproc", preprocessor), ("clf", clf)])
    return pipeline


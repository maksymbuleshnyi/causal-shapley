from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def train_model(X, y, model_type="mlp", random_state=42):
    """Train a single S-learner on (X) -> y for the given model_type."""
    if model_type == "mlp":
        m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000,
                         random_state=random_state)
    elif model_type == "xgb":
        m = xgb.XGBRegressor(n_estimators=200, max_depth=4,
                             random_state=random_state, verbosity=0)
    elif model_type == "linear":
        m = LinearRegression()
    elif model_type == "poly2":
        m = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                          LinearRegression())
    else:
        raise ValueError(f"Unknown model_type {model_type!r}")
    m.fit(X, y)
    return m


def split_train_eval(X, y, eval_size=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    X_eval = X[:eval_size]
    y_eval = y[:eval_size]
    X_train = X[eval_size:]
    y_train = y[eval_size:]
    return X_train, y_train, X_eval, y_eval

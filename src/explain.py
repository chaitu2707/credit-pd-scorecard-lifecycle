"""
Explainability for the tree models. Scorecard reason codes live in scorecard.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_view(X: pd.DataFrame) -> pd.DataFrame:
    """Categorical columns as their level codes, so SHAP sees the same encoding as the model."""
    Z = X.copy()
    for c in Z.columns:
        if isinstance(Z[c].dtype, pd.CategoricalDtype):
            Z[c] = Z[c].cat.codes.astype(float)
    return Z


def shap_importance(model, X: pd.DataFrame, n_sample: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Mean absolute SHAP value per feature on a sample of the out-of-time population."""
    import shap

    Xs = X.sample(min(n_sample, len(X)), random_state=seed)
    Z = numeric_view(Xs)
    # Pass a plain array: LightGBM rejects a frame whose categorical columns
    # are plain numbers, but accepts the same codes as an array.
    sv = shap.TreeExplainer(model).shap_values(Z.values)
    if isinstance(sv, list):
        sv = sv[1]
    imp = pd.DataFrame({"feature": list(Z.columns), "mean_abs_shap": np.abs(sv).mean(axis=0)})
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

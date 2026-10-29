"""
Performance measures for PD models.

Discrimination: gini, ks_statistic (ranking power).
Calibration:    brier_score, brier_skill_score (probability level).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _as_arrays(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same length")
    return y_true, y_prob


def brier_score(y_true, y_prob) -> float:
    """
    Mean squared difference between predicted PD and the 0/1 outcome.
    Lower is better. Zero means every prediction matched the outcome.
    """
    y_true, y_prob = _as_arrays(y_true, y_prob)
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("predicted probabilities must lie in [0, 1]")
    return float(np.mean((y_prob - y_true) ** 2))


def brier_skill_score(y_true, y_prob) -> float:
    """
    Improvement over always predicting the portfolio's base default rate.
    Above 0 adds value; 0 adds nothing; below 0 is worse than the naive model.
    """
    y_true = np.asarray(y_true, dtype=float)
    base_rate = y_true.mean()
    bs_ref = base_rate * (1 - base_rate)  # Brier score of the naive predictor
    return float(1 - brier_score(y_true, y_prob) / bs_ref)


def gini(y_true, y_score) -> float:
    """Gini = 2 * AUC - 1. The ranking measure risk teams usually quote."""
    return float(2 * roc_auc_score(y_true, y_score) - 1)


def ks_statistic(y_true, y_score) -> float:
    """
    Maximum gap between the cumulative share of defaults and goods captured
    as the cut-off moves down the score. Higher means better separation.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def psi(expected, actual, bins: int = 10) -> float:
    """
    Population Stability Index between two samples of one variable.

    Bin edges come from the expected (development) sample's quantiles, so the
    actual sample is measured on the development bins. Rule of thumb:
    below 0.10 stable, 0.10 to 0.25 moderate shift, above 0.25 major shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        raise ValueError("PSI needs non-empty samples")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf  # catch values outside the development range

    e_pct = np.histogram(expected, bins=edges)[0] / expected.size
    a_pct = np.histogram(actual, bins=edges)[0] / actual.size

    eps = 1e-6  # avoids log(0) for empty bins
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

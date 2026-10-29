"""
WoE scorecard: bins with fixed cut-offs, WoE, logistic regression, points scaling,
PD recovery from points, and reason codes.

Sign convention: WoE = ln(share of goods / share of bads), so a positive WoE means
lower risk. The logistic model predicts default (y = 1), so each coefficient on WoE
should be negative. Points are scaled so that a higher score means lower risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from . import config

MISSING = "missing"
FACTOR = config.PDO / np.log(2)
OFFSET = config.BASE_SCORE - FACTOR * np.log(config.BASE_ODDS)


def _numeric_labels(x: pd.Series, cuts) -> tuple[pd.Series, list[str]]:
    edges = [-np.inf, *sorted(float(c) for c in cuts), np.inf]
    cat = pd.cut(x, bins=edges, include_lowest=True)
    ordered = [str(c) for c in cat.cat.categories]
    labels = cat.astype(str).where(x.notna(), MISSING)
    return labels, ordered


def assign_bins(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    """
    Apply fixed cut-offs. spec[feature] is {"type": "numeric", "cuts": [...]}
    or {"type": "categorical"}. Returns bin labels and the ordered bin list per feature.
    """
    bins, orders = {}, {}
    for f, s in spec.items():
        if s["type"] == "categorical":
            raw = df[f].fillna(MISSING).astype(str)
            if "keep" in s:  # levels kept from training; everything else merges into "other"
                raw = raw.where(raw.isin(s["keep"]) | (raw == MISSING), "other")
            bins[f] = raw
            orders[f] = sorted(bins[f].unique())
        else:
            bins[f], orders[f] = _numeric_labels(df[f], s["cuts"])
    return pd.DataFrame(bins, index=df.index), orders


def woe_maps(train_bins: pd.DataFrame, y, smooth: float = 0.5) -> dict:
    """WoE per bin for each feature, fitted on the training sample only."""
    y = np.asarray(y)
    maps = {}
    for f in train_bins.columns:
        g = pd.DataFrame({"b": train_bins[f].values, "y": y}).groupby("b")["y"].agg(["size", "sum"])
        bads = g["sum"]
        goods = g["size"] - g["sum"]
        dist_good = (goods + smooth) / (goods.sum() + smooth * len(g))
        dist_bad = (bads + smooth) / (bads.sum() + smooth * len(g))
        maps[f] = np.log(dist_good / dist_bad)
    return maps


def woe_matrix(bins: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """Map bins to WoE. Bins unseen in training get 0 (no information)."""
    return pd.DataFrame(
        {f: bins[f].map(maps[f]).fillna(0.0).values for f in maps},
        index=bins.index,
    )


def monotonic_check(train_bins: pd.DataFrame, orders: dict, y, spec: dict) -> pd.DataFrame:
    """
    Checks that the bad rate across ordered numeric bins follows the expected
    direction in config.MONOTONIC_SIGN. Missing bins are excluded from the check.
    """
    y = np.asarray(y)
    rows = []
    for f, sign in config.MONOTONIC_SIGN.items():
        if f not in train_bins.columns or spec[f]["type"] != "numeric":
            continue
        rates = pd.DataFrame({"b": train_bins[f].values, "y": y}).groupby("b")["y"].mean()
        seq = [b for b in orders[f] if b in rates.index]
        r = rates.loc[seq].values
        diffs = np.diff(r) * sign
        rows.append({
            "feature": f,
            "expected": "rises" if sign > 0 else "falls",
            "monotonic": bool(np.all(diffs >= -1e-12)),
            "bad_rates_in_order": " | ".join(f"{x:.4f}" for x in r),
        })
    return pd.DataFrame(rows)


def fit_scorecard(train_bins, y_train, valid_bins, y_valid, c_grid=(0.01, 0.1, 1.0, 10.0)) -> dict:
    """
    Fit logistic regression on WoE. The regularisation strength C is chosen by
    log loss on the validation sample. Out-of-time is never used here.
    """
    maps = woe_maps(train_bins, y_train)
    X_tr = woe_matrix(train_bins, maps)
    X_va = woe_matrix(valid_bins, maps)

    best = None
    for c in c_grid:
        model = LogisticRegression(C=c, max_iter=2000).fit(X_tr, y_train)
        ll = log_loss(y_valid, model.predict_proba(X_va)[:, 1])
        if best is None or ll < best["valid_logloss"]:
            best = {"C": c, "valid_logloss": ll, "model": model}

    m = best["model"]
    return {
        "maps": maps,
        "coef": dict(zip(X_tr.columns, m.coef_[0])),
        "intercept": float(m.intercept_[0]),
        "C": best["C"],
        "valid_logloss": best["valid_logloss"],
    }


def pd_from_model(fit: dict, bins: pd.DataFrame) -> np.ndarray:
    """Predicted PD directly from the logistic model."""
    X = woe_matrix(bins, fit["maps"])
    beta = np.array([fit["coef"][f] for f in X.columns])
    logit = fit["intercept"] + X.values @ beta
    return 1 / (1 + np.exp(-logit))


def points_table(fit: dict) -> tuple[pd.DataFrame, float]:
    """Points per feature and bin, plus the base points."""
    rows = []
    for f, woe in fit["maps"].items():
        beta = fit["coef"][f]
        for b, w in woe.items():
            rows.append((f, b, float(w), float(-FACTOR * beta * w)))
    base = OFFSET - FACTOR * fit["intercept"]
    return pd.DataFrame(rows, columns=["feature", "bin", "woe", "points"]), base


def applicant_points(fit: dict, bins: pd.DataFrame) -> pd.DataFrame:
    """Points per feature for each applicant (rows = applicants, columns = features)."""
    pts, _ = points_table(fit)
    out = {}
    for f in fit["maps"]:
        lookup = pts.loc[pts["feature"] == f].set_index("bin")["points"].to_dict()
        out[f] = bins[f].map(lookup).fillna(0.0).values
    return pd.DataFrame(out, index=bins.index)


def score(fit: dict, bins: pd.DataFrame) -> np.ndarray:
    """Scorecard score: base points plus the sum of feature points."""
    _, base = points_table(fit)
    return base + applicant_points(fit, bins).sum(axis=1).values


def pd_from_score(score_values) -> np.ndarray:
    """Recover PD from a score. Must match pd_from_model if the scaling is right."""
    return 1 / (1 + np.exp((np.asarray(score_values) - OFFSET) / FACTOR))


def reason_codes(fit: dict, bins: pd.DataFrame, top_n: int = 4) -> pd.Series:
    """
    For each applicant, the features where the applicant scores furthest below
    the best bin of that feature. These are the adverse-action reasons.
    """
    pts, _ = points_table(fit)
    best_by_feature = pts.groupby("feature")["points"].max()
    applicant = applicant_points(fit, bins)
    cols = [c for c in applicant.columns if c not in config.REASON_CODE_EXCLUDED]
    shortfall = best_by_feature[cols] - applicant[cols]
    top = shortfall.apply(lambda row: [f for f in row.sort_values(ascending=False).index[:top_n] if row[f] > 1e-9], axis=1)
    return top

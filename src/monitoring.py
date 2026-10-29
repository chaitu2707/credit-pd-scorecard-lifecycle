"""
Monitoring of monthly out-of-time batches scored by the frozen champion.

Traffic lights:
  score PSI against training:     green < 0.10, amber < 0.25, red otherwise
  actual / predicted default rate: green 0.80 to 1.25, amber 0.67 to 1.50 (outside 0.80 to 1.25), red otherwise
  Gini drop vs development:       green < 0.03, amber < 0.05, red otherwise
A batch is red if any measure is red. Two consecutive red batches trigger a retraining review.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .metrics import gini, psi

GREEN, AMBER, RED = "green", "amber", "red"


def psi_light(value: float) -> str:
    if value < config.PSI_GREEN:
        return GREEN
    if value < config.PSI_AMBER:
        return AMBER
    return RED


def calibration_light(actual: float, predicted: float) -> str:
    if not predicted or np.isnan(predicted):
        return RED
    r = actual / predicted
    if 0.80 <= r <= 1.25:
        return GREEN
    if 0.67 <= r < 0.80 or 1.25 < r <= 1.50:
        return AMBER
    return RED


def gini_light(drop: float) -> str:
    if drop < 0.03:
        return GREEN
    if drop < 0.05:
        return AMBER
    return RED


def overall_light(lights) -> str:
    if RED in lights:
        return RED
    if AMBER in lights:
        return AMBER
    return GREEN


def monitor(oot: pd.DataFrame, pd_oot, score_oot, score_train, train_features: pd.DataFrame,
            features: list, dev_gini: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (batch table, feature PSI table). One batch per issue month of the out-of-time sample.
    Evaluable marks batches whose 12-month outcome window closed before the data cut-off.
    """
    oot = oot.reset_index(drop=True)
    frame = pd.DataFrame({
        "month": oot["issue_date"].dt.to_period("M").astype(str).values,
        "y": oot["target"].values.astype(float),
        "p": np.asarray(pd_oot, dtype=float),
        "s": np.asarray(score_oot, dtype=float),
    })
    batches, feat_rows = [], []
    for month, g in frame.groupby("month", sort=True):
        idx = g.index.values
        n = len(g)
        actual = g["y"].mean()
        predicted = g["p"].mean()
        psi_score = psi(score_train, g["s"].values)
        cal = calibration_light(actual, predicted)

        assessed = n >= config.MONITOR_MIN_BATCH and g["y"].nunique() == 2
        if assessed:
            batch_gini = gini(g["y"], g["p"])
            drop = dev_gini - batch_gini
            g_light = gini_light(drop)
        else:
            batch_gini, drop, g_light = np.nan, np.nan, "not assessed"

        lights = [psi_light(psi_score), cal] + ([g_light] if assessed else [])

        feat_psi = {}
        for f in features:
            expected = train_features[f].dropna()
            actual_vals = oot.loc[idx, f].dropna()
            feat_psi[f] = psi(expected, actual_vals) if len(actual_vals) and len(expected) else np.nan
            feat_rows.append({"month": month, "feature": f, "psi": feat_psi[f], "light": psi_light(feat_psi[f]) if not np.isnan(feat_psi[f]) else "not assessed"})
        max_f = max(feat_psi, key=lambda k: -np.inf if np.isnan(feat_psi[k]) else feat_psi[k])

        month_start = pd.Period(month).to_timestamp()
        evaluable = (month_start + pd.DateOffset(months=config.HORIZON_MONTHS)) <= config.DATA_CUTOFF
        batches.append({
            "month": month,
            "loans": n,
            "small_batch": n < config.MONITOR_MIN_BATCH,
            "evaluable": bool(evaluable),
            "actual_bad_rate": actual,
            "predicted_pd": predicted,
            "actual_to_predicted": actual / predicted if predicted else np.nan,
            "calibration_light": cal,
            "psi_score": psi_score,
            "psi_score_light": psi_light(psi_score),
            "batch_gini": batch_gini,
            "gini_drop": drop,
            "gini_light": g_light,
            "max_feature_psi": feat_psi[max_f],
            "max_feature": max_f,
            "overall": overall_light(lights),
        })

    batch_df = pd.DataFrame(batches)
    batch_df["retrain_review"] = (batch_df["overall"].eq(RED) & batch_df["overall"].shift(1).eq(RED))
    return batch_df, pd.DataFrame(feat_rows)

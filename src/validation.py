"""
Validation measures: decile tables, intercept recalibration on validation,
cut-off economics and the grade benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def decile_table(y, p, n: int = 10) -> pd.DataFrame:
    """
    Loans, bad rate and mean PD by risk decile. Decile 10 is the highest predicted PD.
    cum_bad_capture_from_top shows the share of all defaults in the riskiest deciles.
    """
    d = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), n, labels=False) + 1
    g = (d.groupby("decile")
           .agg(loans=("y", "size"), bads=("y", "sum"), bad_rate=("y", "mean"), mean_pd=("p", "mean"))
           .reset_index())
    total = d["y"].sum()
    g = g.sort_values("decile", ascending=False)
    g["cum_bad_capture_from_top"] = g["bads"].cumsum() / total
    return g.sort_values("decile").reset_index(drop=True)


def intercept_shift(p, y, lo: float = -10.0, hi: float = 10.0, iters: int = 200) -> float:
    """
    Shift on the logit scale such that the mean recalibrated PD equals the observed
    bad rate. Fitted on validation only. Bisection, so it always converges.
    """
    z = _logit(p)
    target = float(np.mean(y))
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def apply_shift(p, shift: float) -> np.ndarray:
    return _sigmoid(_logit(p) + shift)


def cutoff_table(p, y, amount, cutoffs, lgds) -> pd.DataFrame:
    """
    Approve loans with PD at or below each cut-off. Reports approval rate, bad rate among
    approved loans, the share of defaults declined, and expected loss rate for each LGD.
    Expected loss = PD x LGD x loan amount, over approved exposure.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    amount = np.asarray(amount, dtype=float)
    total_bads = y.sum()
    rows = []
    for c in cutoffs:
        ap = p <= c
        n_ap = int(ap.sum())
        exposure = amount[ap].sum()
        row = {
            "max_pd": c,
            "approved_loans": n_ap,
            "approval_rate": ap.mean(),
            "bad_rate_approved": y[ap].mean() if n_ap else np.nan,
            "defaults_declined_share": y[~ap].sum() / total_bads if total_bads else np.nan,
        }
        for lgd in lgds:
            key = f"expected_loss_rate_lgd_{int(round(lgd * 100))}"
            row[key] = (p[ap] * lgd * amount[ap]).sum() / exposure if exposure else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def grade_benchmark_pd(train_grade, train_y, grade) -> np.ndarray:
    """Lending Club grade as a stand-alone PD: training bad rate per grade, applied as is."""
    rates = pd.Series(np.asarray(train_y, dtype=float)).groupby(
        np.asarray(train_grade).astype(str)).mean()
    overall = float(np.mean(train_y))
    return pd.Series(np.asarray(grade).astype(str)).map(rates).fillna(overall).values

"""
Synthetic loans with the same columns as the real working file. Used by the smoke
test and to check the pipeline before it runs on the real data. Not for results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2012-01-01")
    span_days = (pd.Timestamp("2017-09-30") - start).days
    issue = start + pd.to_timedelta(rng.integers(0, span_days, n), unit="D")

    fico_low = np.clip(rng.normal(690, 30, n), 640, 800)
    annual_inc = np.exp(rng.normal(11, 0.5, n))
    dti = np.clip(rng.gamma(3, 5, n), 0, 45)
    loan_amnt = rng.uniform(1000, 35000, n)
    term_60 = rng.random(n) < 0.25
    grade_idx = np.clip(np.round(rng.normal(2.3, 1.3, n)), 0, 6).astype(int)
    grade = np.array(list("ABCDEFG"))[grade_idx]
    inq = rng.poisson(1.0, n).astype(float)
    mort = rng.poisson(1.0, n).astype(float)
    mort[rng.random(n) < 0.02] = np.nan
    chm = np.clip(rng.normal(190, 70, n), 24, 480)
    installment = loan_amnt * np.where(term_60, 0.022, 0.035)

    drift = np.where(issue >= pd.Timestamp("2016-01-01"), 0.4, 0.0)
    logit = (-3.6 + 0.04 * (dti - 15) - 0.02 * (fico_low - 690) + 0.25 * inq
             + 0.25 * grade_idx - 0.004 * (chm - 190) + drift)
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    purposes = ["debt_consolidation", "credit_card", "home_improvement", "other", "car",
                "small_business", "major_purchase", "medical", "house", "moving"]
    emp = rng.choice(["10+ years", "< 1 year", "3 years", "5 years", None], n)
    df = pd.DataFrame({
        "issue_date": issue,
        "issue_d": pd.Series(issue).dt.strftime("%b-%Y").values,
        "target": target,
        "grade": grade,
        "loan_amnt": loan_amnt,
        "term": np.where(term_60, " 60 months", " 36 months"),
        "installment": installment,
        "purpose": rng.choice(purposes, n, p=[0.45, 0.12, 0.05, 0.1, 0.05, 0.03, 0.07, 0.04, 0.05, 0.04]),
        "emp_length": emp,
        "home_ownership": rng.choice(["MORTGAGE", "RENT", "OWN", "OTHER", "NONE"], n, p=[0.5, 0.4, 0.09, 0.005, 0.005]),
        "annual_inc": annual_inc,
        "verification_status": rng.choice(["Verified", "Not Verified", "Source Verified"], n),
        "dti": dti,
        "fico_range_low": fico_low,
        "fico_range_high": fico_low + 4,
        "earliest_cr_line": [f"{m}-{y}" for m, y in zip(rng.choice(["Jan", "Mar", "Jun", "Sep", "Dec"], n),
                                                         rng.integers(1985, 2011, n))],
        "revol_bal": rng.gamma(2, 6000, n),
        "revol_util": np.where(rng.random(n) < 0.005, np.nan, np.clip(rng.normal(55, 25, n), 0, 150)),
        "open_acc": rng.poisson(11, n).astype(float),
        "total_acc": rng.poisson(24, n).astype(float),
        "pub_rec": rng.poisson(0.15, n).astype(float),
        "delinq_2yrs": rng.poisson(0.2, n).astype(float),
        "collections_12_mths_ex_med": rng.poisson(0.01, n).astype(float),
        "mths_since_last_delinq": np.where(rng.random(n) < 0.55, np.nan, rng.integers(1, 120, n).astype(float)),
        "inq_last_6mths": inq,
        "mort_acc": mort,
    })
    return df

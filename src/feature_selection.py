"""
Feature engineering, Information Value screening, correlation and VIF checks,
and the leakage assertion for Stage 3.

Everything here is fitted or computed on the training sample only, except
where a function explicitly takes a sample to apply to.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

CATEGORICAL = ["purpose", "home_ownership", "verification_status"]

CANDIDATE_FEATURES = [
    # capacity
    "annual_inc", "dti", "installment_to_income", "loan_to_income", "loan_amnt",
    "emp_years",
    # credit history and behaviour
    "fico", "credit_history_months", "revol_util", "revol_bal", "open_acc",
    "total_acc", "inq_last_6mths", "mort_acc",
    # derogatory
    "pub_rec", "delinq_2yrs", "has_delinquency", "mths_since_last_delinq",
    "collections_12_mths_ex_med",
    # loan structure
    "term_months", "purpose", "home_ownership", "verification_status",
]

IV_BANDS = [
    (0.02, "weak, drop"),
    (0.10, "weak"),
    (0.30, "medium"),
    (0.50, "strong"),
    (np.inf, "very strong, leakage review"),
]


def parse_emp_length(value) -> float:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, missing -> NaN."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.startswith("<"):
        return 0.0
    match = re.match(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the derived features. Uses only information known at application."""
    out = df.copy()
    out["emp_years"] = out["emp_length"].map(parse_emp_length)
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    out["term_months"] = out["term"].str.extract(r"(\d+)", expand=False).astype(float)

    earliest = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y", errors="coerce")
    out["credit_history_months"] = (
        (out["issue_date"].dt.year - earliest.dt.year) * 12
        + (out["issue_date"].dt.month - earliest.dt.month)
    )

    income = out["annual_inc"].where(out["annual_inc"] > 0)
    out["installment_to_income"] = out["installment"] * 12 / income
    out["loan_to_income"] = out["loan_amnt"] / income
    out["has_delinquency"] = ((out["delinq_2yrs"] > 0) | (out["pub_rec"] > 0)).astype(int)
    return out


def assert_no_leakage(features: list[str]) -> None:
    """Raise if any feature is a leakage, label-only or benchmark column."""
    forbidden = set(config.LEAKAGE_COLUMNS) | set(config.LABEL_ONLY_COLUMNS) | set(config.BENCHMARK_COLUMNS)
    bad = sorted(set(features) & forbidden)
    if bad:
        raise ValueError(f"Forbidden columns in feature set: {bad}")


def _bin_numeric(x: pd.Series, bins: int) -> pd.Series:
    """
    Bin one numeric feature using training values only. Missing gets its own label.

    - Few distinct values (e.g. term 36/60, or a 0/1 flag): one bin per value.
    - Heavy zero mass (e.g. count of delinquencies, mostly 0): a separate bin
      for 0, with quantile bins on the positive values. Plain quantiles would
      collapse to one bin here and hide the signal.
    - Otherwise: quantile bins.
    """
    valid = x.dropna()
    if valid.nunique() <= bins:
        return x.where(x.notna(), "missing").astype(str)

    if (valid == 0).mean() >= 1 / bins:
        positive = valid[valid > 0]
        if positive.empty:
            return x.where(x.notna(), "missing").astype(str)
        q = np.unique(np.quantile(positive, np.linspace(0, 1, bins)))
        interior = q[1:-1]
        edges = np.unique(np.concatenate([[-np.inf, 0.0], interior, [np.inf]]))
    else:
        q = np.unique(np.quantile(valid, np.linspace(0, 1, bins + 1)))
        edges = np.concatenate([[-np.inf], q[1:-1], [np.inf]])

    binned = pd.cut(x, bins=edges, include_lowest=True).astype(object)
    return binned.where(x.notna(), "missing")


def woe_table(train: pd.DataFrame, feature: str, bins: int = 5, target: str = "target") -> pd.DataFrame:
    """
    WoE and IV for one feature on the training sample.
    Counts get +0.5 smoothing so empty bins do not produce infinite WoE.
    """
    if feature in CATEGORICAL:
        groups = train[feature].fillna("missing").astype(str)
    else:
        groups = _bin_numeric(train[feature], bins).astype(str)

    frame = pd.DataFrame({"bin": groups, "y": train[target].values})
    agg = frame.groupby("bin")["y"].agg(n="size", bads="sum").reset_index()
    agg["goods"] = agg["n"] - agg["bads"]

    dist_good = (agg["goods"] + 0.5) / (agg["goods"].sum() + 0.5 * len(agg))
    dist_bad = (agg["bads"] + 0.5) / (agg["bads"].sum() + 0.5 * len(agg))
    agg["bad_rate"] = agg["bads"] / agg["n"]
    agg["woe"] = np.log(dist_good / dist_bad)
    agg["iv_contribution"] = (dist_good - dist_bad) * agg["woe"]
    return agg


def information_value(train: pd.DataFrame, feature: str, bins: int = 5, target: str = "target") -> float:
    return float(woe_table(train, feature, bins, target)["iv_contribution"].sum())


def iv_band(iv: float) -> str:
    for upper, label in IV_BANDS:
        if iv < upper:
            return label
    return IV_BANDS[-1][1]


def iv_screen(df: pd.DataFrame, features: list[str], bins: int = 5) -> pd.DataFrame:
    """IV for each feature, computed on the training sample only."""
    train = df[df["sample"] == "train"]
    rows = []
    for f in features:
        iv = information_value(train, f, bins)
        rows.append((f, iv, iv_band(iv)))
    return pd.DataFrame(rows, columns=["feature", "iv", "band"]).sort_values("iv", ascending=False)


def correlated_pairs(train: pd.DataFrame, features: list[str], threshold: float = 0.7) -> pd.DataFrame:
    """Pairs of numeric features with absolute correlation above the threshold."""
    numeric = [f for f in features if f not in CATEGORICAL]
    corr = train[numeric].corr(method="pearson")
    pairs = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append((a, b, round(float(r), 3)))
    return pd.DataFrame(pairs, columns=["feature_a", "feature_b", "correlation"])


def vif(train: pd.DataFrame, features: list[str]) -> pd.Series:
    """
    Variance inflation factor per feature: 1 / (1 - R^2) from regressing that
    feature on the others. Above about 5 to 10 signals collinearity.
    """
    X = train[features].dropna()
    X = (X - X.mean()) / X.std(ddof=0)
    values = {}
    for col in features:
        y = X[col].values
        others = X.drop(columns=col).values
        design = np.column_stack([np.ones(len(others)), others])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0.0
        values[col] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(values).sort_values(ascending=False)

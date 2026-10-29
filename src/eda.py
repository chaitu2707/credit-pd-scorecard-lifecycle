"""
Helper functions for the Stage 2 EDA notebooks. Kept here so the notebooks
show results, not long code blocks, and so the checks can be tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import psi


def missingness_by_year(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Share of missing values per feature, by issue year (rows = features)."""
    year = df["issue_date"].dt.year
    return df.groupby(year)[cols].apply(lambda g: g.isna().mean()).T


def term_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Share of 36- and 60-month loans by issue year."""
    return pd.crosstab(df["issue_date"].dt.year, df["term"], normalize="index")


def bad_rate_by_term_year(df: pd.DataFrame) -> pd.DataFrame:
    """12-month bad rate and loan count by issue year and term."""
    year = df["issue_date"].dt.year.rename("year")
    return df.groupby([year, "term"])["target"].agg(loans="size", bad_rate="mean").reset_index()


def feature_psi(train: pd.DataFrame, other: pd.DataFrame, cols: list[str], bins: int = 10) -> pd.DataFrame:
    """PSI of each numeric feature, development sample against a comparison sample."""
    rows = []
    for c in cols:
        a = train[c].dropna()
        b = other[c].dropna()
        if len(a) == 0 or len(b) == 0:
            rows.append((c, np.nan, len(a), len(b)))
            continue
        rows.append((c, psi(a, b, bins=bins), len(a), len(b)))
    return pd.DataFrame(rows, columns=["feature", "psi", "n_train", "n_compare"]).sort_values("psi", ascending=False)


def numeric_profile(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Median and mean of numeric features, for a quick side-by-side read."""
    return df[cols].agg(["median", "mean"]).T

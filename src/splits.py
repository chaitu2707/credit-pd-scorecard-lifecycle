"""
Vintage-based samples. Splits are by issue date, never random, so the
out-of-time sample reflects loans the model has not seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def assign_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Label each loan as train, validation, oot, or excluded by issue date."""
    out = df.copy()
    names = list(config.SAMPLE_WINDOWS)
    conditions = [out["issue_date"].between(*config.SAMPLE_WINDOWS[n]) for n in names]
    out["sample"] = np.select(conditions, names, default="excluded")
    return out


def sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Loans and bad rate per sample. Reported in the validation report."""
    return (
        df.groupby("sample")
        .agg(loans=("target", "size"), bad_rate=("target", "mean"))
        .reset_index()
    )

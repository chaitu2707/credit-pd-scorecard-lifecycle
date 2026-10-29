"""
Build a slim working file for the modelling notebooks.

Keeps only the columns the scorecard and feature selection use, and only the
train, validation and out-of-time samples. Numbers are stored as float32.
No values are changed, so results are identical to the full file.

Run from the repo root:
    python -m src.make_slim
"""
from __future__ import annotations

from . import config

# Raw inputs needed to build the derived features and the candidate set.
RAW_INPUTS = [
    "emp_length", "fico_range_low", "fico_range_high", "term", "earliest_cr_line",
    "installment", "annual_inc", "loan_amnt", "delinq_2yrs", "pub_rec",
    "collections_12_mths_ex_med", "dti", "revol_util", "revol_bal", "open_acc",
    "total_acc", "inq_last_6mths", "mths_since_last_delinq", "mort_acc",
    "purpose", "home_ownership", "verification_status",
]
KEEP = ["issue_date", "target", "sample", "grade"] + RAW_INPUTS
MODELLING_SAMPLES = ["train", "validation", "oot"]


def build(src=config.PROCESSED_PATH, dst=config.SLIM_PATH) -> None:
    import pandas as pd

    df = pd.read_parquet(src)
    before_rows, before_cols = len(df), df.shape[1]

    from .splits import assign_sample
    df = assign_sample(df)
    df = df[df["sample"].isin(MODELLING_SAMPLES)][KEEP].copy()

    for c in df.select_dtypes(include="float64").columns:
        df[c] = df[c].astype("float32")
    for c in ["purpose", "home_ownership", "verification_status", "term", "grade", "sample", "addr_state"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)

    print(f"full file:  {before_rows:,} rows x {before_cols} columns  ({src.stat().st_size / 1e6:,.0f} MB)")
    print(f"slim file:  {len(df):,} rows x {df.shape[1]} columns  ({dst.stat().st_size / 1e6:,.0f} MB)")


if __name__ == "__main__":
    build()

"""
Target construction, maturity filtering and the data quality funnel.

12-month PD label:
    1 (bad)  = charged off or defaulted, with the default estimated to fall
               within 12 months of issue
    0 (good) = fully paid, or charged off only after the 12-month window

Default date is estimated as last payment date + CHARGE_OFF_LAG_DAYS.
This is an approximation and is recorded in docs/assumptions.md.

Run from the repo root:
    python -m src.data_prep
"""
from __future__ import annotations

import pandas as pd

from . import config

# Base outcome over the loan's life. 1 = charged off or defaulted, 0 = fully paid.
# Open or ambiguous statuses map to NaN and are dropped.
STATUS_TO_BASE_OUTCOME = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}


def load_raw(path=config.RAW_PATH, columns=config.LOAD_COLUMNS) -> pd.DataFrame:
    """Read only the columns we need from the raw Lending Club file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found at {path}. See data/raw/README.md."
        )
    return pd.read_csv(path, usecols=columns, low_memory=False)


def prepare(
    raw: pd.DataFrame, lag_days: int = config.CHARGE_OFF_LAG_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the labelling and maturity rules in a fixed order, record how many
    rows each rule removes, and build the 12-month target.

    Returns the labelled frame and the data quality funnel.
    """
    steps = []
    df = raw.copy()
    steps.append(("raw file rows", len(df)))

    # issue_d is text such as "Dec-2015".
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

    # Step 1: keep only loans with a final outcome.
    df["charged_off"] = df["loan_status"].map(STATUS_TO_BASE_OUTCOME)
    df = df[df["charged_off"].notna()]
    steps.append(("final outcome only (drop Current, grace, late)", len(df)))

    # Step 2: maturity. The loan needs a full 12-month window before the cut-off.
    df = df[df["issue_date"] <= config.MAX_ISSUE_DATE]
    steps.append((f"issued on or before {config.MAX_ISSUE_DATE.date()}", len(df)))

    # Step 3: charged-off loans need a last payment date to estimate default timing.
    df["last_pymnt_date"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y")
    no_date = (df["charged_off"] == 1) & df["last_pymnt_date"].isna()
    df = df[~no_date]
    steps.append(("charged-off loans with no last payment date", len(df)))

    # Step 4: 12-month label. Bad only if the estimated default falls inside the window.
    df["est_default_date"] = df["last_pymnt_date"] + pd.Timedelta(days=lag_days)
    df["months_to_default"] = (df["est_default_date"] - df["issue_date"]).dt.days / 30.44
    df["target"] = (
        (df["charged_off"] == 1) & (df["months_to_default"] <= config.HORIZON_MONTHS)
    ).astype(int)

    df = df.reset_index(drop=True)

    funnel = pd.DataFrame(steps, columns=["step", "rows_remaining"])
    funnel["rows_removed"] = -funnel["rows_remaining"].diff().fillna(0).astype(int)
    return df, funnel


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    """How the 12-month rule changes the labels versus 'ever charged off'."""
    ever_bad = int((df["charged_off"] == 1).sum())
    bad_12m = int(df["target"].sum())
    return pd.DataFrame(
        {
            "measure": [
                "charged off over life",
                "bad within 12 months (target = 1)",
                "charged off after 12 months (relabelled good)",
            ],
            "loans": [ever_bad, bad_12m, ever_bad - bad_12m],
        }
    )


def bad_rate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """12-month bad rate by issue year."""
    return (
        df.assign(year=df["issue_date"].dt.year)
        .groupby("year")
        .agg(loans=("target", "size"), bad_rate=("target", "mean"))
        .reset_index()
    )


def main() -> None:
    raw = load_raw()
    labelled, funnel = prepare(raw)

    config.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(config.PROCESSED_PATH, index=False)

    config.FUNNEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    funnel.to_csv(config.FUNNEL_PATH, index=False)

    print(funnel.to_string(index=False))
    print()
    print(label_summary(labelled).to_string(index=False))
    print(f"\nOverall 12-month bad rate: {labelled['target'].mean():.2%}")
    print(bad_rate_summary(labelled).to_string(index=False))


if __name__ == "__main__":
    main()

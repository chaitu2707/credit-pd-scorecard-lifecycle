"""
Label sensitivity: how much the 12-month target depends on the charge-off lag.

Reruns the label with several lags (days from last payment to default) and
compares bad rates by sample, plus the number of loans whose label flips
versus the base case.

Run from the repo root:
    python -m src.label_sensitivity
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .data_prep import load_raw, prepare
from .splits import assign_sample

LAGS_DAYS = [90, 120, 150]
BASE_LAG = config.CHARGE_OFF_LAG_DAYS
TABLE_PATH = config.ROOT / "reports" / "tables" / "label_sensitivity.csv"
FIGURE_PATH = config.ROOT / "reports" / "figures" / "label_sensitivity.png"


def run() -> pd.DataFrame:
    raw = load_raw()

    # Labelled frames per lag. The row set does not depend on the lag, so
    # rows align by position across lags.
    frames = {lag: prepare(raw, lag_days=lag)[0] for lag in LAGS_DAYS}
    base = frames[BASE_LAG]

    rows = []
    for lag, df in frames.items():
        assert len(df) == len(base), "row sets must match across lags"
        sampled = assign_sample(df)
        flipped = int((df["target"].values != base["target"].values).sum())

        row = {
            "lag_days": lag,
            "is_base": lag == BASE_LAG,
            "bad_loans": int(df["target"].sum()),
            "bad_rate_all": df["target"].mean(),
        }
        for name in config.SAMPLE_WINDOWS:
            subset = sampled.loc[sampled["sample"] == name, "target"]
            row[f"bad_rate_{name}"] = subset.mean()
        row["loans_changed_vs_base"] = flipped
        rows.append(row)

    return pd.DataFrame(rows)


def plot(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(table))
    width = 0.2
    series = ["bad_rate_all"] + [f"bad_rate_{n}" for n in config.SAMPLE_WINDOWS]
    labels = ["All loans"] + list(config.SAMPLE_WINDOWS)
    for i, (col, lab) in enumerate(zip(series, labels)):
        ax.bar([p + (i - 1.5) * width for p in x], table[col] * 100, width, label=lab)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{d} days" + (" (base)" if b else "") for d, b in zip(table["lag_days"], table["is_base"])])
    ax.set_ylabel("12-month bad rate (%)")
    ax.set_title("12-month bad rate by charge-off lag and sample")
    ax.legend()
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    table = run()
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)
    plot(table)

    show = table.copy()
    for col in [c for c in show.columns if c.startswith("bad_rate")]:
        show[col] = (show[col] * 100).round(2).astype(str) + "%"
    print(show.to_string(index=False))
    print(f"\nSaved: {TABLE_PATH}")
    print(f"Saved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()

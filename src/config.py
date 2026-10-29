"""
Single source of truth for paths, dates, column lists and confirmed settings.

Every assumption that affects the data or the models lives here and is described in
docs/assumptions.md. Change a value here, not in the modules.
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "loans_labelled.parquet"
SLIM_PATH = ROOT / "data" / "processed" / "loans_slim.parquet"
FUNNEL_PATH = ROOT / "reports" / "tables" / "data_quality_funnel.csv"
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42

# Maturity rule: a 12-month PD needs issue_date + 12 months <= data cut-off.
HORIZON_MONTHS = 12
DATA_CUTOFF = pd.Timestamp("2018-12-31")
MAX_ISSUE_DATE = pd.Timestamp("2017-09-30")

# Vintage-based samples (issue date windows, inclusive).
SAMPLE_WINDOWS = {
    "train":      (pd.Timestamp("2012-01-01"), pd.Timestamp("2014-06-30")),
    "validation": (pd.Timestamp("2014-07-01"), pd.Timestamp("2015-12-31")),
    "oot":        (pd.Timestamp("2016-01-01"), pd.Timestamp("2017-09-30")),
}

# Columns loaded from the raw file. grade and int_rate are loaded for the benchmark only.
LOAD_COLUMNS = [
    "id", "loan_status", "issue_d", "loan_amnt", "term", "installment",
    "purpose", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "dti", "fico_range_low", "fico_range_high",
    "earliest_cr_line", "revol_bal", "revol_util", "open_acc", "total_acc",
    "pub_rec", "delinq_2yrs", "mths_since_last_delinq", "inq_last_6mths",
    "mort_acc", "collections_12_mths_ex_med", "addr_state",
    "grade", "int_rate",
    "last_pymnt_d",  # label construction only, never a feature
]

# Fields only known after origination. They must never be model features.
LEAKAGE_COLUMNS = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "out_prncp",
    "out_prncp_inv", "debt_settlement_flag", "settlement_status",
]

# Lending Club's own pricing view. Benchmark only, except grade, which is a model input by decision.
BENCHMARK_COLUMNS = ["grade", "int_rate"]

# Label construction. Default date is estimated as last payment + 120 days.
CHARGE_OFF_LAG_DAYS = 120
LABEL_ONLY_COLUMNS = ["last_pymnt_d", "last_pymnt_date", "est_default_date", "months_to_default"]

# Stage 3 confirmed shortlist (installment_to_income kept, loan_to_income dropped).
SCORECARD_FEATURES = [
    "fico", "inq_last_6mths", "purpose", "annual_inc", "installment_to_income",
    "credit_history_months", "term_months", "dti", "home_ownership", "mort_acc",
    "grade",
]
CATEGORICAL_SCORECARD = ["purpose", "home_ownership", "term_months", "grade"]

# Stage 4 confirmed cut-offs (from the scorecard notebook, section 1).
SCORECARD_CUTS = {
    "fico": [672.0, 682.0, 697.0, 717.0],
    "inq_last_6mths": [0.0, 1.0, 2.0],
    "annual_inc": [42000.0, 55000.0, 71607.0, 95370.0],
    "installment_to_income": [0.0, 0.1],
    "credit_history_months": [123.0, 158.0, 195.0, 253.0],
    "dti": [10.2, 14.8, 19.1, 24.1],
    "mort_acc": [0.0, 1.0, 2.0, 4.0],
}

# Expected direction of bad rate as the variable rises (+1 rises, -1 falls).
MONOTONIC_SIGN = {
    "fico": -1,
    "dti": +1,
    "inq_last_6mths": +1,
    "installment_to_income": +1,
    "annual_inc": -1,
    "credit_history_months": -1,
    "mort_acc": -1,
}
BASE_SCORE = 600
BASE_ODDS = 50      # good : bad odds at the base score
PDO = 20            # points to double the odds

# Categories below this share of training loans merge into "other".
RARE_CATEGORY_SHARE = 0.01
# Features kept in the score but excluded from reason codes given to applicants.
REASON_CODE_EXCLUDED = ["purpose", "grade"]
# Categories in these features are never merged into "other".
RARE_MERGE_EXEMPT = ["grade", "term_months"]

# Stage 5 to 7 settings.
TUNE_TRIALS = int(os.environ.get("PIPELINE_TRIALS", 50))
LGD_GRID = [0.50, 0.60, 0.70]
CUTOFF_GRID = [0.03, 0.04, 0.05, 0.06, 0.08]
CHAMPION_GAIN_MIN = 0.02          # challenger must add this much out-of-time Gini to replace champion
PSI_GREEN, PSI_AMBER = 0.10, 0.25
MONITOR_MIN_BATCH = 500           # batches smaller than this are not assessed on Gini

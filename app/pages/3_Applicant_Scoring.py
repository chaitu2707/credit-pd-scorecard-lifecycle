"""Score one applicant with the saved scorecard. Reason codes exclude purpose and grade."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_json  # noqa: E402
from src import scorecard as sc  # noqa: E402
from src.validation import apply_shift  # noqa: E402

st.set_page_config(page_title="Applicant scoring", layout="wide")
st.title("Applicant scoring")

card = load_json("scorecard.json")
if card is None:
    st.error("Run `make train` first.")
    st.stop()

spec = card["spec"]
levels = {f: spec[f]["keep"] for f in spec if spec[f]["type"] == "categorical"}

left, right = st.columns(2)
with left:
    fico = st.number_input("FICO score (midpoint of range)", 600, 850, 690)
    inq = st.number_input("Credit enquiries, last 6 months", 0, 20, 0)
    annual_inc = st.number_input("Annual income ($)", 1000, 2_000_000, 65000, step=1000)
    installment = st.number_input("Monthly instalment ($)", 0.0, 5000.0, 350.0, step=10.0)
    chm = st.number_input("Credit history (months)", 0, 600, 180)
with right:
    term = st.selectbox("Term (months)", [36, 60])
    dti = st.number_input("Debt-to-income ratio (%)", 0.0, 60.0, 18.0, step=0.5)
    mort = st.number_input("Mortgage accounts", 0, 30, 1)
    purpose = st.selectbox("Loan purpose", levels["purpose"], index=0)
    home = st.selectbox("Home ownership", levels["home_ownership"], index=0)
    grade = st.selectbox("Lending Club grade", levels["grade"], index=2)

row = pd.DataFrame([{
    "fico": float(fico),
    "inq_last_6mths": float(inq),
    "purpose": purpose,
    "annual_inc": float(annual_inc),
    "installment_to_income": installment * 12 / annual_inc if annual_inc > 0 else np.nan,
    "credit_history_months": float(chm),
    "term_months": float(term),
    "dti": float(dti),
    "home_ownership": home,
    "mort_acc": float(mort),
    "grade": grade,
}])

fit = {
    "maps": {f: pd.Series(card["maps"][f], dtype=float) for f in card["maps"]},
    "coef": card["coef"],
    "intercept": card["intercept"],
}
bins, _ = sc.assign_bins(row, spec)
pd_raw = float(sc.pd_from_model(fit, bins)[0])
pd_cal = float(apply_shift(np.array([pd_raw]), card["recalibration_shift"])[0])
score = float(sc.score(fit, bins)[0])
reasons = sc.reason_codes(fit, bins, top_n=4).iloc[0]

st.subheader("Result")
c1, c2 = st.columns(2)
c1.metric("Score (points)", f"{score:.0f}")
c2.metric("Probability of default, 12 months", f"{pd_cal:.2%}")
st.write("**Main factors lowering the score:** " + (", ".join(reasons) if reasons else "none"))
st.caption("The probability is recalibrated on validation data, so it is the level to use. "
           "The score is unchanged by recalibration. Purpose and grade are not shown as reasons.")

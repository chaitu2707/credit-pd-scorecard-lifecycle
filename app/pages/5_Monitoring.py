"""Monthly out-of-time batches under the frozen champion, with traffic lights."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import table  # noqa: E402

st.set_page_config(page_title="Monitoring", layout="wide")
st.title("Monitoring (monthly out-of-time batches)")

batches = table("monitoring_batches.csv")
feats = table("monitoring_features.csv")
if batches is None:
    st.error("Run `make train` first.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Batches", int(len(batches)))
c2.metric("Red batches", int((batches["overall"] == "red").sum()))
c3.metric("Retraining reviews triggered", int(batches["retrain_review"].sum()))

st.subheader("Actual versus predicted default rate by month")
chart = batches.set_index("month")[["predicted_pd", "actual_bad_rate"]] * 1.0
st.line_chart(chart)

st.subheader("Batch results")
st.dataframe(batches.round(4), use_container_width=True, hide_index=True)

st.subheader("Feature PSI by month (against training)")
if feats is not None:
    pivot = feats.pivot(index="feature", columns="month", values="psi").round(3)
    st.dataframe(pivot, use_container_width=True)
st.caption("Light rules: PSI green below 0.10, amber below 0.25, red above. "
           "Calibration green between 0.80 and 1.25 of predicted. "
           "Two consecutive red batches trigger a retraining review.")

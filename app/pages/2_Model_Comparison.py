"""Compare the scorecard with the challenger models on each sample."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_json, table  # noqa: E402

st.set_page_config(page_title="Model comparison", layout="wide")
st.title("Model comparison")

comp = table("model_comparison.csv")
if comp is None:
    st.error("Run `make train` first.")
    st.stop()

sample = st.radio("Sample", ["train", "validation", "oot"], horizontal=True, index=2)
view = comp[comp["sample"] == sample].sort_values("gini", ascending=False)
st.dataframe(view.round(4), use_container_width=True, hide_index=True)
st.caption("Gini and KS use predicted PD as the score. Brier measures probability accuracy, lower is better.")

st.subheader("Predicted vs actual default rate by risk decile (out-of-time)")
dec = table("decile_oot_champion.csv")
if dec is not None:
    chart = dec.set_index("decile")[["mean_pd", "bad_rate"]] * 100
    st.bar_chart(chart)
    st.dataframe(dec.round(4), use_container_width=True, hide_index=True)

st.subheader("Tree tuning")
tuning = load_json("tuning.json")
if tuning:
    st.json({k: {"best_iteration": v["best_iteration"],
                 "validation_logloss": round(v["best_validation_logloss"], 5)} for k, v in tuning.items()})

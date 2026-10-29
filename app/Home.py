"""Overview: data funnel, sample design and headline results."""
import streamlit as st

from utils import load_json, table

st.set_page_config(page_title="Credit PD model", layout="wide")
st.title("Credit default PD model")
st.caption("Dashboard of saved outputs. Nothing on these pages trains or refits a model.")

summary = load_json("run_summary.json")
if summary is None:
    st.error("No results found. Run `make train` first.")
    st.stop()

verdict = load_json("champion_challenger.json") or {}
c1, c2, c3 = st.columns(3)
c1.metric("Champion Gini, out-of-time", f"{summary['champion_gini_oot']:.3f}")
c2.metric("Champion Gini, validation", f"{summary['champion_gini_validation']:.3f}")
c3.metric("Challenger replaces champion", "Yes" if summary["challenger_replace"] else "No")

st.subheader("Data funnel")
funnel = table("data_quality_funnel.csv")
if funnel is not None:
    st.dataframe(funnel, use_container_width=True, hide_index=True)

st.subheader("Sample design")
samples = table("sample_summary.csv")
if samples is not None:
    st.dataframe(samples, use_container_width=True, hide_index=True)

st.subheader("Champion-challenger verdict")
st.write(verdict.get("rule", ""))
st.json({k: verdict[k] for k in ["champion", "best_challenger_by_validation_gini",
                                  "oot_gini_gain_over_champion", "replace_champion"] if k in verdict})

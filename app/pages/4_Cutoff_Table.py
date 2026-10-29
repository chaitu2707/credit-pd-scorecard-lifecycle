"""Cut-off economics on the out-of-time sample, with the LGD chosen by the user."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import oot_predictions  # noqa: E402
from src import config  # noqa: E402
from src.validation import cutoff_table  # noqa: E402

st.set_page_config(page_title="Cut-off table", layout="wide")
st.title("Cut-off table (out-of-time)")

pred = oot_predictions()
if pred is None:
    st.error("Run `make train` first.")
    st.stop()

st.warning("Expected loss uses predicted PD. Out of time, predicted PD is about half of the observed default rate, "
           "so expected loss is understated: realised loss on approved loans is roughly 1.8 to 1.9 times the figure shown. "
           "The approval ranking is still valid. See the validation report, section 8.")
lgd = st.slider("Loss given default (LGD)", 0.20, 0.90, 0.60, 0.05)
model = st.radio("Score used", ["pd_champion", "pd_ensemble"], horizontal=True)
st.caption("LGD is an assumption, not a model output. The default of 60% is a placeholder. "
           "Approval rate and bad rate do not depend on LGD; expected loss does.")

table_out = cutoff_table(pred[model], pred["target"], pred["loan_amnt"], config.CUTOFF_GRID, [lgd])
st.dataframe(table_out.round(4), use_container_width=True, hide_index=True)

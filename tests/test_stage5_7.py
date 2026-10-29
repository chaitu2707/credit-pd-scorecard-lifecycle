import numpy as np
import pandas as pd
import pytest

from src import config, models as md, validation as va, monitoring as mo
from src.metrics import psi


def test_decile_table_counts_and_capture():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.2, 1000)
    y = (rng.random(1000) < p).astype(int)
    t = va.decile_table(y, p)
    assert len(t) == 10 and t["loans"].sum() == 1000
    assert t.loc[t["decile"] == 10, "mean_pd"].item() > t.loc[t["decile"] == 1, "mean_pd"].item()
    assert t.loc[t["decile"] == 1, "cum_bad_capture_from_top"].item() == pytest.approx(1.0)


def test_intercept_shift_matches_observed_rate():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.1, 5000)
    y = (rng.random(5000) < np.minimum(p * 2.5, 1)).astype(int)  # model under-predicts
    shift = va.intercept_shift(p, y)
    recal = va.apply_shift(p, shift)
    assert recal.mean() == pytest.approx(y.mean(), abs=1e-4)
    assert shift > 0


def test_cutoff_approval_rises_with_cutoff():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 0.2, 2000)
    y = (rng.random(2000) < p).astype(int)
    amt = rng.uniform(1000, 30000, 2000)
    t = va.cutoff_table(p, y, amt, [0.03, 0.06, 0.10], [0.6])
    assert list(t["approval_rate"]) == sorted(t["approval_rate"])
    assert all(0 <= x <= 1 for x in t["expected_loss_rate_lgd_60"].dropna())


def test_platt_maps_to_observed_rate():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.2, 4000)
    y = (rng.random(4000) < np.minimum(p * 2, 1)).astype(int)
    cal = md.fit_platt(p, y)
    out = md.apply_platt(cal, p)
    assert out.mean() == pytest.approx(y.mean(), rel=0.1)


def test_ensemble_is_simple_average():
    a, b = np.array([0.02, 0.10]), np.array([0.04, 0.20])
    assert md.ensemble_average([a, b]).tolist() == pytest.approx([0.03, 0.15])


def test_traffic_lights():
    assert mo.psi_light(0.05) == "green" and mo.psi_light(0.15) == "amber" and mo.psi_light(0.3) == "red"
    assert mo.calibration_light(0.04, 0.04) == "green"
    assert mo.calibration_light(0.07, 0.03) == "red"
    assert mo.gini_light(0.06) == "red"
    assert mo.overall_light(["green", "amber"]) == "amber"


def test_tree_frame_uses_fixed_levels():
    train = pd.DataFrame({"purpose": ["car", "credit_card", None], "fico": [1.0, 2.0, np.nan]})
    levels = {"purpose": ["car", "credit_card", "missing"]}
    X = md.make_tree_frame(train, ["purpose", "fico"], levels)
    assert list(X["purpose"].cat.categories) == ["car", "credit_card", "missing"]
    assert X["purpose"].iloc[2] == "missing"

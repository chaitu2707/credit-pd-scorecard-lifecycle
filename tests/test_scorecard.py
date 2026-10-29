import numpy as np
import pandas as pd

from src import config
from src.scorecard import (
    assign_bins, fit_scorecard, monotonic_check, pd_from_model, pd_from_score,
    reason_codes, score,
)


def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = -2.5 + 0.8 * x1 - 0.5 * x2
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


SPEC = {"x1": {"type": "numeric", "cuts": [-0.5, 0.5]},
        "x2": {"type": "numeric", "cuts": [-0.5, 0.5]}}


def test_points_recover_model_pd():
    tr, va = _synthetic(20000, 0), _synthetic(8000, 1)
    btr, _ = assign_bins(tr, SPEC)
    bva, _ = assign_bins(va, SPEC)
    fit = fit_scorecard(btr, tr["y"], bva, va["y"], c_grid=(1.0,))
    model_pd = pd_from_model(fit, bva)
    points_pd = pd_from_score(score(fit, bva))
    assert np.max(np.abs(model_pd - points_pd)) < 1e-9


def test_base_score_maps_to_base_odds():
    # at BASE_SCORE the good:bad odds should equal BASE_ODDS, so PD = 1 / (1 + 50)
    assert pd_from_score([config.BASE_SCORE])[0] == np.float64(1 / (1 + config.BASE_ODDS)) or \
        abs(pd_from_score([config.BASE_SCORE])[0] - 1 / (1 + config.BASE_ODDS)) < 1e-12


def test_coefficients_have_expected_sign():
    tr, va = _synthetic(20000, 2), _synthetic(8000, 3)
    btr, _ = assign_bins(tr, SPEC)
    bva, _ = assign_bins(va, SPEC)
    fit = fit_scorecard(btr, tr["y"], bva, va["y"], c_grid=(1.0,))
    assert all(c < 0 for c in fit["coef"].values())


def test_monotonic_check_detects_violation(monkeypatch):
    df = pd.DataFrame({"fico": np.linspace(600, 800, 1000), "target": np.r_[np.zeros(500), np.ones(500)]})
    spec = {"fico": {"type": "numeric", "cuts": [650, 700, 750]}}
    bins, orders = assign_bins(df, spec)
    # the data rises with FICO (bad rate goes 0 -> 1), so +1 is correct.
    # Forcing -1 (falling) must be flagged as a violation.
    monkeypatch.setitem(config.MONOTONIC_SIGN, "fico", -1)
    res = monotonic_check(bins, orders, df["target"], spec)
    assert res.loc[res["feature"] == "fico", "monotonic"].item() is False


def test_reason_codes_return_list_per_applicant():
    tr, va = _synthetic(5000, 4), _synthetic(2000, 5)
    btr, _ = assign_bins(tr, SPEC)
    bva, _ = assign_bins(va, SPEC)
    fit = fit_scorecard(btr, tr["y"], bva, va["y"], c_grid=(1.0,))
    codes = reason_codes(fit, bva, top_n=2)
    assert len(codes) == len(bva)
    assert all(isinstance(c, list) and len(c) <= 2 for c in codes)


def test_rare_categories_merge_into_other():
    df = pd.DataFrame({"purpose": ["car"] * 500 + ["credit_card"] * 9400 + ["wedding"] * 50 + ["x"] * 50})
    spec = {"purpose": {"type": "categorical", "keep": ["car", "credit_card"]}}
    bins, _ = assign_bins(df, spec)
    assert set(bins["purpose"].unique()) == {"car", "credit_card", "other"}
    assert (bins.loc[df["purpose"] == "wedding", "purpose"] == "other").all()


def test_reason_codes_exclude_configured_features(monkeypatch):
    tr, va = _synthetic(5000, 6), _synthetic(2000, 7)
    btr, _ = assign_bins(tr, SPEC)
    bva, _ = assign_bins(va, SPEC)
    fit = fit_scorecard(btr, tr["y"], bva, va["y"], c_grid=(1.0,))
    monkeypatch.setattr(config, "REASON_CODE_EXCLUDED", ["x1"])
    codes = reason_codes(fit, bva, top_n=2)
    assert all("x1" not in c for c in codes)

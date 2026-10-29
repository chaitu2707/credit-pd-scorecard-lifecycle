import numpy as np
import pandas as pd
import pytest

from src.feature_selection import (
    assert_no_leakage, correlated_pairs, information_value, iv_band,
    parse_emp_length, vif, woe_table,
)


def test_parse_emp_length():
    assert parse_emp_length("10+ years") == 10
    assert parse_emp_length("< 1 year") == 0
    assert parse_emp_length("3 years") == 3
    assert np.isnan(parse_emp_length(np.nan))


def test_iv_strong_for_separating_feature():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    x = y * 10 + rng.normal(0, 1, 4000)  # feature almost perfectly separates the classes
    df = pd.DataFrame({"x": x, "target": y})
    assert information_value(df, "x") > 0.3


def test_iv_small_for_noise():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=20000), "target": rng.integers(0, 2, 20000)})
    assert information_value(df, "x") < 0.02


def test_woe_missing_gets_own_bin():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0] * 50, "target": [0, 1, 1, 0] * 50})
    table = woe_table(df, "x", bins=2)
    assert "missing" in table["bin"].values


def test_iv_band_thresholds():
    assert iv_band(0.01) == "weak, drop"
    assert iv_band(0.2) == "medium"
    assert iv_band(0.6) == "very strong, leakage review"


def test_vif_flags_collinear_pair():
    rng = np.random.default_rng(2)
    a = rng.normal(size=1000)
    df = pd.DataFrame({"a": a, "b": a + rng.normal(0, 0.01, 1000), "c": rng.normal(size=1000)})
    v = vif(df, ["a", "b", "c"])
    assert v["a"] > 10 and v["b"] > 10 and v["c"] < 2


def test_correlated_pairs_detects_pair():
    rng = np.random.default_rng(3)
    a = rng.normal(size=500)
    df = pd.DataFrame({"a": a, "b": a * 2, "c": rng.normal(size=500)})
    pairs = correlated_pairs(df, ["a", "b", "c"], threshold=0.7)
    assert len(pairs) == 1


def test_leakage_assertion_blocks_grade_and_payments():
    with pytest.raises(ValueError):
        assert_no_leakage(["dti", "grade"])
    with pytest.raises(ValueError):
        assert_no_leakage(["dti", "total_pymnt"])
    assert_no_leakage(["dti", "fico"])  # clean set passes


from src.feature_selection import _bin_numeric


def test_low_cardinality_feature_is_not_collapsed():
    # term-like feature: 36 vs 60 months with different default rates
    rng = np.random.default_rng(4)
    term = rng.choice([36.0, 60.0], 5000)
    y = np.where(term == 60, rng.random(5000) < 0.2, rng.random(5000) < 0.02).astype(int)
    df = pd.DataFrame({"x": term, "target": y})
    assert information_value(df, "x") > 0.1


def test_zero_heavy_feature_keeps_zero_vs_positive_split():
    # count feature: 90% zeros; positives carry the risk
    rng = np.random.default_rng(5)
    n = 10000
    # positives spread over 1..30, so the feature has many distinct values and
    # the zero-heavy path (not the one-bin-per-value path) is exercised
    x = np.where(rng.random(n) < 0.9, 0.0, rng.integers(1, 31, n).astype(float))
    y = np.where(x > 0, rng.random(n) < 0.3, rng.random(n) < 0.03).astype(int)
    df = pd.DataFrame({"x": x, "target": y})
    assert information_value(df, "x") > 0.1
    bins = _bin_numeric(pd.Series(x), 5).astype(str)
    assert (bins == "(-inf, 0.0]").any()

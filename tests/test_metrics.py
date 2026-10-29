import numpy as np
import pytest

from src.metrics import brier_score, brier_skill_score, gini, ks_statistic


def test_brier_perfect_prediction_is_zero():
    y = np.array([0, 1, 1, 0])
    assert brier_score(y, y.astype(float)) == pytest.approx(0.0)


def test_brier_known_value():
    y = np.array([0, 1])
    p = np.array([0.2, 0.6])
    # (0.2^2 + 0.4^2) / 2 = 0.1
    assert brier_score(y, p) == pytest.approx(0.1)


def test_brier_skill_of_base_rate_predictor_is_zero():
    y = np.array([0, 0, 0, 1])
    p = np.full(4, y.mean())
    assert brier_skill_score(y, p) == pytest.approx(0.0)


def test_brier_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        brier_score([0, 1], [0.5, 1.2])


def test_gini_perfect_ranking_is_one():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert gini(y, s) == pytest.approx(1.0)


def test_ks_perfect_separation_is_one():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert ks_statistic(y, s) == pytest.approx(1.0)


from src.metrics import psi


def test_psi_identical_samples_is_near_zero():
    x = np.linspace(0, 1, 1000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-6)


def test_psi_shifted_sample_is_positive_and_large():
    rng = np.random.default_rng(0)
    expected = rng.normal(0, 1, 5000)
    actual = rng.normal(1, 1, 5000)
    assert psi(expected, actual) > 0.25


from src.eda import term_mix, missingness_by_year


def test_missingness_by_year_is_fraction():
    import pandas as pd
    df = pd.DataFrame({
        "issue_date": pd.to_datetime(["2015-01-01", "2015-06-01", "2016-01-01", "2016-06-01"]),
        "x": [1.0, None, None, None],
    })
    out = missingness_by_year(df, ["x"])
    assert out.loc["x", 2015] == pytest.approx(0.5)
    assert out.loc["x", 2016] == pytest.approx(1.0)


def test_term_mix_rows_sum_to_one():
    import pandas as pd
    df = pd.DataFrame({
        "issue_date": pd.to_datetime(["2015-01-01", "2015-06-01", "2016-01-01"]),
        "term": [" 36 months", " 60 months", " 36 months"],
    })
    mix = term_mix(df)
    assert mix.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])

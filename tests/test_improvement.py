import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsrs.improvement import (
    rolling_percentile, damped_rsrs, long_cash_state, adaptive_trend_rsrs,
)


def sample(n=900):
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    base = 100 + np.linspace(0, 60, n) + 2*np.sin(np.arange(n)/17)
    close = pd.Series(base, index=idx)
    return pd.DataFrame({
        "open": close, "high": close+1.5, "low": close-1,
        "close": close, "volume": 1e8, "ret": close.pct_change(),
    })


def test_rolling_percentile_has_hand_computed_answer():
    x = pd.Series([1., 3., 2., 4.])
    got = rolling_percentile(x, window=3, min_periods=2)
    assert np.isnan(got.iloc[0])
    assert np.isclose(got.iloc[1], 1.)
    assert np.isclose(got.iloc[2], 2/3)
    assert np.isclose(got.iloc[3], 1.)
    y=pd.Series([1.,2.,np.nan])
    assert np.isnan(rolling_percentile(y,window=3,min_periods=2).iloc[-1])


def test_damped_formula_matches_report_transformation():
    d = sample()
    q = damped_rsrs(d)
    expected = q.zscore * q.r2.pow(2*q.volatility_rank)
    assert np.allclose(q.damped_rsrs.dropna(), expected.dropna())
    assert q.volatility_rank.dropna().between(0, 1).all()


def test_state_machine_exits_before_same_day_reentry():
    idx = pd.date_range("2020-01-01", periods=5)
    entry = pd.Series([1, 0, 1, 1, 0], index=idx, dtype=bool)
    exit_ = pd.Series([0, 0, 1, 0, 1], index=idx, dtype=bool)
    got = long_cash_state(entry, exit_)
    assert got.tolist() == [1, 1, 0, 1, 0]


def test_append_future_does_not_change_past_improved_signal():
    d = sample(900)
    cut = 820
    short = adaptive_trend_rsrs(d.iloc[:cut])
    full = adaptive_trend_rsrs(d)
    pd.testing.assert_series_equal(
        short.position, full.position.iloc[:cut], check_names=False
    )


if __name__ == "__main__":
    tests = [v for k,v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test(); print(f"[PASS] {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")

"""RSRS 指标与回测的单元测试。

直接运行::

    python tests/test_rsrs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsrs.backtest import backtest, max_drawdown, sharpe
from rsrs.indicators import rsrs_beta, rsrs_signal, rsrs_zscore


def _price_index(n=800, seed=0):
    """构造一段带趋势与震荡的合成指数。"""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0002, 0.012, n)
    close = 1000 * np.cumprod(1 + ret)
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = pd.Series(close, index=dates)
    high = close * (1 + np.abs(rng.normal(0.004, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0.004, 0.003, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": 1.0})
    df["ret"] = df["close"] / df["close"].shift(1) - 1.0
    return df


def test_beta_is_linear_slope():
    """完美线性 high = 2*low + 1 时，beta 应精确等于 2。"""
    n = 200
    idx = pd.bdate_range("2020-01-01", periods=n)
    low = pd.Series(np.linspace(10, 20, n), index=idx)
    high = 2.0 * low + 1.0
    beta = rsrs_beta(high, low, n=18)
    assert np.allclose(beta.dropna(), 2.0, atol=1e-9), beta.dropna().describe()


def test_zscore_standardizes():
    beta = pd.Series(np.sin(np.arange(1000) / 20.0) + 5.0)
    z = rsrs_zscore(beta, m=100)
    assert z.isna().sum() == 99  # 前 m-1 个为 NaN
    assert np.allclose(z.dropna().mean(), 0.0, atol=0.3)  # 近似，因为 sin 周期性


def test_signal_state_machine():
    z = pd.Series([0.0, 0.8, 0.5, -0.9, -0.5, 1.0, 0.2])
    sig = rsrs_signal(z, s_buy=0.7, s_sell=-0.7, initial=0.0)
    assert list(sig) == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], list(sig)


def test_signal_holds_through_nan():
    z = pd.Series([0.9, np.nan, np.nan, -0.8, np.nan])
    sig = rsrs_signal(z, s_buy=0.7, s_sell=-0.7)
    assert list(sig) == [1.0, 1.0, 1.0, 0.0, 0.0], list(sig)


def test_backtest_lag_prevents_lookahead():
    """lag=0 必须报错；lag=1 时决策日次日才生效。"""
    df = _price_index()
    # 构造一个「当天涨就满仓」的理想信号，验证滞后生效
    sig = (df["ret"].shift(-1) > 0).astype(float)  # 故意用未来信息，仅测试机械滞后
    try:
        backtest(df, sig, lag=0)
    except ValueError:
        pass
    else:
        raise AssertionError("lag=0 应报错")

    res = backtest(df, sig, lag=1)
    # 生效仓位比决策信号滞后一天
    assert (res.position == sig.shift(1).fillna(0.0)).all()


def test_metrics_bounds():
    ret = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 1000))
    eq = (1 + ret).cumprod()
    assert -1.0 < max_drawdown(eq) <= 0.0
    assert np.isfinite(sharpe(ret))


def test_cost_reduces_return():
    df = _price_index(n=2000)
    sig = pd.Series(1.0, index=df.index)  # 永远满仓
    gross = backtest(df, sig, lag=1, cost=0.0)
    net = backtest(df, sig, lag=1, cost=0.001)
    # 满仓切到满仓无换手，两者应几乎一致；再构造换手场景对比
    sig2 = (pd.Series(np.arange(len(df)) % 60 < 30, index=df.index)).astype(float)
    g2 = backtest(df, sig2, lag=1, cost=0.0)
    n2 = backtest(df, sig2, lag=1, cost=0.001)
    assert g2.equity.iloc[-1] > n2.equity.iloc[-1]


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append((fn.__name__, repr(e)))
            print(f"  [FAIL] {fn.__name__}: {e!r}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())

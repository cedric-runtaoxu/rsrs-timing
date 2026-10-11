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
from rsrs.indicators import rsrs_beta, rsrs_family, rsrs_r2, rsrs_signal, rsrs_zscore
from rsrs.strategies import apply_entry_filter, ma_filter, volume_corr_filter


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


def test_zscore_warmup_convention():
    """预热口径：不足 M 时用已有数据（原报告第 22 页），而不是强制凑满整窗。

    这是本项目最关键的一处口径修正。若要求凑满 M 个观测，回测前 600 个交易日
    将无信号，标准分策略年化会从 25.6% 掉到 12.2%，无法复现原报告。
    """
    idx = pd.bdate_range("2000-01-01", periods=1000)
    beta = pd.Series(np.arange(1000, dtype=float) + np.sin(np.arange(1000) / 20.0), index=idx)

    z_expanding = rsrs_zscore(beta, m=600, min_periods=10)
    assert int(z_expanding.isna().sum()) == 9, "扩展口径下前 9 个为空"
    assert z_expanding.first_valid_index() == idx[9]

    z_full = rsrs_zscore(beta, m=600, min_periods=600)
    assert int(z_full.isna().sum()) == 599, "满窗口径下前 599 个为空"


def test_zscore_standardizes():
    beta = pd.Series(np.sin(np.arange(1000) / 20.0) + 5.0)
    z = rsrs_zscore(beta, m=100, min_periods=100)
    assert z.isna().sum() == 99  # 满窗口径下前 m-1 个为 NaN
    assert np.allclose(z.dropna().mean(), 0.0, atol=0.3)  # 近似，因为 sin 周期性


def test_r2_and_derived_indicators():
    """R² ∈ [0,1]；修正标准分 = z×R²；右偏标准分 = 修正标准分×beta。"""
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(3)
    low = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)
    high = 1.05 * low + rng.normal(0, 0.5, n)

    r2 = rsrs_r2(high, low, n=18)
    assert r2.min() >= -1e-12 and r2.max() <= 1 + 1e-12

    fam = rsrs_family(pd.DataFrame({"high": high, "low": low}), n=18, m=100)
    assert np.allclose((fam["zscore"] * fam["r2"]).dropna(), fam["modified"].dropna())
    assert np.allclose((fam["modified"] * fam["beta"]).dropna(), fam["right_skewed"].dropna())


def test_slope_strategy_thresholds():
    """斜率策略用非对称阈值（买入 1.0 / 卖出 0.8），与原报告第 2.1 节一致。"""
    beta = pd.Series([0.5, 1.1, 0.9, 0.85, 0.7, 1.2])
    sig = rsrs_signal(beta, s_buy=1.0, s_sell=0.8)
    # 1.1>1.0 开仓；0.9/0.85 落在 0.8~1.0 之间 -> 维持持仓；0.7<0.8 平仓；1.2 再开仓
    assert list(sig) == [0.0, 1.0, 1.0, 1.0, 0.0, 1.0], list(sig)


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


def test_ma_filter_uses_lagged_ma():
    """价格过滤必须用「前一日 MA」与「前三日 MA」，不能用当日 MA（会引入额外信息）。"""
    idx = pd.bdate_range("2020-01-01", periods=60)
    close = pd.Series(np.linspace(100, 200, 60), index=idx)   # 单调上涨
    ok = ma_filter(close, window=20)
    assert bool(ok.loc[idx[25]:].all()), "单调上涨时前一日 MA 应始终高于前三日 MA"

    close_down = pd.Series(np.linspace(200, 100, 60), index=idx)
    ok_down = ma_filter(close_down, window=20)
    assert not bool(ok_down.loc[idx[25]:].any())


def test_entry_filter_blocks_buys_only():
    """开仓过滤只影响买点，不影响卖点。"""
    base = pd.Series([0, 1, 1, 1, 0, 1, 1], dtype="float64")
    allow = pd.Series([True, False, False, True, True, False, True])
    out = apply_entry_filter(base, allow)
    # 第2、3 天想开仓但被拦 -> 空仓；第4 天放行 -> 开仓；第5 天信号转空 -> 无条件平仓
    assert list(out) == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], list(out)


def test_volume_corr_filter_shape():
    idx = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(7)
    vol = pd.Series(rng.lognormal(0, 0.5, 50), index=idx)
    ind = pd.Series(rng.normal(0, 1, 50), index=idx)
    f = volume_corr_filter(vol, ind, window=10)
    assert f.dtype == bool
    assert f.iloc[:9].sum() == 0            # 前 9 天不足窗口
    assert int(f.sum()) > 0


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

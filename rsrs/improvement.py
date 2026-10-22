"""RSRS失效诊断与改进策略的纯计算函数。

本模块与原研报复现隔离。所有函数只使用当前日及过去数据；交易仍由
``backtest``统一按次日收盘执行。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import rsrs_family, rsrs_signal


def rolling_percentile(
    series: pd.Series,
    window: int = 756,
    min_periods: int = 252,
) -> pd.Series:
    """当前值在截至当日滚动窗口中的经验分位数，范围(0,1]。"""
    if window < 2 or not 2 <= min_periods <= window:
        raise ValueError("window与min_periods必须满足 2 <= min_periods <= window")

    def last_rank(values: np.ndarray) -> float:
        if not np.isfinite(values[-1]):
            return np.nan
        current=values[-1]
        values = values[np.isfinite(values)]
        return float(np.mean(values <= current)) if len(values) else np.nan

    return series.rolling(window, min_periods=min_periods).apply(
        last_rank, raw=True
    ).rename("rolling_percentile")


def damped_rsrs(
    df: pd.DataFrame,
    n: int = 18,
    m: int = 600,
    min_periods: int = 10,
) -> pd.DataFrame:
    """波动率钝化RSRS：在高波动状态下提高拟合优度惩罚。

    实现为 ``z * (R²)**(2*q)``，其中q是N日收益波动率在过去M日的
    经验分位数。q越高，低拟合优度信号被压缩得越明显。
    """
    family = rsrs_family(df, n=n, m=m, min_periods=min_periods)
    volatility = df["ret"].rolling(n, min_periods=n).std(ddof=1)
    volatility_rank = rolling_percentile(
        volatility, window=m, min_periods=min_periods
    )
    indicator = (
        family["zscore"] * family["r2"].pow(2.0 * volatility_rank)
    ).rename("damped_rsrs")
    return family.assign(
        return_volatility=volatility,
        volatility_rank=volatility_rank,
        damped_rsrs=indicator,
    )


def long_cash_state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """开仓/平仓条件驱动的0/1状态机；同日冲突时已有仓位优先退出。"""
    entry, exit_ = entry.align(exit_, join="outer", fill_value=False)
    out = np.zeros(len(entry), dtype="float64")
    state = 0.0
    for i, (enter, leave) in enumerate(zip(entry.fillna(False), exit_.fillna(False))):
        if state == 0.0 and bool(enter):
            state = 1.0
        elif state == 1.0 and bool(leave):
            state = 0.0
        out[i] = state
    return pd.Series(out, index=entry.index, name="target_position")


def adaptive_trend_rsrs(
    df: pd.DataFrame,
    indicator: pd.Series | None = None,
    rank_window: int = 756,
    rank_min_periods: int = 252,
    entry_quantile: float = 0.80,
    exit_quantile: float = 0.20,
    trend_window: int = 250,
    trend_slope_gap: int = 20,
) -> pd.DataFrame:
    """状态自适应RSRS：滚动分位数 + 长期趋势 + 非对称退出。

    开仓：RSRS位于过去窗口前20%，价格在长期均线上，且均线较20日前上升。
    平仓：RSRS落入后20%，或价格跌破长期均线。过滤数据缺失时不建仓。
    """
    if not 0 < exit_quantile < entry_quantile < 1:
        raise ValueError("必须满足 0 < exit_quantile < entry_quantile < 1")
    if indicator is None:
        indicator = rsrs_family(df, n=18, m=600)["right_skewed"]
    rank = rolling_percentile(indicator, rank_window, rank_min_periods)
    ma = df["close"].rolling(trend_window, min_periods=trend_window).mean()
    price_above = df["close"] > ma
    trend_up = ma > ma.shift(trend_slope_gap)
    entry = (rank >= entry_quantile) & price_above & trend_up
    exit_ = (rank <= exit_quantile) | (~price_above)
    position = long_cash_state(entry, exit_)
    return pd.DataFrame({
        "indicator": indicator,
        "indicator_rank": rank,
        "trend_ma": ma,
        "price_above_trend": price_above,
        "trend_up": trend_up,
        "entry_condition": entry,
        "exit_condition": exit_,
        "position": position,
    })


def volatility_damped_signal(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """固定18/600/±0.7参数的波动率钝化信号。"""
    details = damped_rsrs(df, n=18, m=600, min_periods=10)
    return rsrs_signal(details["damped_rsrs"], .7, -.7), details

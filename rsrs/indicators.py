"""RSRS 指标计算。

时间轴约定（重要，避免未来函数）
--------------------------------
``beta_t`` 只用第 ``t`` 日及之前的最高价 / 最低价计算；
``z_t`` 只用第 ``t`` 日及之前的 ``beta`` 序列标准化。
信号在 ``t`` 日**收盘**才可知（因为要用到当日最高最低价），
因此回测里实际仓位滞后一日生效（见 ``backtest.backtest`` 的 ``lag`` 参数）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsrs_beta(high: pd.Series, low: pd.Series, n: int = 18) -> pd.Series:
    """滚动 OLS 斜率：``high = alpha + beta * low``。

    用滚动协方差 / 滚动方差解析式求解，等价于逐窗最小二乘，
    但向量化后 O(N) 完成，不依赖 scipy / statsmodels。

    Parameters
    ----------
    n : 回归窗口长度（交易日）。光大原始版取 18。
    """
    cov = high.rolling(n, min_periods=n).cov(low)
    var = low.rolling(n, min_periods=n).var()
    beta = cov / var
    beta = beta.replace([np.inf, -np.inf], np.nan)
    return beta.rename("beta")


def rsrs_zscore(beta: pd.Series, m: int = 600) -> pd.Series:
    """把 ``beta`` 用过去 ``m`` 日的均值与标准差做标准化。

    标准化窗口 ``m`` 远大于回归窗口 ``n``，这样 ``z`` 衡量的才是
    「当前斜率相对其长期中枢的偏离」，而不是短期噪声。
    """
    mu = beta.rolling(m, min_periods=m).mean()
    sd = beta.rolling(m, min_periods=m).std(ddof=1)
    z = (beta - mu) / sd.replace(0.0, np.nan)
    return z.rename("zscore")


def rsrs_signal(
    z: pd.Series,
    s_buy: float = 0.7,
    s_sell: float = -0.7,
    initial: float = 0.0,
) -> pd.Series:
    """由 ``z`` 产生 0/1 仓位信号（带状态记忆）。

    * ``z > s_buy``  → 满仓（1）
    * ``z < s_sell`` → 空仓（0）
    * 其余 → 维持上一状态

    「维持」这一条很关键：它让策略只在强信号出现时才交易，
    显著降低换手率，避免在阈值附近来回打脸。
    """
    zvals = z.to_numpy(dtype="float64")
    out = np.empty(len(z), dtype="float64")
    state = float(initial)
    for i in range(len(zvals)):
        v = zvals[i]
        if not np.isnan(v):
            if v > s_buy:
                state = 1.0
            elif v < s_sell:
                state = 0.0
        out[i] = state
    return pd.Series(out, index=z.index, name="position")


def run_rsrs(
    df: pd.DataFrame,
    n: int = 18,
    m: int = 600,
    s_buy: float = 0.7,
    s_sell: float = -0.7,
) -> pd.DataFrame:
    """一步到位：输入指数日行情，输出含 beta / zscore / position 的表。"""
    beta = rsrs_beta(df["high"], df["low"], n=n)
    z = rsrs_zscore(beta, m=m)
    position = rsrs_signal(z, s_buy=s_buy, s_sell=s_sell)
    return pd.DataFrame({"beta": beta, "zscore": z, "position": position})

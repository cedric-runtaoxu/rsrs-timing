"""RSRS 指标计算（覆盖原报告的四个基础指标）。

时间轴约定（重要，避免未来函数）
--------------------------------
``beta_t`` 只用第 ``t`` 日及之前的最高价 / 最低价计算；
``z_t`` 只用第 ``t`` 日及之前的 ``beta`` 序列标准化。
信号在 ``t`` 日**收盘**才可知（因为要用到当日最高最低价），
因此回测里实际仓位滞后一日生效（见 ``backtest.backtest`` 的 ``lag`` 参数）。

原报告的四个基础指标（光大证券 2017-05-01，第 1.4、2.1、2.3 节）
----------------------------------------------------------------
1. **斜率**      ``beta``，回归 ``high = alpha + beta * low`` 的斜率项；
2. **标准分**    ``z = (beta - mean) / std``，窗口 M；
3. **修正标准分** ``z * R2``，用回归决定系数惩罚拟合质量差的信号；
4. **右偏标准分** ``修正标准分 * beta``，利用 beta 恒正使分布右偏。

报告参数（第 2.1、2.3 节）：
    斜率策略         N = 18，阈值 买入 1.0 / 卖出 0.8
    标准分族         N = 18，M = 600，阈值 ±0.7
    右偏标准分       N = 16，M = 300，阈值 ±0.7

标准化窗口的预热口径（报告第 22 页原文）
----------------------------------------
「如若 M=600，那么在回测中前 600 天并没有 600 个斜率数据来计算标准分，
比如**回测第 50 天就只使用了 50 个斜率数据**计算标准分指标。」

即：窗口长度上限为 M，不足 M 时用**已有的全部**数据（扩展窗口）。
本模块以 ``min_periods`` 参数实现该口径，默认 10（即最少 10 个观测即可标准化）。
若强行要求凑满 M 个观测（``min_periods=m``），回测前 600 个交易日将无信号，
无法复现报告结果 —— 这是本项目早期版本的错误，已修正。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 标准化窗口的预热口径：不足 M 时用已有数据，最少需要这么多个斜率观测
DEFAULT_MIN_PERIODS = 10


def rsrs_beta(high: pd.Series, low: pd.Series, n: int = 18) -> pd.Series:
    """滚动 OLS 斜率：``high = alpha + beta * low``。

    用滚动协方差 / 滚动方差解析式求解，等价于逐窗最小二乘，
    但向量化后 O(N) 完成，不依赖 scipy / statsmodels。

    Parameters
    ----------
    n : 回归窗口长度（交易日）。原报告斜率策略与标准分族取 18，右偏标准分取 16。
    """
    cov = high.rolling(n, min_periods=n).cov(low)
    var = low.rolling(n, min_periods=n).var()
    beta = (cov / var).replace([np.inf, -np.inf], np.nan)
    return beta.rename("beta")


def rsrs_r2(high: pd.Series, low: pd.Series, n: int = 18) -> pd.Series:
    """滚动回归的决定系数 R²（用于构造修正标准分）。

    一元 OLS 下 ``R² = cov(high, low)² / (var(high) · var(low))``，
    同样有解析式，不需要逐窗跑回归。
    """
    cov = high.rolling(n, min_periods=n).cov(low)
    vh = high.rolling(n, min_periods=n).var()
    vl = low.rolling(n, min_periods=n).var()
    r2 = (cov ** 2) / (vh * vl)
    return r2.clip(lower=0.0, upper=1.0).rename("r2")


def rsrs_zscore(
    beta: pd.Series,
    m: int = 600,
    min_periods: int = DEFAULT_MIN_PERIODS,
    exclude_current: bool = False,
) -> pd.Series:
    """把 ``beta`` 用过去 ``m`` 日的均值与标准差做标准化（标准分）。

    Parameters
    ----------
    m : 标准化窗口长度上限。
    min_periods : 窗口内最少观测数。**默认 10，即不足 M 时用已有数据**，
        对应原报告第 22 页所述的预热口径。设成 ``m`` 会强制凑满整窗，
        导致前 M 个交易日无信号。
    exclude_current : 是否把当期 beta 排除在均值/标准差之外（即 ``shift(1)``）。
        原报告表述为「取前 M 日的斜率时间序列……计算当日斜率的标准分」，
        倾向于不含当期；两种口径差异很小，本项目默认包含当期，
        并提供该开关用于敏感性检验。
    """
    mu = beta.rolling(m, min_periods=min_periods).mean()
    sd = beta.rolling(m, min_periods=min_periods).std(ddof=1)
    if exclude_current:
        mu, sd = mu.shift(1), sd.shift(1)
    z = (beta - mu) / sd.replace(0.0, np.nan)
    return z.rename("zscore")


def rsrs_family(
    df: pd.DataFrame,
    n: int = 18,
    m: int = 600,
    min_periods: int = DEFAULT_MIN_PERIODS,
    exclude_current: bool = False,
) -> pd.DataFrame:
    """一次算出标准分族的全部指标。

    Returns
    -------
    DataFrame，列依次为：
        ``beta``          斜率
        ``r2``            回归决定系数
        ``zscore``        标准分 = (beta - mean) / std
        ``modified``      修正标准分 = 标准分 × R²
        ``right_skewed``  右偏标准分 = 修正标准分 × beta

    右偏标准分在原报告中使用 N=16、M=300，调用时传入对应参数即可。
    """
    beta = rsrs_beta(df["high"], df["low"], n=n)
    r2 = rsrs_r2(df["high"], df["low"], n=n)
    z = rsrs_zscore(beta, m=m, min_periods=min_periods, exclude_current=exclude_current)
    modified = (z * r2).rename("modified")
    right_skewed = (modified * beta).rename("right_skewed")
    return pd.DataFrame({
        "beta": beta,
        "r2": r2,
        "zscore": z,
        "modified": modified,
        "right_skewed": right_skewed,
    })


def rsrs_signal(
    z: pd.Series,
    s_buy: float = 0.7,
    s_sell: float = -0.7,
    initial: float = 0.0,
) -> pd.Series:
    """由指标值产生 0/1 仓位信号（带状态记忆）。

    * 指标 > ``s_buy``  → 满仓（1）
    * 指标 < ``s_sell`` → 空仓（0）
    * 其余 → 维持上一状态

    「维持」这一条与原报告第 2 章的阈值交易框架一致：只在穿越阈值时改变仓位。
    该函数对斜率和标准分同样适用 —— 斜率策略只是把阈值换成
    ``s_buy=1.0``、``s_sell=0.8``（原报告第 2.1 节）。
    """
    zvals = z.to_numpy(dtype="float64")
    out = np.empty(len(zvals), dtype="float64")
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
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> pd.DataFrame:
    """便捷入口：输出含 beta / r2 / 标准分 / 修正标准分 / 右偏标准分 / position 的表。"""
    fam = rsrs_family(df, n=n, m=m, min_periods=min_periods)
    position = rsrs_signal(fam["zscore"], s_buy=s_buy, s_sell=s_sell)
    return fam.assign(position=position)

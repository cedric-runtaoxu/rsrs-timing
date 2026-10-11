"""原报告第 3 章的优化策略：均线过滤与成交量相关性过滤。

原报告第 3.1 节「价格优化交易策略」
-----------------------------------
1. 计算 RSRS 标准分指标买卖信号；
2. 如果指标发出买入信号，**同时满足前一日 MA(20) 的值大于前三日 MA(20) 的值**，则买入；
3. 如果指标发出卖出信号，则卖出手中持股。

原报告第 3.2 节「交易量相关性优化交易策略」
-------------------------------------------
1. 计算 RSRS 标准分指标买卖信号；
2. 如果指标发出买入信号，**同时满足前 10 日交易量与修正标准分之间的相关性为正**，则买入；
3. 如果指标发出卖出信号，则卖出手中持股。

两者的共同结构是：**只过滤开仓，不过滤平仓**。原报告对此的解释是 ——
RSRS 大概率左侧开仓、左侧平仓，做多策略中「左侧开仓预测错误」的危害远大于
「左侧平仓预测错误」，因此只需过滤买点。

原报告第 22 页说明，表 5、表 6 的优化策略统一使用 N=18、M=600、S=0.7 参数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ma_filter(close: pd.Series, window: int = 20) -> pd.Series:
    """原报告的价格过滤条件：前一日 MA(window) > 前三日 MA(window)。

    注意是「前一日」与「前三日」，不是当日 —— 当日 MA 依赖当日收盘价，
    若用当日 MA 会引入额外的信息，与原报告口径不符。
    """
    up, _ = ma_direction(close, window)
    return up.rename("ma_filter")


def ma_direction(close: pd.Series, window: int = 20):
    """返回 ``(均线上行, 均线下行)`` 两个布尔序列。

    上行定义为 前一日 MA > 前三日 MA；下行为 前一日 MA < 前三日 MA。
    原报告的多空策略在第 2、4 条分别使用这两个方向（报告 3.1 节）。
    """
    ma = close.rolling(window, min_periods=window).mean()
    a, b = ma.shift(1), ma.shift(3)
    return (a > b).rename("ma_up"), (a < b).rename("ma_down")


def long_short_position(
    indicator: pd.Series,
    ma_up: pd.Series,
    ma_down: pd.Series,
    s_buy: float = 0.7,
    s_sell: float = -0.7,
) -> pd.Series:
    """原报告第 3.1 节的多空策略（含第 4、5 条卖空规则），产出 ±1 仓位。

    规则：

    1. 指标 > ``s_buy`` 且均线上行 → 持多（+1）
    2. 指标 < ``s_sell`` → 平多
    3. 指标 < ``s_sell`` 且均线下行 → 持空（-1）
    4. 指标 > ``s_buy`` → 平空

    与只做多版本相同，开仓受均线方向约束，平仓不受约束。
    """
    v = indicator.to_numpy(dtype="float64")
    up = ma_up.reindex(indicator.index).fillna(False).to_numpy(dtype=bool)
    dn = ma_down.reindex(indicator.index).fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(v), dtype="float64")
    state = 0.0
    for i in range(len(v)):
        x = v[i]
        if not np.isnan(x):
            if x > s_buy:                 # 买入信号
                if state < 0:
                    state = 0.0           # 买入平空仓
                if up[i]:
                    state = 1.0           # 开多
            elif x < s_sell:              # 卖出信号
                if state > 0:
                    state = 0.0           # 卖出平多
                if dn[i]:
                    state = -1.0          # 开空
        out[i] = state
    return pd.Series(out, index=indicator.index, name="position")


def volume_corr_filter(
    volume: pd.Series,
    indicator: pd.Series,
    window: int = 10,
) -> pd.Series:
    """原报告的成交量过滤条件：前 ``window`` 日成交量与修正标准分的相关性为正。"""
    v = volume.astype("float64")
    ind = indicator.astype("float64")
    corr = v.rolling(window, min_periods=window).corr(ind)
    return (corr > 0).rename("volume_corr_filter")


def apply_entry_filter(
    base_position: pd.Series,
    allow_entry: pd.Series,
) -> pd.Series:
    """在基础信号之上只限制开仓，平仓不受影响。

    Parameters
    ----------
    base_position : 未过滤的 0/1 仓位信号。
    allow_entry : 布尔序列，True 表示当日允许开仓。

    Notes
    -----
    当指标发出买入信号但被过滤时保持空仓；此后只要指标仍处于「应持仓」状态
    且过滤器转为允许，即可开仓 —— 这与原报告「发出买入信号且满足条件则买入」
    的表述一致。
    """
    p = base_position.to_numpy(dtype="float64")
    a = allow_entry.reindex(base_position.index).fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(p), dtype="float64")
    state = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            if state == 0.0 and a[i]:
                state = 1.0          # 开仓信号 + 过滤器放行
            # 已持仓或过滤器未放行：维持
        else:
            state = 0.0              # 平仓信号：无条件平仓
        out[i] = state
    return pd.Series(out, index=base_position.index, name="position")

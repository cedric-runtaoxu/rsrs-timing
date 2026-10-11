"""回测与绩效统计。

交易假设
--------
* 仓位：满仓（1）/ 空仓（0），无杠杆、无做空。
* 成交：信号在 t 日收盘产生，实际在 t + ``lag`` 日生效（默认 lag=1，次日收盘成交）。
* 成本：每次仓位变化收取 ``cost`` 的单边费率（默认 0.1%，
  对应沪深300 ETF 的近似双边摩擦，或股指期货的滑点估计）。
* 基准：同一指数买入持有（buy & hold）。

这些假设都要写进复现报告 —— 择时策略的收益对「成交日」和「成本」非常敏感。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    ret: pd.Series
    """策略日收益（已扣成本，含空仓 0 收益）。"""
    benchmark_ret: pd.Series
    position: pd.Series
    """实际生效仓位（已按 lag 滞后）。"""
    equity: pd.Series
    benchmark_equity: pd.Series
    n_trades: int
    """仓位变化次数（一次买卖切换 = 1）。"""
    params: dict = field(default_factory=dict)


def backtest(
    df: pd.DataFrame,
    position: pd.Series,
    lag: int = 1,
    cost: float = 0.001,
    benchmark: str = "buy_hold",
) -> BacktestResult:
    """按 ``position``（0/1，t 日收盘决策）回测。

    Parameters
    ----------
    lag : 信号日到生效日的滞后。**设成 0 就是未来函数**（用了当日高低价却当日成交）。
    cost : 单边交易费率，仓位变化当天扣除。
    """
    if lag < 1:
        raise ValueError("lag 必须 >= 1，否则存在未来函数")

    pos = position.reindex(df.index).ffill().fillna(0.0)
    # 生效仓位 = 决策信号滞后 lag 日
    pos_effective = pos.shift(lag).fillna(0.0)

    ret = df["ret"].fillna(0.0)
    # 交易成本发生在仓位切换当天（以生效仓位为准）
    turnover = pos_effective.diff().abs().fillna(0.0)
    cost_series = turnover * cost

    strat_ret = pos_effective * ret - cost_series
    bench_ret = ret

    equity = (1.0 + strat_ret).cumprod()
    bench_equity = (1.0 + bench_ret).cumprod()

    n_trades = int((pos.diff().ne(0)).sum())

    return BacktestResult(
        ret=strat_ret,
        benchmark_ret=bench_ret,
        position=pos_effective,
        equity=equity,
        benchmark_equity=bench_equity,
        n_trades=n_trades,
        params={"lag": lag, "cost": cost, "benchmark": benchmark},
    )


# ------------------------------------------------------------------ 绩效指标


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def annualized_return(ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    years = len(ret) / periods
    growth = float((1.0 + ret).prod())
    return growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else np.nan


def annualized_vol(ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(ret.std(ddof=1) * np.sqrt(periods))


def sharpe(ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    sd = ret.std(ddof=1)
    return float(ret.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def calmar(ret: pd.Series, equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    return annualized_return(ret) / abs(mdd) if mdd < 0 else np.nan


def win_rate(ret: pd.Series) -> float:
    return float((ret > 0).mean())


def excess_return(ret: pd.Series, benchmark_ret: pd.Series) -> pd.Series:
    return ret - benchmark_ret


def summarize(result: BacktestResult) -> Dict[str, float]:
    """汇总成报告里那张指标表。"""
    ann = annualized_return(result.ret)
    mdd = max_drawdown(result.equity)
    return {
        "累计收益": total_return(result.equity),
        "年化收益": ann,
        "年化波动": annualized_vol(result.ret),
        "夏普比率": sharpe(result.ret),
        "最大回撤": mdd,
        "Calmar": calmar(result.ret, result.equity),
        "胜率(日)": win_rate(result.ret),
        "基准累计收益": total_return(result.benchmark_equity),
        "基准年化收益": annualized_return(result.benchmark_ret),
        "基准最大回撤": max_drawdown(result.benchmark_equity),
        "超额年化收益": annualized_return(result.ret) - annualized_return(result.benchmark_ret),
        "交易次数": float(result.n_trades),
    }


def summary_frame(summary: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"指标": k, "数值": v} for k, v in summary.items()])

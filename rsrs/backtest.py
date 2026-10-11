"""回测与绩效统计。

交易假设
--------
* 仓位：满仓（1）/ 空仓（0），无杠杆、无做空。
* 成交：信号在 t 日收盘产生，实际在 t + ``lag`` 日生效（默认 lag=1，次日收盘成交）。
* 成本：每次仓位变化收取 ``cost`` 的单边费率；原报告按**双边** 0.4% / 0.6% 列示，
  本模块的 ``cost`` 是单边费率，双边 0.4% 对应单边 0.002。
* 基准：同一指数买入持有（buy & hold）。

关于「交易次数」的口径
----------------------
原报告表 2 给出 持仓总天数 ÷ 交易次数 = 平均持仓天数
（例如标准分策略 1353 ÷ 52 = 26.02），说明报告的「交易次数」指**建仓次数**
（即完整买卖回合数），而不是仓位变化次数。本模块同时报告两个口径：

* ``n_trades``   建仓次数（等价于完整回合数），与原报告可比；
* ``n_switches`` 仓位变化次数（建仓 + 平仓），约为前者的两倍。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

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
    """建仓次数（完整买卖回合数），与原报告口径一致。"""
    n_switches: int
    """仓位变化次数（建仓 + 平仓）。"""
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    """每笔交易的明细：start / end / days / ret（不计成本）。"""
    params: dict = field(default_factory=dict)


def trade_table(pos: pd.Series, gross_ret: pd.Series) -> pd.DataFrame:
    """把连续的同向持仓段切成一笔笔交易，返回每笔的区间收益（不计成本）。

    支持多空：做空段（position = -1）的收益按 ``-1 × 指数收益`` 复合。
    """
    p = pos.reindex(gross_ret.index).fillna(0.0).to_numpy()
    r = gross_ret.to_numpy(dtype="float64")
    idx = gross_ret.index
    rows = []
    i, n = 0, len(p)
    while i < n:
        if p[i] != 0.0:
            j = i
            while j + 1 < n and p[j + 1] == p[i]:
                j += 1
            seg = p[i] * r[i:j + 1]
            rows.append({
                "start": idx[i],
                "end": idx[j],
                "days": j - i + 1,
                "side": "long" if p[i] > 0 else "short",
                "ret": float(np.prod(1.0 + seg) - 1.0),
            })
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows)


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
    pos_effective = pos.shift(lag).fillna(0.0)

    gross_ret = df["ret"].fillna(0.0)
    turnover = pos_effective.diff().abs().fillna(0.0)
    cost_series = turnover * cost

    strat_ret = pos_effective * gross_ret - cost_series
    bench_ret = gross_ret

    trades = trade_table(pos_effective, pos_effective * gross_ret)

    return BacktestResult(
        ret=strat_ret,
        benchmark_ret=bench_ret,
        position=pos_effective,
        equity=(1.0 + strat_ret).cumprod(),
        benchmark_equity=(1.0 + bench_ret).cumprod(),
        n_trades=int(len(trades)),
        n_switches=int(pos_effective.diff().ne(0).sum()),
        trades=trades,
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
    """汇总成与原报告表 2 / 表 4 可直接对照的指标表。"""
    ann = annualized_return(result.ret)
    mdd = max_drawdown(result.equity)
    pos = result.position.fillna(0.0)
    holding_days = int((pos != 0.0).sum())
    long_days = int((pos > 0).sum())
    short_days = int((pos < 0).sum())
    n_trades = int(result.n_trades)

    # 按天统计：只在持仓日内计算获利/亏损天数与胜率（与原报告表 2 口径一致）
    # 多空通用：单日损益 = 仓位 × 指数收益
    gross = (pos * result.benchmark_ret).loc[pos != 0.0]
    gain_days = int((gross > 0).sum())
    loss_days = int((gross < 0).sum())

    out: Dict[str, float] = {
        "累计收益": total_return(result.equity),
        "年化收益": ann,
        "年化波动": annualized_vol(result.ret),
        "夏普比率": sharpe(result.ret),
        "最大回撤": mdd,
        "Calmar": calmar(result.ret, result.equity),
        "持仓总天数": float(holding_days),
        "多头持仓天数": float(long_days),
        "空头持仓天数": float(short_days),
        "交易次数": float(n_trades),
        "平均持仓天数": holding_days / n_trades if n_trades else np.nan,
        "仓位切换次数": float(result.n_switches),
        "获利天数": float(gain_days),
        "亏损天数": float(loss_days),
        "胜率(按天)": gain_days / (gain_days + loss_days) if (gain_days + loss_days) else np.nan,
        "基准累计收益": total_return(result.benchmark_equity),
        "基准年化收益": annualized_return(result.benchmark_ret),
        "基准最大回撤": max_drawdown(result.benchmark_equity),
        "超额年化收益": ann - annualized_return(result.benchmark_ret),
    }

    if len(result.trades):
        tr = result.trades["ret"]
        wins, losses = tr[tr > 0], tr[tr < 0]
        out.update({
            "盈利次数": float(len(wins)),
            "亏损次数": float(len(losses)),
            "胜率(按次)": float(len(wins) / len(tr)),
            "平均盈利率(按次)": float(wins.mean()) if len(wins) else np.nan,
            "平均亏损率(按次)": float(losses.mean()) if len(losses) else np.nan,
            "平均盈亏比(按次)": float(wins.mean() / abs(losses.mean()))
            if len(wins) and len(losses) else np.nan,
            "单次最大盈利": float(tr.max()),
            "单次最大亏损": float(tr.min()),
        })
    return out


def summary_frame(summary: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"指标": k, "数值": v} for k, v in summary.items()])

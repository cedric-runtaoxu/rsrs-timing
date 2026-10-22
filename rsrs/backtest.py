"""回测与绩效统计。

交易假设
--------
* 仓位：多头（1）/ 空仓（0）/ 空头（-1），按日度指数目标暴露计收益。
* 成交：默认 next_close，信号在 t 日收盘产生，t+lag 日收盘成交，
  从下一段收盘到收盘收益开始计损益。legacy_close 仅用于旧版数字诊断。
* 成本：每次仓位变化收取 ``cost`` 的单边费率；原报告按**双边** 0.4% / 0.6% 列示，
  本模块的 ``cost`` 是单边费率，双边 0.4% 对应单边 0.002。
* 基准：同一指数买入持有（buy & hold）。

关于「交易次数」的口径
----------------------
原报告表 2 给出 持仓总天数 ÷ 交易次数 = 平均持仓天数
（例如标准分策略 1353 ÷ 52 = 26.02），说明报告的「交易次数」指**建仓次数**
（期末未平仓也计入，因此不等于已完成回合数），而不是仓位变化次数：

* ``n_trades``   收益持仓段数，含期末未平仓，不等价于完成回合数；
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
    """持仓段数，含期末未平仓。"""
    n_switches: int
    """仓位变化次数（建仓 + 平仓）。"""
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    """每笔交易的明细：start / end / days / ret（不计成本）。"""
    params: dict = field(default_factory=dict)
    execution_position: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    """收盘成交后的仓位；position 是当日收盘收益使用的仓位。"""
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    short_carry: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


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
    execution: str = "next_close",
    short_cost: float = 0.0,
) -> BacktestResult:
    """按 ``position``（-1/0/1，t 日收盘决策）回测。

    Parameters
    ----------
    lag : 信号日到收盘执行日的交易日间隔，默认 1。
    cost : 单边交易费率，仓位变化当天扣除。
    short_cost : 空头占用的假设年化持有费用，按252交易日分摊；主研究为0。
    execution : next_close 为可执行的延迟收盘口径；legacy_close 将信号
        shift(lag) 直接乘当日收益，只用于历史结果对照，不能称为次日收盘成交。
    """
    if lag < 1:
        raise ValueError("lag 必须 >= 1，否则存在未来函数")
    if execution not in ("next_close", "legacy_close"):
        raise ValueError("execution 必须为 next_close 或 legacy_close")
    if not 0 <= cost < 1:
        raise ValueError("cost 必须在 [0,1) 内")
    if not 0 <= short_cost < 1:
        raise ValueError("short_cost 必须在 [0,1) 内")
    if df.empty or not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("输入必须为非空、日期唯一且递增的数据")

    pos = position.reindex(df.index).ffill().fillna(0.0)
    if not np.isfinite(pos).all() or not pos.between(-1,1).all():
        raise ValueError("目标仓位必须为[-1,1]内的有限数值")
    executed = pos.shift(lag).fillna(0.0)
    pos_effective = executed.shift(1).fillna(0.0) if execution == "next_close" else executed

    gross_ret = df["ret"].fillna(0.0)
    turnover = executed.diff().abs()
    turnover.iloc[0] = abs(executed.iloc[0])
    cost_series = turnover * cost

    carry = -pos_effective.clip(upper=0) * short_cost / TRADING_DAYS
    strat_ret = pos_effective * gross_ret - cost_series - carry
    bench_ret = gross_ret

    # trade_table 内部只乘一次方向；传入指数收益，避免空头被重复翻转。
    trades = trade_table(pos_effective, gross_ret)

    return BacktestResult(
        ret=strat_ret,
        benchmark_ret=bench_ret,
        position=pos_effective,
        equity=(1.0 + strat_ret).cumprod(),
        benchmark_equity=(1.0 + bench_ret).cumprod(),
        n_trades=int(len(trades)),
        n_switches=int((turnover > 0).sum()),
        trades=trades,
        params={"lag": lag, "cost": cost, "benchmark": benchmark, "execution": execution, "short_cost":short_cost},
        execution_position=executed,
        turnover=turnover,
        short_carry=carry,
    )


# ------------------------------------------------------------------ 绩效指标


def total_return(equity: pd.Series) -> float:
    """equity 为从初始资金 1 累乘的每日期末净值，首日收益不能被除掉。"""
    return float(equity.iloc[-1] - 1.0)


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
    peak = equity.cummax().clip(lower=1.0)
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

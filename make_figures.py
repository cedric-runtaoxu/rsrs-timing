# -*- coding: utf-8 -*-
"""生成复现报告所需的图表，输出到 ``report/``。

对应原报告的图 9（净值）、图 14/15/20（指标分布）、图 16—图 22（预测性）、
图 28（M 参数敏感性）。

用法::

    python make_figures.py
    python make_figures.py --outdir report

需要 matplotlib；未安装时脚本会提示并跳过，不影响其余复现流程。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from replicate import REPORT_PREDICTIVE, REPORT_TABLE7, build_strategies
from rsrs.backtest import backtest
from rsrs.data import load_index
from rsrs.indicators import rsrs_family, rsrs_signal
from rsrs.plot import setup_cjk_font
from rsrs.predictive import binned_stats, effective_sample_n
from rsrs.strategies import apply_entry_filter, ma_filter, volume_corr_filter

START, END = "2005-04-08", "2017-04-28"
REPORT = Path(__file__).resolve().parent / "report"

C_LONG = "#c0392b"
C_SHORT = "#2471a3"
C_BENCH = "#333333"
GRID = dict(alpha=0.3)


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        setup_cjk_font()
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # pragma: no cover
        print(f"[跳过] 未安装 matplotlib（{exc}）；执行 pip install matplotlib 后可生成图表")
        return None


def _save(fig, plt, outdir: Path, name: str, top: float | None = None) -> Path:
    """保存图；``top`` 给出为总标题预留的空间（tight_layout 的 rect 上边界）。"""
    path = outdir / name
    if top is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  已生成 {path}")
    return path


# ---------------------------------------------------------------- 图 1


def fig_indicators(df, signals, outdir, plt):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, sig in signals.items():
        r = backtest(df, sig, lag=1, cost=0.0)
        ax.plot(r.equity.index, r.equity, lw=1.1, label=name)
    bench = (1.0 + df["ret"].fillna(0.0)).cumprod()
    ax.plot(bench.index, bench, lw=1.7, color=C_BENCH, ls="--", label="沪深300 买入持有")
    ax.set_yscale("log")
    ax.set_ylabel("净值（对数轴）")
    ax.set_title("图 1　四个基础策略的净值表现（沪深300，2005-04 ~ 2017-04，不计成本）")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(**GRID)
    return _save(fig, plt, outdir, "fig01_nav_indicators.png")


# ---------------------------------------------------------------- 图 2


def fig_nav_drawdown(df, signals, outdir, plt):
    r = backtest(df, signals["标准分策略"], lag=1, cost=0.0)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(r.equity.index, r.equity, lw=1.3, color=C_LONG, label="标准分策略")
    axes[0].plot(r.benchmark_equity.index, r.benchmark_equity, lw=1.3, color=C_BENCH,
                 alpha=0.85, label="沪深300 买入持有")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("净值（对数轴）")
    axes[0].set_title("图 2　标准分策略净值与回撤（不计成本）")
    axes[0].legend(loc="upper left")
    axes[0].grid(**GRID)

    dd = r.equity / r.equity.cummax() - 1.0
    bdd = r.benchmark_equity / r.benchmark_equity.cummax() - 1.0
    axes[1].fill_between(dd.index, dd.to_numpy(), 0, color=C_LONG, alpha=0.4, label="策略回撤")
    axes[1].plot(bdd.index, bdd, lw=0.9, color=C_BENCH, alpha=0.8, label="基准回撤")
    axes[1].set_ylabel("回撤")
    axes[1].legend(loc="lower left")
    axes[1].grid(**GRID)
    return _save(fig, plt, outdir, "fig02_nav_drawdown.png")


# ---------------------------------------------------------------- 图 3


def fig_distribution(df, outdir, plt):
    fam18 = rsrs_family(df, n=18, m=600)
    fam16 = rsrs_family(df, n=16, m=300)
    specs = (("标准分（N=18, M=600）", fam18["zscore"]),
             ("修正标准分（N=18, M=600）", fam18["modified"]),
             ("右偏标准分（N=16, M=300）", fam16["right_skewed"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (title, s) in zip(axes, specs):
        v = s.replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(v, bins=80, color=C_SHORT, alpha=0.75)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_title(f"{title}\n均值 {v.mean():.3f}  标准差 {v.std(ddof=1):.3f}  "
                     f"偏度 {v.skew():.3f}  峰度 {v.kurt():.3f}", fontsize=10)
        ax.set_ylabel("频数")
        ax.grid(**GRID)
    fig.suptitle("图 3　三个指标的取值分布（对应原报告图 14 / 图 15 / 图 20）",
                 y=0.98, fontsize=11)
    return _save(fig, plt, outdir, "fig03_distribution.png", top=0.90)


# ---------------------------------------------------------------- 图 4


def fig_predictive(df, outdir, plt, width=0.2, horizon=10):
    fam18 = rsrs_family(df, n=18, m=600)
    fam16 = rsrs_family(df, n=16, m=300)
    specs = (("标准分", fam18["zscore"]), ("修正标准分", fam18["modified"]),
             ("右偏标准分", fam16["right_skewed"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (name, ind) in zip(axes, specs):
        g = binned_stats(ind, df["close"], horizon=horizon, binning="width", width=width)
        ax.plot(g["center"], g["exp_ret"], "o-", ms=3.2, lw=1.0, color=C_SHORT,
                label=f"期望收益（整体 r={g['center'].corr(g['exp_ret']):.2f}）")
        ax.axhline(0, color="#999", lw=0.8)
        ax.axvline(0, color="#bbb", lw=0.8)
        ax.set_xlabel("指标取值")
        ax.set_ylabel("未来 10 日期望收益", color=C_SHORT)
        ax.grid(**GRID)

        ax2 = ax.twinx()
        ax2.plot(g["center"], g["up_rate"], "s--", ms=3.0, lw=1.0, color="#d68910",
                 label=f"上涨概率（整体 r={g['center'].corr(g['up_rate']):.2f}）")
        ax2.set_ylabel("未来 10 日上涨概率", color="#d68910")
        ax2.set_ylim(0, 1)

        ref = REPORT_PREDICTIVE[name]
        ax.set_title(f"{name}　研报整体 r = {ref[4]:.2f}"
                     f"{'' if ref[5] is None else f' / {ref[5]:.2f}'}", fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    fig.suptitle(f"图 4　指标对未来 10 日收益的预测性（等宽分箱，箱宽 {width}，"
                 f"共 {len(df)} 个交易日但独立观测仅约 {effective_sample_n(df['close'], horizon)} 个）",
                 y=0.99, fontsize=11)
    return _save(fig, plt, outdir, "fig04_predictive.png", top=0.88)


# ---------------------------------------------------------------- 图 5


def fig_m_sensitivity(df, outdir, plt):
    ma_ok = ma_filter(df["close"], 20)
    rows = []
    for m in sorted(REPORT_TABLE7):
        fam = rsrs_family(df, n=18, m=m)
        base = rsrs_signal(fam["zscore"], s_buy=0.7, s_sell=-0.7)
        vol_ok = volume_corr_filter(df["volume"], fam["modified"], 10)

        def nav(sig):
            return float((1.0 + backtest(df, sig, lag=1, cost=0.0).ret).prod())

        rows.append({"M": m, "单指标": nav(base),
                     "价格优化": nav(apply_entry_filter(base, ma_ok)),
                     "相关性优化": nav(apply_entry_filter(base, vol_ok))})
    ours = pd.DataFrame(rows).set_index("M")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for ax, (col, idx) in zip(axes, (("单指标", 0), ("价格优化", 1), ("相关性优化", 2))):
        ref = [REPORT_TABLE7[m][idx] for m in ours.index]
        ax.plot(ours.index, ref, "o--", ms=3.5, lw=1.1, color=C_BENCH, label="原报告")
        ax.plot(ours.index, ours[col], "s-", ms=3.5, lw=1.2, color=C_LONG, label="本复现")
        ax.axvline(600, color="#bbb", lw=0.8, ls=":")
        ax.set_xlabel("标准分计算周期 M")
        ax.set_ylabel("组合净值")
        ax.set_title(col, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(**GRID)
    fig.suptitle("图 5　M 参数敏感性：原报告与本复现（不计成本）", y=0.99, fontsize=11)
    return _save(fig, plt, outdir, "fig05_m_sensitivity.png", top=0.88)


# ---------------------------------------------------------------- 入口


def main() -> int:
    ap = argparse.ArgumentParser(description="生成复现报告图表")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()

    plt = _mpl()
    if plt is None:
        return 1

    outdir = Path(args.outdir) if args.outdir else REPORT
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_index("sh000300").loc[args.start:args.end]
    print(f"标的 沪深300，区间 {df.index[0].date()} ~ {df.index[-1].date()}，"
          f"{len(df)} 个交易日")
    signals, _ = build_strategies(df)

    print("生成图表：")
    fig_indicators(df, signals, outdir, plt)
    fig_nav_drawdown(df, signals, outdir, plt)
    fig_distribution(df, outdir, plt)
    fig_predictive(df, outdir, plt)
    fig_m_sensitivity(df, outdir, plt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

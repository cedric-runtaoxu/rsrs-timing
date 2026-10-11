# -*- coding: utf-8 -*-
"""RSRS 择时演示脚本。

与复现口径的区别（重要）
------------------------
本脚本是**通用演示工具**：默认跑全样本（2005—2024）并计入单边 0.1% 的交易成本，
用于快速查看指标行为、交易信号与净值曲线。

严谨的**复现口径**（与原报告可比）是「样本 2005-04-08 ~ 2017-04-28、不计成本」，
由 ``replicate.py`` 生成，结果见 README 第四节。

为避免两者混淆，本脚本的输出文件统一使用 ``demo_`` 前缀；复现脚本的输出使用
``replicate_`` 前缀。

用法::

    python run.py                              # 沪深300，全样本，单边成本 0.001
    python run.py --end 2017-04-28 --cost 0    # 与复现口径一致的单指标视图
    python run.py --symbol sh000001
    python run.py --n 18 --m 600 --s-buy 0.7 --s-sell -0.7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rsrs.backtest import backtest, summarize, summary_frame
from rsrs.data import SYMBOLS, load_index
from rsrs.indicators import run_rsrs
from rsrs.plot import plot_rsrs

REPORT = Path(__file__).resolve().parent / "report"
PREFIX = "demo"


def fmt_table(s: pd.DataFrame) -> str:
    return s.to_string(index=False, float_format=lambda x: f"{x:,.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="RSRS 择时演示（全样本 + 成本口径）")
    ap.add_argument("--symbol", default="sh000300", choices=list(SYMBOLS))
    ap.add_argument("--start", default=None, help="起始日，默认取数据起点")
    ap.add_argument("--end", default=None, help="截止日，默认取数据终点")
    ap.add_argument("--n", type=int, default=18, help="回归窗口")
    ap.add_argument("--m", type=int, default=600, help="标准化窗口")
    ap.add_argument("--s-buy", type=float, default=0.7, help="买入阈值")
    ap.add_argument("--s-sell", type=float, default=-0.7, help="卖出阈值")
    ap.add_argument("--lag", type=int, default=1, help="信号滞后（>=1，防未来函数）")
    ap.add_argument("--cost", type=float, default=0.001, help="单边交易费率")
    args = ap.parse_args()

    REPORT.mkdir(parents=True, exist_ok=True)
    df = load_index(args.symbol, start=args.start, end=args.end)
    name = SYMBOLS[args.symbol]

    print("=" * 74)
    print("演示口径（与 README 第四节的复现口径不同，结果不可直接对照）")
    print(f"  标的      : {name}（{args.symbol}）")
    print(f"  区间      : {df.index[0].date()} ~ {df.index[-1].date()}，{len(df)} 个交易日")
    print(f"  单边成本  : {args.cost:.4f}")
    print(f"  参数      : N={args.n}, M={args.m}, 阈值 ±{args.s_buy}")
    print("  复现口径  : 2005-04-08 ~ 2017-04-28、不计成本，见 replicate.py")
    print("=" * 74)

    ind = run_rsrs(df, n=args.n, m=args.m, s_buy=args.s_buy, s_sell=args.s_sell)
    result = backtest(df, ind["position"], lag=args.lag, cost=args.cost)
    result.params.update({"n": args.n, "m": args.m, "s_buy": args.s_buy, "s_sell": args.s_sell})

    summary = summarize(result)
    print("\n== 回测绩效 ==")
    print(fmt_table(summary_frame(summary)))

    # ---- 落盘：统一使用 demo_ 前缀，避免与复现产物混淆 ----
    tag = args.symbol
    ind.to_csv(REPORT / f"{PREFIX}_indicator_{tag}.csv", encoding="utf-8-sig")
    pd.DataFrame({
        "equity": result.equity,
        "benchmark_equity": result.benchmark_equity,
        "strategy_ret": result.ret,
        "benchmark_ret": result.benchmark_ret,
        "position": result.position,
    }).to_csv(REPORT / f"{PREFIX}_equity_{tag}.csv", encoding="utf-8-sig")
    summary_frame(summary).to_csv(REPORT / f"{PREFIX}_metrics_{tag}.csv",
                                  index=False, encoding="utf-8-sig")

    fig = plot_rsrs(df, ind, result, REPORT, symbol=tag,
                    title=f"RSRS 择时演示（{name}，全样本，单边成本 {args.cost:.4f}）",
                    fname=f"{PREFIX}_{tag}.png")
    if fig:
        print(f"\n图表已生成: {fig}")
    print(f"\n演示产物已写入: {REPORT}（文件名以 {PREFIX}_ 开头）")
    print("复现产物见 README 第四节，由 replicate.py / check_replication.py 生成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

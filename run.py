"""RSRS 择时复现 —— 主入口。

用法::

    python run.py                 # 沪深300，默认参数
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


def fmt_table(s: pd.DataFrame) -> str:
    return s.to_string(index=False, float_format=lambda x: f"{x:,.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="RSRS 择时指标复现")
    ap.add_argument("--symbol", default="sh000300", choices=list(SYMBOLS))
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
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
    print(f"标的: {name}（{args.symbol}），{df.index[0].date()} ~ {df.index[-1].date()}，{len(df)} 个交易日")

    ind = run_rsrs(df, n=args.n, m=args.m, s_buy=args.s_buy, s_sell=args.s_sell)
    result = backtest(df, ind["position"], lag=args.lag, cost=args.cost)
    result.params.update({"n": args.n, "m": args.m, "s_buy": args.s_buy, "s_sell": args.s_sell})

    summary = summarize(result)
    print("\n== 回测绩效 ==")
    print(fmt_table(summary_frame(summary)))

    # ---- 落盘 ----
    tag = args.symbol
    ind.to_csv(REPORT / f"rsrs_indicator_{tag}.csv", encoding="utf-8-sig")
    pd.DataFrame({
        "equity": result.equity,
        "benchmark_equity": result.benchmark_equity,
        "strategy_ret": result.ret,
        "benchmark_ret": result.benchmark_ret,
        "position": result.position,
    }).to_csv(REPORT / f"rsrs_equity_{tag}.csv", encoding="utf-8-sig")
    summary_frame(summary).to_csv(REPORT / f"rsrs_metrics_{tag}.csv", index=False, encoding="utf-8-sig")

    fig = plot_rsrs(df, ind, result, REPORT, symbol=tag,
                    title=f"RSRS 择时（{name}）")
    if fig:
        print(f"\n图表已生成: {fig}")
    print(f"\n结果已写入: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

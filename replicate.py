# -*- coding: utf-8 -*-
"""对照原报告表 2 / 表 4 / 表 3 的复现脚本。

原报告基准（沪深300，2005-03 ~ 2017-04，不计成本，光大证券 2017-05-01）
--------------------------------------------------------------------------
表2  斜率策略    年化 21.71%  夏普 1.10  最大回撤 -51.69%  交易 41 次  持仓 1441 天
表2  标准分策略  年化 24.12%  夏普 1.27  最大回撤 -46.16%  交易 52 次  持仓 1353 天
表4  修正标准分  年化 22.47%  夏普 1.18  最大回撤 -49.11%  交易 54 次  持仓 1386 天
表4  右偏标准分  年化 25.82%  夏普 1.20  最大回撤 -50.49%  交易 39 次  持仓 1783 天
表2  同期沪深300 总收益 350.94%

用法::

    python replicate.py                        # 用报告样本区间
    python replicate.py --end 2024-12-31       # 延长样本
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rsrs.backtest import backtest, summarize
from rsrs.data import SYMBOLS, load_index
from rsrs.indicators import rsrs_family, rsrs_signal
from rsrs.predictive import effective_sample_n, predictive_table, sensitivity_to_bins
from rsrs.strategies import (
    apply_entry_filter,
    long_short_position,
    ma_direction,
    ma_filter,
    volume_corr_filter,
)

REPORT = Path(__file__).resolve().parent / "report"

# 原报告公开数值，用于逐项对照
REPORT_NUMBERS = {
    "斜率策略":     dict(年化=0.2171, 夏普=1.10, 最大回撤=-0.5169, 交易次数=41, 持仓总天数=1441),
    "标准分策略":   dict(年化=0.2412, 夏普=1.27, 最大回撤=-0.4616, 交易次数=52, 持仓总天数=1353),
    "修正标准分策略": dict(年化=0.2247, 夏普=1.18, 最大回撤=-0.4911, 交易次数=54, 持仓总天数=1386),
    "右偏标准分策略": dict(年化=0.2582, 夏普=1.20, 最大回撤=-0.5049, 交易次数=39, 持仓总天数=1783),
}

# 原报告表 5（价格优化，均线过滤）与表 6（成交量相关性优化）
# 第 22 页说明：优化策略统一使用 N=18、M=600、S=0.7
REPORT_NUMBERS_OPT = {
    "价格优化·标准分":   dict(年化=0.2737, 夏普=1.76, 最大回撤=-0.1548, 交易次数=38),
    "价格优化·修正标准分": dict(年化=0.2613, 夏普=1.64, 最大回撤=-0.1576, 交易次数=39),
    "价格优化·右偏标准分": dict(年化=0.2819, 夏普=1.56, 最大回撤=-0.2231, 交易次数=32),
    "量相关优化·标准分":  dict(年化=0.2620, 夏普=1.66, 最大回撤=-0.1377, 交易次数=41),
    "量相关优化·修正标准分": dict(年化=0.2622, 夏普=1.65, 最大回撤=-0.1576, 交易次数=35),
    "量相关优化·右偏标准分": dict(年化=0.2652, 夏普=1.41, 最大回撤=-0.3307, 交易次数=28),
}


def build_optimized_signals(df: pd.DataFrame, fam: pd.DataFrame) -> dict:
    """原报告第 3 章的 6 个优化策略：3 个基础指标 × 2 种开仓过滤。"""
    ma_ok = ma_filter(df["close"], window=20)
    vol_ok = volume_corr_filter(df["volume"], fam["modified"], window=10)

    out = {}
    for label, col in (("标准分", "zscore"),
                       ("修正标准分", "modified"),
                       ("右偏标准分", "right_skewed")):
        base = rsrs_signal(fam[col], s_buy=0.7, s_sell=-0.7)
        out[f"价格优化·{label}"] = apply_entry_filter(base, ma_ok)
        out[f"量相关优化·{label}"] = apply_entry_filter(base, vol_ok)
    return out


def build_strategies(df: pd.DataFrame) -> dict:
    """构造原报告的四个基础指标信号。"""
    fam18600 = rsrs_family(df, n=18, m=600)   # 标准分族
    fam16300 = rsrs_family(df, n=16, m=300)   # 右偏标准分用 N=16, M=300

    return {
        # 斜率策略：beta > 1.0 买入，beta < 0.8 卖出（报告第 2.1 节，阈值取自均值0.903±标准差0.116）
        "斜率策略": rsrs_signal(fam18600["beta"], s_buy=1.0, s_sell=0.8),
        # 标准分策略：z > 0.7 买，z < -0.7 卖
        "标准分策略": rsrs_signal(fam18600["zscore"], s_buy=0.7, s_sell=-0.7),
        # 修正标准分策略：z × R²
        "修正标准分策略": rsrs_signal(fam18600["modified"], s_buy=0.7, s_sell=-0.7),
        # 右偏标准分策略：修正标准分 × beta（N=16, M=300）
        "右偏标准分策略": rsrs_signal(fam16300["right_skewed"], s_buy=0.7, s_sell=-0.7),
    }, fam18600


# 原报告表 8：价格优化右偏标准分在多空与不同市场的表现
REPORT_TABLE8 = {
    "沪深300多头": dict(年化=0.2819, 夏普=1.56, 最大回撤=-0.2231, 交易次数=32, 持仓总天数=1327),
    "沪深300多空": dict(年化=0.3798, 夏普=1.46, 最大回撤=-0.3232, 交易次数=65, 持仓总天数=2212),
    "上证50": dict(年化=0.2469, 夏普=1.41, 最大回撤=-0.1966, 交易次数=32, 持仓总天数=1297),
    "中证500": dict(年化=0.3266, 夏普=1.60, 最大回撤=-0.2621, 交易次数=39, 持仓总天数=1139),
}

# 原报告表 7：标准分计算周期 M 的参数敏感性，值为（单指标净值, 价格优化净值, 相关性优化净值）
REPORT_TABLE7 = {
    450: (11.504, 15.198, 11.407),
    500: (12.484, 16.811, 14.238),
    550: (12.047, 16.216, 13.851),
    600: (13.374, 18.232, 16.343),
    650: (13.136, 18.019, 13.047),
    700: (13.084, 18.069, 13.609),
    750: (12.872, 18.031, 13.751),
    800: (11.841, 16.549, 13.234),
}

# 原报告第 2.3 节披露的预测性相关系数
# 顺序：(左侧期望收益, 左侧上涨概率, 右侧期望收益, 右侧上涨概率, 整体期望收益, 整体上涨概率)
REPORT_PREDICTIVE = {
    "标准分": (-0.21, -0.01, 0.45, 0.27, 0.31, None),
    "修正标准分": (0.50, 0.57, 0.53, 0.52, 0.68, 0.65),
    "右偏标准分": (0.84, 0.80, 0.56, 0.42, 0.75, 0.66),
}

MARKETS = (
    ("sh000300", "沪深300多头", "long"),
    ("sh000300", "沪深300多空", "long_short"),
    ("sh000016", "上证50", "long"),
    ("sh000905", "中证500", "long"),
)


def pct(x: float) -> str:
    return "  n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:6.2f}%"


def nav_of(df: pd.DataFrame, sig: pd.Series, cost: float = 0.0) -> float:
    return float((1.0 + backtest(df, sig, lag=1, cost=cost).ret).prod())


def multi_market_table(start: str, end: str) -> pd.DataFrame:
    """原报告表 8：价格优化右偏标准分在多空与不同市场（N=18, M=600, S=0.7）。"""
    rows = []
    for symbol, label, mode in MARKETS:
        d = load_index(symbol).loc[start:end]
        fam = rsrs_family(d, n=18, m=600)
        ma_up, ma_down = ma_direction(d["close"], 20)
        if mode == "long":
            base = rsrs_signal(fam["right_skewed"], s_buy=0.7, s_sell=-0.7)
            sig = apply_entry_filter(base, ma_up)
        else:
            sig = long_short_position(fam["right_skewed"], ma_up, ma_down, 0.7, -0.7)
        s = summarize(backtest(d, sig, lag=1, cost=0.0))
        ref = REPORT_TABLE8[label]
        rows.append({
            "组合": label,
            "年化_研报": ref["年化"], "年化_复现": s["年化收益"],
            "夏普_研报": ref["夏普"], "夏普_复现": s["夏普比率"],
            "回撤_研报": ref["最大回撤"], "回撤_复现": s["最大回撤"],
            "交易_研报": ref["交易次数"], "交易_复现": s["交易次数"],
            "持仓_研报": ref["持仓总天数"], "持仓_复现": s["持仓总天数"],
        })
    return pd.DataFrame(rows).set_index("组合")


def m_sensitivity_table(start: str, end: str) -> pd.DataFrame:
    """原报告表 7：M 的参数敏感性（标准分策略 / 价格优化 / 相关性优化）。"""
    d = load_index("sh000300").loc[start:end]
    ma_ok = ma_filter(d["close"], 20)
    rows = []
    for m in sorted(REPORT_TABLE7):
        fam = rsrs_family(d, n=18, m=m)
        base = rsrs_signal(fam["zscore"], s_buy=0.7, s_sell=-0.7)
        vol_ok = volume_corr_filter(d["volume"], fam["modified"], 10)
        ours = (
            nav_of(d, base),
            nav_of(d, apply_entry_filter(base, ma_ok)),
            nav_of(d, apply_entry_filter(base, vol_ok)),
        )
        ref = REPORT_TABLE7[m]
        rows.append({
            "M": m,
            "单指标_研报": ref[0], "单指标_复现": ours[0],
            "价格优化_研报": ref[1], "价格优化_复现": ours[1],
            "相关性优化_研报": ref[2], "相关性优化_复现": ours[2],
        })
    return pd.DataFrame(rows).set_index("M")


def compare_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        s = summarize(res)
        ref = REPORT_NUMBERS[name]
        rows.append({
            "策略": name,
            "年化_研报": ref["年化"], "年化_复现": s["年化收益"], "年化_差": s["年化收益"] - ref["年化"],
            "夏普_研报": ref["夏普"], "夏普_复现": s["夏普比率"], "夏普_差": s["夏普比率"] - ref["夏普"],
            "回撤_研报": ref["最大回撤"], "回撤_复现": s["最大回撤"], "回撤_差": s["最大回撤"] - ref["最大回撤"],
            "交易_研报": ref["交易次数"], "交易_复现": s["交易次数"],
            "持仓_研报": ref["持仓总天数"], "持仓_复现": s["持仓总天数"],
            "净值_复现": s["累计收益"] + 1.0,
            "胜率按天": s["胜率(按天)"], "胜率按次": s.get("胜率(按次)", np.nan),
        })
    return pd.DataFrame(rows).set_index("策略")


def main() -> int:
    ap = argparse.ArgumentParser(description="对照原报告复现 RSRS 四个基础指标")
    ap.add_argument("--symbol", default="sh000300", choices=list(SYMBOLS))
    ap.add_argument("--start", default="2005-04-08")
    ap.add_argument("--end", default="2017-04-28")
    ap.add_argument("--outdir", default=None, help="结果输出目录，默认 report/")
    ap.add_argument("--dry-run", action="store_true", help="只打印对照表，不写文件")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else REPORT
    outdir.mkdir(parents=True, exist_ok=True)

    def save(frame: pd.DataFrame, fname: str) -> None:
        if args.dry_run:
            return
        frame.to_csv(outdir / fname, encoding="utf-8-sig")

    name = SYMBOLS[args.symbol]
    full = load_index(args.symbol)
    df = full.loc[args.start:args.end]
    print(f"标的: {name}（{args.symbol}）  复现区间: {df.index[0].date()} ~ {df.index[-1].date()}，{len(df)} 个交易日")
    bench_total = full.loc[args.start:args.end, "close"].iloc[-1] / full.loc[args.start:args.end, "close"].iloc[0] - 1
    print(f"基准（买入持有）总收益: {pct(bench_total)}   （研报同期沪深300: 350.94%）")
    print("  注：本项为已知差异。我们使用价格指数（腾讯行情），研报使用 WIND 数据；")
    print("      350.94% / 242.79% 的比值约 1.32，与沪深300 全收益指数相对价格指数")
    print("      12 年的累计股息（约 2.3%/年）量级吻合，推测研报基准可能采用全收益口径，")
    print("      待用 WIND 数据核实。这不影响四个策略之间的横向比较。\n")

    signals, fam = build_strategies(df)

    print("=== 指标统计量核对 ===")
    b = fam["beta"].dropna()
    print(f"  斜率 beta (N=18)   均值 {b.mean():.3f}  标准差 {b.std(ddof=1):.3f}  "
          f"偏度 {b.skew():.3f}  峰度 {b.kurt():.3f}")
    print(f"  研报表1          均值 0.903  标准差 0.116  偏度 -0.488  峰度 0.998")
    z = fam["zscore"].dropna()
    print(f"  标准分 z (M=600)   均值 {z.mean():.3f}  标准差 {z.std(ddof=1):.3f}  "
          f"偏度 {z.skew():.3f}  峰度 {z.kurt():.3f}")
    print(f"  研报表3          均值 -0.040 标准差 1.063 偏度 -0.508 峰度 1.105\n")

    print("=== 策略绩效对照（不计成本）===")
    results = {n: backtest(df, sig, lag=1, cost=0.0) for n, sig in signals.items()}
    tbl = compare_table(results)
    show = tbl.copy()
    for c in ("年化_研报", "年化_复现", "年化_差", "回撤_研报", "回撤_复现", "回撤_差", "胜率按天", "胜率按次"):
        show[c] = show[c].map(pct)
    for c in ("夏普_研报", "夏普_复现", "夏普_差", "净值_复现"):
        show[c] = show[c].map(lambda v: "  n/a" if not np.isfinite(v) else f"{v:6.2f}")
    print(show.to_string())
    save(tbl, "replicate_vs_report.csv")

    print("\n=== 成本敏感性（研报：标准分 无成本13.37 / 双边0.4%→10.88 / 双边0.6%→9.82）===")
    for name_, sig in signals.items():
        navs = []
        for cost in (0.0, 0.002, 0.003):
            r = backtest(df, sig, lag=1, cost=cost)
            navs.append((1 + r.ret).prod())
        print(f"  {name_:<12s} 净值: 无成本 {navs[0]:6.2f}  双边0.4% {navs[1]:6.2f}  双边0.6% {navs[2]:6.2f}")

    print("\n=== 优化策略对照：价格优化（MA20 过滤）与成交量相关性优化 ===")
    print("    原报告第 22 页：优化策略统一使用 N=18、M=600、S=0.7")
    opt_signals = build_optimized_signals(df, fam)
    rows = []
    for nm, sig in opt_signals.items():
        s = summarize(backtest(df, sig, lag=1, cost=0.0))
        ref = REPORT_NUMBERS_OPT[nm]
        rows.append({
            "策略": nm,
            "年化_研报": ref["年化"], "年化_复现": s["年化收益"],
            "夏普_研报": ref["夏普"], "夏普_复现": s["夏普比率"],
            "回撤_研报": ref["最大回撤"], "回撤_复现": s["最大回撤"],
            "交易_研报": ref["交易次数"], "交易_复现": s["交易次数"],
        })
    opt_tbl = pd.DataFrame(rows).set_index("策略")
    show2 = opt_tbl.copy()
    for c in ("年化_研报", "年化_复现", "回撤_研报", "回撤_复现"):
        show2[c] = show2[c].map(pct)
    for c in ("夏普_研报", "夏普_复现"):
        show2[c] = show2[c].map(lambda v: f"{v:6.2f}")
    print(show2.to_string())
    save(opt_tbl, "replicate_optimized.csv")

    print("\n=== 表 8 对照：价格优化右偏标准分（多空与多市场，N=18 M=600 S=0.7）===")
    mm = multi_market_table(args.start, args.end)
    show3 = mm.copy()
    for c in ("年化_研报", "年化_复现", "回撤_研报", "回撤_复现"):
        show3[c] = show3[c].map(pct)
    for c in ("夏普_研报", "夏普_复现"):
        show3[c] = show3[c].map(lambda v: f"{v:6.2f}")
    print(show3.to_string())
    save(mm, "replicate_multi_market.csv")

    print("\n=== 表 7 对照：标准分计算周期 M 的参数敏感性（组合净值）===")
    ms = m_sensitivity_table(args.start, args.end)
    show4 = ms.copy()
    for c in show4.columns:
        show4[c] = show4[c].map(lambda v: f"{v:7.3f}")
    print(show4.to_string())
    save(ms, "replicate_m_sensitivity.csv")

    print("\n=== 预测性检验对照（原报告第 2.3 节：指标 vs 未来 10 日收益）===")
    print("    原报告未披露分箱方案；本表按原图横轴步长取等宽分箱，箱宽 0.2")
    pt = predictive_table(df, horizon=10, binning="width", width=0.2)
    pred_rows = []
    for name, ref in REPORT_PREDICTIVE.items():
        ours = pt.loc[name]
        pred_rows.append({
            "指标": name,
            "左侧_期望收益_研报": ref[0], "左侧_期望收益_复现": ours["左侧_期望收益"],
            "右侧_期望收益_研报": ref[2], "右侧_期望收益_复现": ours["右侧_期望收益"],
            "整体_期望收益_研报": ref[4], "整体_期望收益_复现": ours["整体_期望收益"],
            "整体_上涨概率_研报": ref[5], "整体_上涨概率_复现": ours["整体_上涨概率"],
        })
    ptbl = pd.DataFrame(pred_rows).set_index("指标")
    print(ptbl.round(3).to_string())
    save(ptbl, "replicate_predictive.csv")

    print("\n=== 预测性相关系数对分箱宽度的敏感性（整体相关系数）===")
    sens = sensitivity_to_bins(df, horizon=10).round(3)
    print(sens.to_string())
    save(sens.reset_index(), "replicate_predictive_sensitivity.csv")
    print(f"\n    10 日重叠窗口下，名义样本 {len(df)} 个交易日，"
          f"相互独立的观测仅约 {effective_sample_n(df['close'], 10)} 个。")

    # 逐年收益，便于观察信号分布
    print("\n=== 分年度（标准分策略 vs 沪深300）===")
    r = results["标准分策略"]
    yr = pd.DataFrame({"策略": r.ret, "基准": r.benchmark_ret})
    yr = ((1 + yr).groupby(yr.index.year).prod() - 1)
    print((yr * 100).round(2).to_string())
    save(yr, "replicate_yearly.csv")

    print(f"\n结果已写入: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

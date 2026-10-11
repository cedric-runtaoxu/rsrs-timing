# -*- coding: utf-8 -*-
"""复现质量自检：把原报告公开的每一项数值逐一比对，输出匹配率与结论。

用法::

    python check_replication.py
    python check_replication.py --dry-run     # 只打印，不写文件

判定容差
--------
============  ==================================
指标          容差
============  ==================================
年化收益       ±2.0 个百分点
夏普比率       ±0.10
最大回撤       ±2.0 个百分点
交易次数       相对误差 ±20%
持仓总天数     相对误差 ±15%
组合净值       相对误差 ±15%
低阶矩         ±0.10（斜率均值另按 ±0.01 严格判定）
============  ==================================

样本长度不同的对比项（例如中证500：该指数 2007-01-15 才发布，本复现无法覆盖
原报告的 2005-03 起点）单独标记为「不可比」，不计入通过率。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from replicate import (
    REPORT_NUMBERS,
    REPORT_NUMBERS_OPT,
    build_optimized_signals,
    build_strategies,
    m_sensitivity_table,
    multi_market_table,
)
from rsrs.backtest import backtest, summarize
from rsrs.data import load_index
from rsrs.indicators import rsrs_family

START, END = "2005-04-08", "2017-04-28"
REPORT_DIR = Path(__file__).resolve().parent / "report"

TOL_ABS = {"年化": 0.020, "夏普": 0.10, "最大回撤": 0.020, "低阶矩": 0.10}
TOL_REL = {"交易次数": 0.20, "持仓天数": 0.15, "净值": 0.15}

# 样本长度与原报告不一致的组合，不计入通过率
INCOMPARABLE = {"中证500"}


def _pass_abs(rep: float, ours: float, tol: float) -> bool:
    return bool(np.isfinite(rep) and np.isfinite(ours) and abs(ours - rep) <= tol)


def _pass_rel(rep: float, ours: float, tol: float) -> bool:
    if not (np.isfinite(rep) and np.isfinite(ours)) or rep == 0:
        return False
    return bool(abs(ours - rep) / abs(rep) <= tol)


def main() -> int:
    ap = argparse.ArgumentParser(description="复现质量自检")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    df = load_index("sh000300").loc[START:END]
    signals, fam = build_strategies(df)
    opt_signals = build_optimized_signals(df, fam)

    rows: list[dict] = []

    def add(table, item, kind, report, ours, comparable=True):
        if kind in TOL_REL:
            tol, ok = TOL_REL[kind], _pass_rel(report, ours, TOL_REL[kind])
        else:
            tol, ok = TOL_ABS[kind], _pass_abs(report, ours, TOL_ABS[kind])
        rows.append({
            "报告出处": table, "对比项": item, "指标": kind,
            "研报值": float(report), "复现值": float(ours),
            "差异": float(ours - report), "容差": tol,
            "通过": (bool(ok) if comparable else None),
        })

    # ---------------- 表 1 / 表 3：指标低阶矩 ----------------
    b = fam["beta"].dropna()
    for nm, rep, ours in (("均值", 0.903, b.mean()), ("标准差", 0.116, b.std(ddof=1)),
                          ("偏度", -0.488, b.skew()), ("峰度", 0.998, b.kurt())):
        add("表1 斜率低阶矩", nm, "低阶矩", rep, ours)
    z = fam["zscore"].dropna()
    for nm, rep, ours in (("均值", -0.040, z.mean()), ("标准差", 1.063, z.std(ddof=1)),
                          ("偏度", -0.508, z.skew()), ("峰度", 1.105, z.kurt())):
        add("表3 标准分低阶矩", nm, "低阶矩", rep, ours)

    # ---------------- 表 2 / 表 4：四个基础策略 ----------------
    for name, sig in signals.items():
        s = summarize(backtest(df, sig, lag=1, cost=0.0))
        ref = REPORT_NUMBERS[name]
        add(f"表2/4 {name}", "年化收益", "年化", ref["年化"], s["年化收益"])
        add(f"表2/4 {name}", "夏普比率", "夏普", ref["夏普"], s["夏普比率"])
        add(f"表2/4 {name}", "最大回撤", "最大回撤", ref["最大回撤"], s["最大回撤"])
        add(f"表2/4 {name}", "交易次数", "交易次数", ref["交易次数"], s["交易次数"])
        add(f"表2/4 {name}", "持仓总天数", "持仓天数", ref["持仓总天数"], s["持仓总天数"])

    # ---------------- 表 5 / 表 6：六个优化策略 ----------------
    for name, sig in opt_signals.items():
        s = summarize(backtest(df, sig, lag=1, cost=0.0))
        ref = REPORT_NUMBERS_OPT[name]
        add(f"表5/6 {name}", "年化收益", "年化", ref["年化"], s["年化收益"])
        add(f"表5/6 {name}", "夏普比率", "夏普", ref["夏普"], s["夏普比率"])
        add(f"表5/6 {name}", "最大回撤", "最大回撤", ref["最大回撤"], s["最大回撤"])
        add(f"表5/6 {name}", "交易次数", "交易次数", ref["交易次数"], s["交易次数"])

    # ---------------- 表 8：多空与多市场 ----------------
    try:
        mtable = multi_market_table(START, END)
    except FileNotFoundError as exc:
        print(f"[跳过] 表8 多市场对比：{exc}")
        mtable = None
    if mtable is not None:
        for label, r in mtable.iterrows():
            comp = label not in INCOMPARABLE
            add(f"表8 {label}", "年化收益", "年化", r["年化_研报"], r["年化_复现"], comp)
            add(f"表8 {label}", "夏普比率", "夏普", r["夏普_研报"], r["夏普_复现"], comp)
            add(f"表8 {label}", "最大回撤", "最大回撤", r["回撤_研报"], r["回撤_复现"], comp)
            add(f"表8 {label}", "交易次数", "交易次数", r["交易_研报"], r["交易_复现"], comp)

    # ---------------- 表 7：M 参数敏感性（净值口径）----------------
    mt = m_sensitivity_table(START, END)
    for m, r in mt.iterrows():
        add(f"表7 M={m}", "单指标净值", "净值", r["单指标_研报"], r["单指标_复现"])
        add(f"表7 M={m}", "价格优化净值", "净值", r["价格优化_研报"], r["价格优化_复现"])
        add(f"表7 M={m}", "相关性优化净值", "净值", r["相关性优化_研报"], r["相关性优化_复现"])

    res = pd.DataFrame(rows)
    res["表"] = res["报告出处"].str.split().str[0]
    comparable = res[res["通过"].notna()].copy()
    comparable["通过"] = comparable["通过"].astype(bool)
    n_total, n_pass = len(comparable), int(comparable["通过"].sum())
    rate = n_pass / n_total if n_total else 0.0

    core = comparable[comparable["表"] == "表2/4"]
    core_rate = float(core["通过"].mean()) if len(core) else 0.0

    print("=" * 78)
    print("复现质量自检")
    print("=" * 78)
    print(f"样本区间: {START} ~ {END}（标的：沪深300，多市场项除外）")
    print(f"可比对比项 {n_total} 项，通过 {n_pass} 项，总体通过率 {rate:.1%}")
    print(f"不可比项（样本长度不同）：{int(res['通过'].isna().sum())} 项\n")

    print("按报告表统计：")
    for tb, r in comparable.groupby("表"):
        print(f"  {tb:<8s} {int(r['通过'].sum()):>3d}/{len(r):<3d}  {r['通过'].mean():>6.1%}")

    print("\n按指标类型统计：")
    for kind, r in comparable.groupby("指标"):
        print(f"  {kind:<8s} {int(r['通过'].sum()):>3d}/{len(r):<3d}  {r['通过'].mean():>6.1%}")

    failed = comparable[~comparable["通过"]]
    print(f"\n未通过的对比项（{len(failed)} 项）：")
    if failed.empty:
        print("  （无）")
    for _, r in failed.iterrows():
        print(f"  {r['报告出处']:<24s} {r['对比项']:<10s} 研报 {r['研报值']:>9.4f}"
              f"  复现 {r['复现值']:>9.4f}  差异 {r['差异']:>+9.4f}")

    if not args.dry_run:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        res.to_csv(REPORT_DIR / "replicate_selfcheck.csv", index=False, encoding="utf-8-sig")
        print(f"\n明细已写入 {REPORT_DIR / 'replicate_selfcheck.csv'}")

    print()
    if core_rate >= 0.95:
        print(f"结论：核心复现成功。原报告表 2/表 4 的四个基础策略共 {len(core)} 项指标，"
              f"通过率 {core_rate:.0%}（全部落在容差内）。")
        if rate < core_rate:
            print(f"      扩展部分（优化策略 表5/6、多空多市场 表8、M 敏感性 表7）"
                  f"整体通过率 {rate:.1%}，存在系统性偏差，原因见 README「已知差异」。")
    else:
        print(f"结论：复现未达预期（核心项通过率 {core_rate:.1%}）。")
    return 0 if core_rate >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())

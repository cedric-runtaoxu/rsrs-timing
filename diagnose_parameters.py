# -*- coding: utf-8 -*-
"""残差诊断：检验复现中几处系统性偏差的成因假设。

``check_replication.py`` 的结论是：原报告表 2 / 表 4 的四个基础策略全部落在容差
内（20/20），但表 5 / 表 6 的优化策略与表 7 的相关性优化净值存在系统性偏差。
本脚本逐一检验成因假设，结果对照写入 README 第六节「残差诊断」。

用法::

    python diagnose_parameters.py

假设 1｜价格优化·右偏标准分的参数
    原报告第 22 页称优化策略统一使用 N=18、M=600，但该设定下本复现为年化
    24.00%，与原报告表 5 / 表 8 的 28.19% 相差 4.19 个百分点。改用 N=16
    （与右偏标准分基础版一致的回归窗口）后结果更接近，说明报告对优化版右偏
    标准分的参数披露与其基础版存在不一致。

假设 2｜成交量相关性过滤的口径
    原报告中的「（不同分布的）RSRS 标准分指标」是标准分、修正标准分、右偏标准分
    三个指标的**族统称**（表 5、表 6 与图 23、图 25、图 27 的标题均如此表述）。
    表 9 每行以「在 X 策略的基础上」限定前提，故其后半句的「标准分」应理解为该
    策略自身的指标；而第 3.2 节正文以「修正标准分」表述过滤条件。两者不矛盾。

    由此只留下一处实现层面的读法问题：相关性取该策略自身的指标，还是按第 3.2 节
    正文统一取修正标准分？本脚本把两种读法连同「前 10 日」是否含当日共四种组合
    逐一复现。原文的窗口两处均明确写 10 日，故 20 / 30 日窗口只作**敏感性对照**，
    不作为对原文口径的推测。

假设 3｜表 7 单指标净值的水平差
    若复现值与报告值的比值在各 M 上保持恒定，说明两者形状一致、仅存在水平
    差异，可归因于数据源（腾讯 vs WIND）而非方法错误。
"""

from __future__ import annotations

from rsrs.backtest import backtest, summarize
from rsrs.data import load_index
from rsrs.indicators import rsrs_family, rsrs_signal
from rsrs.strategies import apply_entry_filter, ma_filter, volume_corr_filter

START, END = "2005-04-08", "2017-04-28"
REPORT_TABLE7_SINGLE = {   # 原报告表 7「单指标净值」列
    450: 11.504, 500: 12.484, 550: 12.047, 600: 13.374,
    650: 13.136, 700: 13.084, 750: 12.872, 800: 11.841,
}


def main() -> int:
    df = load_index("sh000300").loc[START:END]
    ma20 = ma_filter(df["close"], window=20)

    def perf(sig):
        s = summarize(backtest(df, sig, lag=1, cost=0.0))
        return s["年化收益"] * 100, s["夏普比率"], s["最大回撤"] * 100, s["交易次数"]

    print("假设 1｜价格优化·右偏标准分")
    print("  原报告表5/表8：年化 28.19%  夏普 1.56  最大回撤 -22.31%  交易 32 次")
    for n, m in ((18, 600), (16, 300), (16, 600), (18, 300)):
        fam = rsrs_family(df, n=n, m=m)
        sig = apply_entry_filter(rsrs_signal(fam["right_skewed"], 0.7, -0.7), ma20)
        a, sh, dd, tr = perf(sig)
        print(f"  N={n:>2d} M={m:>3d} → 年化 {a:6.2f}%  夏普 {sh:5.2f}  "
              f"最大回撤 {dd:7.2f}%  交易 {tr:4.0f}")

    print("\n假设 2｜量相关优化·标准分（原文窗口明确为 10 日）")
    print("  原报告表6：年化 26.20%  夏普 1.66  最大回撤 -13.77%  交易 41 次")
    print("  检验两种读法：相关性取该策略自身指标，或按 3.2 节正文统一取修正标准分")
    fam = rsrs_family(df, n=18, m=600)
    base = rsrs_signal(fam["zscore"], s_buy=0.7, s_sell=-0.7)

    def vcorr(ind, window=10, exclude_today=False):
        v = df["volume"].astype("float64")
        i = ind.astype("float64")
        if exclude_today:
            v, i = v.shift(1), i.shift(1)
        return v.rolling(window, min_periods=window).corr(i) > 0

    for col, label in (("modified", "修正标准分"), ("zscore", "标准分")):
        for excl in (False, True):
            vf = vcorr(fam[col], 10, excl)
            a, sh, dd, tr = perf(apply_entry_filter(base, vf))
            tag = "不含当日" if excl else "含当日  "
            print(f"  相关性基于{label}，10 日 · {tag} → 年化 {a:6.2f}%  "
                  f"夏普 {sh:5.2f}  最大回撤 {dd:7.2f}%  交易 {tr:4.0f}")

    print("  —— 以下为超出原文描述的对照（原文明确写 10 日），仅观察方向 ——")
    for w in (20, 30):
        vf = vcorr(fam["modified"], w, False)
        a, sh, dd, tr = perf(apply_entry_filter(base, vf))
        print(f"  相关性基于修正标准分，窗口 {w:>2d} 日 → 年化 {a:6.2f}%  "
              f"夏普 {sh:5.2f}  最大回撤 {dd:7.2f}%  交易 {tr:4.0f}")
    print("  结论：四种符合原文的组合年化差异在 0.4 个百分点以内、回撤同向偏深，")
    print("        指标选择与窗口端点均非差异来源；差异尚未定位（数据源或未披露细节）。")

    print("\n假设 3｜表 7 单指标净值的水平差")
    ratios = []
    for m, ref in REPORT_TABLE7_SINGLE.items():
        fam = rsrs_family(df, n=18, m=m)
        sig = rsrs_signal(fam["zscore"], s_buy=0.7, s_sell=-0.7)
        nav = float((1.0 + backtest(df, sig, lag=1, cost=0.0).ret).prod())
        ratios.append(nav / ref)
        print(f"  M={m:>3d}  研报 {ref:7.3f}  复现 {nav:7.3f}  比值 {nav / ref:5.3f}")
    lo, hi = min(ratios), max(ratios)
    print(f"  比值区间 [{lo:.3f}, {hi:.3f}]，极差 {hi - lo:.3f} → "
          f"形状{'一致' if hi - lo < 0.05 else '存在差异'}，"
          f"仅水平差约 {(sum(ratios) / len(ratios) - 1) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

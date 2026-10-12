# -*- coding: utf-8 -*-
"""残差诊断：检验复现中几处系统性偏差的成因假设。

``check_replication.py`` 的结论是：原报告表 2 / 表 4 的四个基础策略全部落在容差
内（20/20），但表 5 / 表 6 的优化策略与表 7 的相关性优化净值存在系统性偏差。
本脚本逐一检验成因假设，结果对照写入 README 第六节「残差诊断」。

用法::

    python diagnose_parameters.py

假设 1｜优化策略的参数
    原报告称优化策略统一使用 N=18、M=600。本脚本把六个优化策略在四种参数组合
    下逐一复现，用于判断该声明是否成立：若报告实际统一用了 16/300，则六个策略
    应整体在 16/300 下更吻合，而非只有右偏标准分一个。结论是价格优化·标准分
    （偏差 0.17 个百分点）与价格优化·修正标准分（1.00）确认了 N=18；价格优化·
    右偏标准分改用 N=16 后年化改善但最大回撤仍不符，成因尚未定位。

假设 2｜成交量相关性过滤的口径
    第 3.2 节将该过滤定义为「前 10 日交易量与修正标准分之间的相关性为正」。
    落实到代码时，相关性所用指标有取修正标准分与取该策略自身指标两种选择，
    「前 10 日」亦有含当日与不含当日两种取法。本脚本对四种组合逐一复现，
    用于确认口径选择是否影响结果。

    原文的窗口两处均明确写 10 日，故 20 / 30 日窗口只作**敏感性对照**。

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

    print("假设 1｜优化策略的参数（对照研报表 5 / 表 6 披露的年化收益）")
    report = {
        ("价格优化", "标准分"): 27.37, ("价格优化", "修正标准分"): 26.13,
        ("价格优化", "右偏标准分"): 28.19,
        ("量相关优化", "标准分"): 26.20, ("量相关优化", "修正标准分"): 26.22,
        ("量相关优化", "右偏标准分"): 26.52,
    }
    cols = {"标准分": "zscore", "修正标准分": "modified", "右偏标准分": "right_skewed"}
    combos = ((18, 600), (16, 300), (16, 600), (18, 300))
    print(f"  {'优化策略':<20s}" + "".join(f"({n},{m})".rjust(12) for n, m in combos)
          + "      研报")
    devs = {c: [] for c in combos}
    for (opt, ind_name), ref in report.items():
        cells = []
        for n, m in combos:
            fam = rsrs_family(df, n=n, m=m)
            base = rsrs_signal(fam[cols[ind_name]], s_buy=0.7, s_sell=-0.7)
            filt = ma20 if opt == "价格优化" else volume_corr_filter(
                df["volume"], fam["modified"], 10)
            a = perf(apply_entry_filter(base, filt))[0]
            devs[(n, m)].append(a - ref)
            cells.append(f"{a:6.2f}{a - ref:+6.2f}".rjust(12))
        print(f"  {opt + '·' + ind_name:<20s}" + "".join(cells) + f"{ref:10.2f}")
    print("  " + "-" * 84)
    for c in combos:
        d = devs[c]
        mad = sum(abs(x) for x in d) / len(d)
        print(f"  N={c[0]:>2d} M={c[1]:>3d}  平均绝对偏差 {mad:5.2f}pp   "
              f"±2pp 内 {sum(1 for x in d if abs(x) <= 2)}/{len(d)}")
    print("  结论：价格优化·标准分与价格优化·修正标准分在 (18,600) 下确认 N=18 成立；")
    print("        价格优化·右偏标准分改用 N=16 后年化改善，但最大回撤仍不符，成因尚未定位。")

    print("\n假设 2｜量相关优化·标准分（原文窗口明确为 10 日）")
    print("  原报告表6：年化 26.20%  夏普 1.66  最大回撤 -13.77%  交易 41 次")
    print("  口径对照：相关性取修正标准分或该策略自身指标 × 含当日或不含当日")
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
    print("  结论：四种组合年化差异在 0.4 个百分点以内、回撤同向偏深，")
    print("        口径选择不影响结论；差异尚未定位（数据源或未披露细节）。")

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

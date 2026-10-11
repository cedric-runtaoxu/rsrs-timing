"""预测性检验：复现原报告第 2.3 节关于指标预测能力的一组论断。

原报告的做法
------------
以 RSRS 标准分、修正标准分、右偏标准分分别作为解释变量，以沪深300 指数
**其后 10 个交易日**（约两周）的收益为被解释变量：

1. 按指标取值分箱（原报告图 16—图 22 的横轴在 -4.9 ~ 3.8 之间，步长约 0.3，
   对应等宽分箱）；
2. 统计每个箱内的指标均值（箱中心）、未来 10 日期望收益、未来 10 日上涨概率；
3. 报告「箱中心」与上述两个统计量的相关系数。

原报告给出的相关系数（格式：与期望收益 / 与上涨概率）
-------------------------------------------------------
============  ===============  ===============  ===============
指标          左侧（< 0）      右侧（> 0）      整体
============  ===============  ===============  ===============
标准分        -0.21 / -0.01    0.45 / 0.27      0.31 / —
修正标准分     0.50 /  0.57     0.53 / 0.52      0.68 / 0.65
右偏标准分     0.84 /  0.80     0.56 / 0.42      0.75 / 0.66
============  ===============  ===============  ===============

摘要中「RSRS 右偏标准分指标与市场未来 2 周收益率的相关系数高达 75%」，
对应的即上表右偏标准分的整体相关系数 0.75。

复现时必须注意的两点
--------------------
1. **分箱方案未披露**。原报告只给出图形，未说明分箱数与分箱方式。本模块同时
   支持等宽分箱（``binning="width"``，贴近原图横轴）与等频分箱
   （``binning="quantile"``），并可用 :func:`sensitivity_to_bins` 量化结果对
   分箱方案的敏感性 —— 该敏感性本身是评估这一论断可靠性的关键。
2. **重叠窗口**。10 日窗口逐日滚动，名义样本 2 900 余个，但相互独立的观测
   仅约 293 个，相关系数的显著性被显著高估。原报告未披露分箱数，也未做
   显著性检验。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def forward_returns(close: pd.Series, horizon: int = 10) -> pd.Series:
    """未来 ``horizon`` 个交易日的收益率（用于度量预测能力，非策略输入）。"""
    return (close.shift(-horizon) / close - 1.0).rename("fwd_ret")


def _assign_bins(ind: pd.Series, binning: str, width: float, n_bins: int) -> pd.Series:
    if binning == "width":
        lo = np.floor(ind.min() / width) * width
        hi = np.ceil(ind.max() / width) * width + width
        return pd.cut(ind, np.arange(lo, hi, width))
    return pd.qcut(ind, n_bins, labels=False, duplicates="drop")


def binned_stats(
    indicator: pd.Series,
    close: pd.Series,
    horizon: int = 10,
    binning: str = "width",
    width: float = 0.2,
    n_bins: int = 40,
    min_n: int = 1,
) -> pd.DataFrame:
    """按指标分箱，返回每箱的箱中心、期望收益、上涨概率与样本数。

    Parameters
    ----------
    binning : ``"width"`` 等宽分箱（默认，贴近原报告图形横轴）；``"quantile"`` 等频分箱。
    width : 等宽分箱的箱宽（指标单位）。原报告图形步长约为 0.3。
    n_bins : 等频分箱的箱数。
    min_n : 每箱最少样本数，低于该值的箱剔除（极值箱样本极少，统计量噪声大）。
    """
    fwd = forward_returns(close, horizon)
    d = pd.DataFrame({"ind": indicator, "fwd": fwd}).replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return pd.DataFrame(columns=["center", "exp_ret", "up_rate", "n"])

    d["bin"] = _assign_bins(d["ind"], binning, width, n_bins)
    g = d.groupby("bin", observed=True).agg(
        center=("ind", "mean"),
        exp_ret=("fwd", "mean"),
        up_rate=("fwd", lambda x: float((x > 0).mean())),
        n=("fwd", "size"),
    )
    g = g[g["n"] >= min_n].sort_values("center")
    return g


def predictive_correlation(
    indicator: pd.Series,
    close: pd.Series,
    horizon: int = 10,
    binning: str = "width",
    width: float = 0.2,
    n_bins: int = 40,
    min_n: int = 1,
    side: str = "all",
    min_bins: int = 4,
    method: str = "pearson",
) -> Dict[str, float]:
    """箱中心与（期望收益、上涨概率）的相关系数。

    ``side`` 取 ``"all"`` / ``"left"``（箱中心 < 0）/ ``"right"``（箱中心 > 0）。
    """
    g = binned_stats(indicator, close, horizon=horizon, binning=binning,
                     width=width, n_bins=n_bins, min_n=min_n)
    if side == "left":
        g = g[g["center"] < 0]
    elif side == "right":
        g = g[g["center"] > 0]
    if len(g) < min_bins:
        return {"corr_exp_ret": np.nan, "corr_up_rate": np.nan, "n_bins": len(g)}
    return {
        "corr_exp_ret": float(g["center"].corr(g["exp_ret"], method=method)),
        "corr_up_rate": float(g["center"].corr(g["up_rate"], method=method)),
        "n_bins": len(g),
    }


def _indicator_specs(df: pd.DataFrame):
    from .indicators import rsrs_family

    fam18 = rsrs_family(df, n=18, m=600)
    fam16 = rsrs_family(df, n=16, m=300)
    return (
        ("标准分", fam18["zscore"]),
        ("修正标准分", fam18["modified"]),
        ("右偏标准分", fam16["right_skewed"]),
    )


def predictive_table(
    df: pd.DataFrame,
    horizon: int = 10,
    binning: str = "width",
    width: float = 0.2,
    n_bins: int = 40,
    min_n: int = 1,
    method: str = "pearson",
) -> pd.DataFrame:
    """三个指标在左侧 / 右侧 / 整体的预测性相关系数。"""
    rows = []
    for name, ind in _indicator_specs(df):
        row = {"指标": name}
        for side, label in (("left", "左侧"), ("right", "右侧"), ("all", "整体")):
            c = predictive_correlation(ind, df["close"], horizon=horizon,
                                       binning=binning, width=width, n_bins=n_bins,
                                       min_n=min_n, side=side, method=method)
            row[f"{label}_期望收益"] = c["corr_exp_ret"]
            row[f"{label}_上涨概率"] = c["corr_up_rate"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("指标")


def sensitivity_to_bins(
    df: pd.DataFrame,
    horizon: int = 10,
    widths=(0.1, 0.15, 0.2, 0.3, 0.5),
    min_n: int = 1,
) -> pd.DataFrame:
    """整体相关系数对分箱宽度的敏感性 —— 用于评估该论断的稳健性。"""
    rows = []
    for w in widths:
        t = predictive_table(df, horizon=horizon, binning="width", width=w, min_n=min_n)
        for name in t.index:
            rows.append({
                "箱宽": w, "指标": name,
                "整体_期望收益": t.loc[name, "整体_期望收益"],
                "整体_上涨概率": t.loc[name, "整体_上涨概率"],
            })
    return pd.DataFrame(rows).set_index(["指标", "箱宽"])


def effective_sample_n(close: pd.Series, horizon: int = 10) -> int:
    """重叠窗口下的近似有效样本量（按非重叠切片估计）。"""
    return int(len(close) / max(horizon, 1))

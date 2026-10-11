"""结果可视化（matplotlib 可选）。

不装 matplotlib 也能完成全部计算与 CSV 落盘；图表只在装了时生成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .backtest import BacktestResult


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无界面环境也能出图
        setup_cjk_font()
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # pragma: no cover - 取决于是否安装 matplotlib
        return None


def setup_cjk_font() -> None:
    """注册中文字体，避免图表里的中文标签显示成方框。

    依次尝试 Windows 的微软雅黑 / 黑体 / 宋体，以及 Linux / macOS 的常见
    CJK 字体；找不到就静默跳过（图表仍可用，只是中文可能乱码）。
    """
    import os

    import matplotlib
    from matplotlib import font_manager

    win = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [os.path.join(win, "Fonts", f) for f in ("msyh.ttc", "simhei.ttf", "simsun.ttc")]
    candidates += [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
            except Exception:  # pragma: no cover - 字体解析失败就试下一个
                continue


def plot_rsrs(
    df: pd.DataFrame,
    indicators: pd.DataFrame,
    result: BacktestResult,
    outdir: str | Path,
    symbol: str = "sh000300",
    title: str = "RSRS 择时回测",
    fname: Optional[str] = None,
) -> Optional[Path]:
    plt = _mpl()
    if plt is None:
        return None

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z = indicators["zscore"]
    s_buy = float(result.params.get("s_buy", 0.7))
    s_sell = float(result.params.get("s_sell", -0.7))

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2, 1.6]},
    )

    # --- 第 1 幅：指数收盘价 + 交易信号 ---
    ax = axes[0]
    ax.plot(df.index, df["close"], lw=1.0, color="#333", label="收盘价")
    pos = result.position
    buys = df.index[pos.eq(1) & pos.shift(1).fillna(0).eq(0)]
    sells = df.index[pos.eq(0) & pos.shift(1).fillna(0).eq(1)]
    ax.scatter(buys, df.loc[buys, "close"], marker="^", s=42, color="crimson", label="买入", zorder=3)
    ax.scatter(sells, df.loc[sells, "close"], marker="v", s=42, color="seagreen", label="卖出", zorder=3)
    ax.set_ylabel("指数点位")
    ax.set_title(f"{title}（{symbol}）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # --- 第 2 幅：RSRS 标准分 + 阈值 ---
    ax = axes[1]
    ax.plot(z.index, z, lw=0.8, color="#1f77b4", label="RSRS 标准分")
    ax.axhline(s_buy, color="crimson", ls="--", lw=1, label=f"买入阈值 {s_buy}")
    ax.axhline(s_sell, color="seagreen", ls="--", lw=1, label=f"卖出阈值 {s_sell}")
    ax.axhline(0, color="#999", lw=0.8)
    ax.fill_between(z.index, s_sell, s_buy, color="orange", alpha=0.08)
    ax.set_ylabel("z")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # --- 第 3 幅：净值对比 + 回撤 ---
    ax = axes[2]
    ax.plot(result.equity.index, result.equity, lw=1.2, color="#d62728", label="RSRS 策略")
    ax.plot(result.benchmark_equity.index, result.benchmark_equity, lw=1.2, color="#555", alpha=0.85, label="买入持有")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    axr = ax.twinx()
    dd = result.equity / result.equity.cummax() - 1.0
    axr.fill_between(dd.index, dd.to_numpy(), 0, color="crimson", alpha=0.25, label="回撤")
    axr.set_ylabel("回撤", color="crimson")
    axr.set_ylim(dd.min() * 1.3, 0)

    fig.tight_layout()
    path = outdir / (fname or f"rsrs_{symbol}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

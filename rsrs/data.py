"""数据加载与校验。

数据来源（本仓库已内置 sample）：
    腾讯行情接口（免费、无需 token），字段 date,open,high,low,close,volume。
    脚本 _fetch_data.py 可一键重新下载，或换成你自己的 Wind / Tushare / 聚宽数据。

数据文件格式：CSV，列名 ``date,open,high,low,close,volume``，按日期升序。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYMBOLS = {
    "sh000300": "沪深300",
    "sh000001": "上证综指",
    "sz399001": "深证成指",
}


def load_index(
    symbol: str = "sh000300",
    start: Optional[str] = None,
    end: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """读取并校验指数日行情，返回 ``DataFrame(index=date)``。

    列：open / high / low / close / volume / ret（日收益率）。
    """
    base = Path(data_dir) if data_dir else DATA_DIR
    path = base / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少数据文件 {path}。请先运行 _fetch_data.py 下载，或放入你自己的 CSV。"
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date")
    df = df.set_index("date")

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"数据缺少必要列: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df.dropna(subset=["open", "high", "low", "close"])

    # ---- 基本数据校验：抓明显的脏数据 ----
    # 免费数据源存在 0.001~0.01 级的取整误差（例如 high 比 open 低 0.003），
    # 因此校验带一个很小的点位容差；超出容差判定为源数据错误，直接丢弃并告警。
    tol = 0.01
    invalid = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1) - tol)
        | (df["low"] > df[["open", "close", "high"]].min(axis=1) + tol)
    )
    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"[data] 发现 {n_invalid} 行 OHLC 自相矛盾（源数据错误），已丢弃：")
        print(f"       {df.index[invalid].date.tolist()[:10]}")
        df = df[~invalid]
    if (df["close"] <= 0).any():
        df = df[df["close"] > 0]

    df["ret"] = df["close"] / df["close"].shift(1) - 1.0

    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df

# -*- coding: utf-8 -*-
"""一键重新下载指数日行情（腾讯行情接口，免费、无需 token）。

用法::

    python _fetch_data.py

产出 data/sh000300.csv、data/sh000001.csv、data/sz399001.csv，
字段 date,open,high,low,close,volume，按日期升序。
日常复现请优先使用你自己的数据源（Wind / Tushare / 聚宽），并核对口径。
"""
import json
import ssl
import time
from datetime import datetime, timedelta
from pathlib import Path

import urllib.request

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
START, END, BATCH = "2005-01-01", "2024-12-31", 2000


def tencent(symbol, start, end, count=BATCH):
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,{start},{end},{count},qfq"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        data = json.loads(r.read().decode("utf-8", "ignore"))
    node = (data.get("data") or {}).get(symbol, {})
    rows = []
    for k in node.get("qfqday") or node.get("day") or []:
        rows.append((k[0], float(k[1]), float(k[3]), float(k[4]), float(k[2]), float(k[5])))
    return rows


def fetch_all(symbol):
    all_rows, end = [], END
    while True:
        rows = tencent(symbol, START, end)
        if not rows:
            break
        existing = {r[0] for r in all_rows}
        all_rows = [r for r in rows if r[0] not in existing] + all_rows
        first = all_rows[0][0]
        if len(rows) < BATCH or first <= START:
            break
        end = (datetime.strptime(first, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(0.6)
    seen = {}
    for r in all_rows:
        seen[r[0]] = r
    return [seen[d] for d in sorted(seen)]


def main():
    for symbol in ("sh000300", "sh000001", "sz399001"):
        rows = fetch_all(symbol)
        p = OUT / f"{symbol}.csv"
        with open(p, "w", encoding="utf-8") as f:
            f.write("date,open,high,low,close,volume\n")
            f.write("\n".join(",".join(str(x) for x in r) for r in rows) + "\n")
        print(f"[OK] {symbol}: {len(rows)} 行, {rows[0][0]} ~ {rows[-1][0]} -> {p}")
        time.sleep(0.6)


if __name__ == "__main__":
    main()

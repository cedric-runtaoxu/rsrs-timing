# -*- coding: utf-8 -*-
"""一键重新下载指数日行情（腾讯行情接口，免费、无需 token）。

用法::

    python _fetch_data.py                    # 写入 ./data/
    python _fetch_data.py --outdir D:\\x\\data
    python _fetch_data.py --symbols sh000300 sh000016

产出 CSV 字段：date,open,high,low,close,volume，按日期升序。

接口说明
--------
* 端点：``https://web.ifzq.gtimg.cn/appstock/app/fqkline/get``
* 单次最多返回约 2000 根 K 线，故按结束日期向前分页回溯；
* 对**指数**而言 ``qfq`` 参数无实际作用（指数不除权），此处仅为接口兼容保留；
* 该接口**不提供成交额字段**，只有成交量。本项目的策略均不需要成交额；
* 接口非官方、无文档，可能随时变更或限流。仓库内已附带数据快照，
  日常复现无需联网；如需更新数据，建议换用 Wind / Tushare / 聚宽并核对口径。
"""
import argparse
import json
import ssl
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

CTX = ssl.create_default_context()
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

START, END, BATCH = "2005-01-01", "2024-12-31", 2000

SYMBOLS = {
    "sh000300": "沪深300",
    "sh000016": "上证50",
    "sh000905": "中证500",
    "sh000001": "上证综指",
    "sz399001": "深证成指",
}


def tencent(symbol, start, end, count=BATCH):
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={symbol},day,{start},{end},{count},qfq")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        data = json.loads(r.read().decode("utf-8", "ignore"))
    node = (data.get("data") or {}).get(symbol, {})
    rows = []
    # 接口原始顺序：[日期, 开, 收, 高, 低, 成交量]
    for k in node.get("qfqday") or node.get("day") or []:
        rows.append((k[0], float(k[1]), float(k[3]), float(k[4]), float(k[2]), float(k[5])))
    return rows


def fetch_all(symbol, start=START, end=END):
    all_rows, cursor = [], end
    while True:
        rows = tencent(symbol, start, cursor)
        if not rows:
            break
        existing = {r[0] for r in all_rows}
        all_rows = [r for r in rows if r[0] not in existing] + all_rows
        first = all_rows[0][0]
        if len(rows) < BATCH or first <= start:
            break
        cursor = (datetime.strptime(first, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(0.6)
    seen = {}
    for r in all_rows:
        seen[r[0]] = r
    return [seen[d] for d in sorted(seen)]


def main():
    ap = argparse.ArgumentParser(description="下载指数日行情")
    ap.add_argument("--outdir", default=None, help="输出目录，默认 ./data")
    ap.add_argument("--start", default=START, help="请求起点 YYYY-MM-DD")
    ap.add_argument("--end", default=END, help="请求截止 YYYY-MM-DD；输出以实际返回日期为准")
    ap.add_argument("--symbols", nargs="*", default=list(SYMBOLS), choices=list(SYMBOLS))
    args = ap.parse_args()

    out = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent / "data"
    out.mkdir(parents=True, exist_ok=True)

    for symbol in args.symbols:
        rows = fetch_all(symbol, start=args.start, end=args.end)
        rows = [r for r in rows if args.start <= r[0] <= args.end]
        if not rows:
            raise RuntimeError(f"{symbol}: 请求区间无数据，不覆盖文件")
        p = out / f"{symbol}.csv"
        with open(p, "w", encoding="utf-8") as f:
            f.write("date,open,high,low,close,volume\n")
            f.write("\n".join(",".join(str(x) for x in r) for r in rows) + "\n")
        print(f"[OK] {SYMBOLS[symbol]}({symbol}): {len(rows)} 行, "
              f"{rows[0][0]} ~ {rows[-1][0]} -> {p}")
        time.sleep(0.6)


if __name__ == "__main__":
    main()

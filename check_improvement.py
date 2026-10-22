"""核查改进研究的配置、产物、哈希及完整披露。"""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent


def check(out=None):
    out=Path(out) if out else ROOT/"report"/"improvement"
    meta=json.loads((out/"metadata.json").read_text(encoding="utf-8"))
    config=meta["config"]
    assert config["execution"]=="next_close" and config["cost_one_way"]==.001
    assert config["improved"]=={
        "base":"right_skewed(18,600)","rank_window":756,"rank_min_periods":252,
        "entry_quantile":.8,"exit_quantile":.2,"trend_window":250,"trend_slope_gap":20,
    }
    for path,sha in meta["source_sha256"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    for path,sha in meta["code_sha256"].items():
        data=(ROOT/path).read_bytes().replace(b"\r\n",b"\n")
        assert hashlib.sha256(data).hexdigest()==sha,path
    for path,sha in json.loads((out/"generated_sha256.json").read_text(encoding="utf-8")).items():
        assert hashlib.sha256((out/path).read_bytes()).hexdigest()==sha,path
    performance=pd.read_csv(out/"performance.csv")
    assert len(performance)==3*6*5
    sensitivity=pd.read_csv(out/"sensitivity.csv")
    assert len(sensitivity)==27*4
    assert len(sensitivity[["滚动窗口","趋势窗口","进入分位","退出分位"]].drop_duplicates())==27
    signals=pd.read_csv(out/"signals.csv")
    entries=signals["entry_condition"].astype(str).str.lower().eq("true")
    assert (signals.loc[entries,"indicator_rank"]>=.8).all()
    assert signals.loc[entries,"price_above_trend"].astype(str).str.lower().eq("true").all()
    assert signals.loc[entries,"trend_up"].astype(str).str.lower().eq("true").all()
    print("PASS: 改进配置、90组绩效、27组邻域、逐日入场条件及全部哈希")


if __name__=="__main__": check()

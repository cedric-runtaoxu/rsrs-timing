"""核查逐章复现产物、口径与可追溯性，不计算研报数字匹配率。"""
from pathlib import Path
import hashlib
import json
import pandas as pd
from rsrs.paper_study import N_GRID,M_GRID,STAT_ROWS

ROOT=Path(__file__).resolve().parent

def check(out=None):
    out=Path(out) if out else ROOT/'report/paper'
    meta=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    assert meta['spec']['cost']==0 and meta['spec']['execution']=='next_close'
    assert tuple(meta['n_grid'])==N_GRID and tuple(meta['m_grid'])==M_GRID
    coverage=pd.read_csv(out/'coverage.csv')
    assert len(coverage)==42
    assert set(coverage.loc[coverage['类别']=='图','编号'])==set(range(1,33))
    assert set(coverage.loc[coverage['类别']=='表','编号'])==set(range(1,11))
    for file in coverage['结果文件']:
        assert (out/file).is_file(),file
    for path,sha in meta['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    for path,sha in meta['code_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==sha,path
    for path,sha in json.loads((out/'generated_sha256.json').read_text(encoding='utf-8')).items():
        assert hashlib.sha256((out/path).read_bytes()).hexdigest()==sha,path
    for n in [2,4,5,6,8]:
        tab=pd.read_csv(out/f'tables/table_{n:02d}.csv').set_index('统计量')
        assert list(tab.index)==STAT_ROWS
        assert ((tab.loc['获利天数']+tab.loc['亏损天数'])<=tab.loc['持仓总天数']).all()
        assert ((tab.loc['盈利次数']+tab.loc['亏损次数'])<=tab.loc['交易次数']).all()
    assert pd.read_csv(out/'data/account_checks.csv')['独立核账误差'].max()<1e-12
    assert pd.read_csv(out/'data/prefix_checks.csv')['历史信号一致'].all()
    cross=pd.read_csv(out/'data/cross_market_costs.csv')
    assert len(cross)==12 and set(cross['实际终点'])=={'2017-03-31'}
    assert set(cross['单边成本'])=={0,.002,.003}
    print('PASS: 32图主题、10表、原文网格、费用与期间、数据及产物哈希、交易统计和独立核账')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,default=ROOT/'report/paper')
    check(p.parse_args().outdir)

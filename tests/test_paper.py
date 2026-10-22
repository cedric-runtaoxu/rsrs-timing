"""原文式统计与分箱的手算测试；python tests/test_paper.py。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
import numpy as np
import pandas as pd
from rsrs.backtest import backtest
from rsrs.paper_study import period_stats,grouped_prediction,PAPER_SPEC,N_GRID,M_GRID
from rsrs.research_spec import SPEC


def test_long_short_trade_statistics_have_hand_computed_answer():
    idx=pd.bdate_range('2005-04-08',periods=7)
    d=pd.DataFrame({'ret':[0.,0.,.1,-.1,0.,-.1,.1]},index=idx)
    bt=backtest(d,pd.Series([1.,1.,0.,-1.,-1.,0.,0.],index=idx),cost=0)
    st=period_stats(bt)
    assert st['持仓总天数']==4 and st['交易次数']==2
    assert st['获利天数']==st['亏损天数']==2
    assert st['胜率（按天）']==.5 and st['胜率（按次）']==0
    assert st['平均持仓天数']==2
    assert abs(st['平均亏损率（按次）']+.01)<1e-12
    assert abs(st['期末净值']-.9801)<1e-12


def test_zero_days_and_unclosed_trade_are_not_silently_dropped():
    idx=pd.bdate_range('2005-04-08',periods=4)
    bt=backtest(pd.DataFrame({'ret':[0,0,0,.1]},index=idx),pd.Series(1.,index=idx),cost=0)
    st=period_stats(bt)
    assert st['持仓总天数']==2 and st['零收益持仓天数']==1
    assert st['胜率（按天）']==.5 and st['交易次数']==1
    assert st['期末成交仓位']==1
    sliced=period_stats(bt,str(idx[2].date()),str(idx[-1].date()))
    assert sliced['左端结转持仓'] and sliced['持仓总天数']==2


def test_prediction_excludes_unknown_tail_and_future_append():
    idx=pd.bdate_range('2005-04-08',periods=30)
    close=pd.Series(np.arange(100.,130.),index=idx)
    indicator=pd.Series(np.linspace(-2,2,30),index=idx)
    end=str(idx[17].date())
    g=grouped_prediction(indicator,close,bins=4,end=end)
    assert g['样本数'].sum()==8
    assert (g.loc[g['样本数']>0,'上涨概率']==1).all()
    altered=close.copy();altered.iloc[18:]=99999
    pd.testing.assert_frame_equal(g,grouped_prediction(indicator,altered,bins=4,end=end))
    pairs=(close.shift(-10)/close-1).iloc[:8]
    mean=(g['未来10日收益']*g['样本数']).sum()/8
    assert abs(mean-pairs.mean())<1e-12


def test_paper_and_extension_rules_differ_only_in_explicit_cost_and_end():
    a,b=asdict(PAPER_SPEC),asdict(SPEC)
    assert {k for k in a if a[k]!=b[k]}=={'cost','end'}
    assert a['cost']==0 and b['cost']==.001
    assert N_GRID==tuple(range(14,26))
    assert M_GRID==tuple(range(450,801,50))


if __name__=='__main__':
    tests=[v for k,v in list(globals().items()) if k.startswith('test_')]
    for test in tests:
        test();print('[PASS]',test.__name__)
    print(f'{len(tests)}/{len(tests)} passed')

"""有手算答案的执行、收益及因果性回归测试。python tests/test_research.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from rsrs.backtest import backtest, total_return, max_drawdown
from rsrs.indicators import rsrs_family
from rsrs.strategies import filtered_signal
from rsrs.data import load_index


def test_close_execution_has_no_pre_fill_return():
    # t0 信号；t1 以 110 买入；t2 以 121 卖出。只赚 10%，不能赚 t0->t1。
    d=pd.DataFrame({'close':[100.,110.,121.,108.9]})
    d['ret']=d.close.pct_change()
    r=backtest(d,pd.Series([1.,0.,0.,0.]),cost=.001)
    np.testing.assert_allclose(r.ret,[0.,-.001,.10-.001,0.],atol=1e-12)
    assert r.n_switches==2
    assert r.n_trades==1


def test_short_trade_direction_applied_once():
    d=pd.DataFrame({'ret':[0.,0.,-.1,0.]})
    r=backtest(d,pd.Series([-1.,0.,0.,0.]),cost=0)
    assert abs(r.trades.iloc[0]['ret']-.1)<1e-12


def test_initial_loss_and_drawdown_count():
    e=pd.Series([.9,.95])
    assert abs(total_return(e)+.05)<1e-12
    assert abs(max_drawdown(e)+.1)<1e-12


def test_no_phantom_switch():
    r=backtest(pd.DataFrame({'ret':[0.,.1,-.1]}),pd.Series([0.,0.,0.]))
    assert r.n_switches==0


def test_filter_cannot_enter_in_neutral_zone():
    s=pd.Series([.8,.2,.9,.2,-.8])
    a=pd.Series([False,True,True,False,False])
    assert filtered_signal(s,a).tolist()==[0.,0.,1.,1.,0.]


def test_price_and_signal_prefix_invariance():
    rng=np.random.default_rng(42)
    c=pd.Series(100*np.exp(np.cumsum(rng.normal(0,.01,100))))
    d=pd.DataFrame({'high':c*1.01,'low':c*.99})
    first=rsrs_family(d.iloc[:65],n=18,m=30)
    full=rsrs_family(d,n=18,m=30)
    pd.testing.assert_frame_equal(first,full.iloc[:65])


def test_mask_keeps_bad_day_and_close_return():
    d=load_index('sh000300',invalid_ohlc='mask')
    t=pd.Timestamp('2015-03-27')
    assert t in d.index and np.isnan(d.loc[t,'high'])
    assert pd.notna(d.loc[t,'ret'])


def test_predictive_labels_stay_inside_period_and_align_execution():
    from research import sample_for_prediction
    dates=pd.bdate_range('2020-01-01',periods=8)
    close=pd.Series([100.,110.,121.,130.,150.,140.,160.,170.],index=dates)
    ind=pd.Series(1.,index=dates)
    pairs=sample_for_prediction(ind,close,str(dates[0].date()),str(dates[5].date()),2,1)
    assert len(pairs)==3  # t0,t1,t2；t3的t+1+2已越过区间终点
    assert abs(pairs.y.iloc[0]-(130/110-1))<1e-12
    assert pairs.label_end.max()==dates[5]


def test_paired_bootstrap_identical_series_has_zero_interval():
    from research import bootstrap_mean_difference
    a=pd.Series(np.random.default_rng(8).normal(0,.01,300))
    result=bootstrap_mean_difference(a,a,block=30,reps=100)
    assert result['日均超额年化']==result['CI下']==result['CI上']==0


def test_research_configuration_is_immutable():
    from dataclasses import FrozenInstanceError
    from rsrs.research_spec import SPEC
    try:
        SPEC.cost=.002
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('主研究配置必须不可变')


def test_append_future_does_not_change_past_strategy_or_account():
    from research import signals, run_strategy
    rng=np.random.default_rng(73)
    idx=pd.bdate_range('2015-01-01',periods=1000)
    close=pd.Series(100*np.exp(np.cumsum(rng.normal(0,.015,len(idx)))),index=idx)
    d=pd.DataFrame({'close':close,'high':close*(1+rng.uniform(.001,.03,len(idx))),
                    'low':close*(1-rng.uniform(.001,.03,len(idx))),
                    'volume':rng.lognormal(10,1,len(idx)),'ret':close.pct_change()})
    whole,_=signals(d)
    prefix,_=signals(d.iloc[:800])
    for name in whole:
        pd.testing.assert_series_equal(whole[name].iloc[:800],prefix[name])
        full=run_strategy(d,whole[name])
        partial=run_strategy(d.iloc[:800],prefix[name])
        pd.testing.assert_series_equal(full.ret.iloc[:800],partial.ret)
        pd.testing.assert_series_equal(full.execution_position.iloc[:800],partial.execution_position)


def test_ma_filter_requires_specified_two_observation_dates():
    from rsrs.strategies import ma_filter
    close=pd.Series([10.,10.,20.,10.,10.,10.,10.])
    actual=ma_filter(close,window=2)
    # t4: MA(t3)=15>MA(t1)=10；t5: MA(t4)=10<MA(t2)=15。
    assert actual.iloc[4] and not actual.iloc[5]


def test_short_exit_does_not_require_entry_filter():
    from rsrs.strategies import long_short_position
    x=pd.Series([-.8,.1,.8,.8,-.8,-.8])
    up=pd.Series([False,False,False,True,False,False])
    down=pd.Series([True,False,False,False,False,True])
    assert long_short_position(x,up,down).tolist()==[-1.,-1.,0.,1.,0.,-1.]


def test_volume_short_uses_positive_correlation_for_both_directions():
    from rsrs.strategies import long_short_position
    x=pd.Series([.8,-.8,.8,-.8])
    positive=pd.Series([True,True,False,False])
    assert long_short_position(x,positive,positive).tolist()==[1.,-1.,0.,0.]


def test_threshold_equality_missing_and_neutral_preserve_position():
    from rsrs.strategies import long_short_position
    x=pd.Series([.8,.7,-.7,np.nan,0.,-.8,-.7,.7,np.nan])
    yes=pd.Series(True,index=x.index)
    assert long_short_position(x,yes,yes).tolist()==[1.,1.,1.,1.,1.,-1.,-1.,-1.,-1.]


def test_reversal_cost_and_short_holding_cost_hand_calculation():
    d=pd.DataFrame({'ret':[0.,.5,.1,-.1,.2]})
    s=pd.Series([1.,-1.,0.,0.,0.])
    r=backtest(d,s,cost=.001,short_cost=.252)
    np.testing.assert_allclose(r.execution_position,[0,1,-1,0,0])
    np.testing.assert_allclose(r.ret,[0,-.001,.098,.098,0],atol=1e-12)
    assert r.turnover.sum()==4  # 开多1、反手2、平空1
    assert abs(r.short_carry.sum()-.001)<1e-12


def test_trade_ledger_uses_execution_dates_and_keeps_open_status():
    from rsrs.research_checks import execution_trades,independent_account_check
    dates=pd.bdate_range('2020-01-01',periods=5)
    d=pd.DataFrame({'close':[100.,110.,121.,108.9,100.]},index=dates)
    d['ret']=d.close.pct_change()
    s=pd.Series([1.,-1.,0.,1.,1.],index=dates)
    r=backtest(d,s,cost=.001)
    tr=execution_trades(d,r.execution_position)
    assert tr.iloc[0]['建仓日']==dates[1] and tr.iloc[0]['平仓日']==dates[2]
    assert abs(tr.iloc[0]['毛收益']-.1)<1e-12
    assert abs(tr.iloc[1]['毛收益']-.1)<1e-12
    assert not tr.iloc[-1]['已平仓'] and tr.iloc[-1]['持仓收益天数']==0
    assert independent_account_check(d,s,r)<1e-12


def test_indicator_matches_independent_least_squares():
    from rsrs.indicators import rsrs_beta,rsrs_r2
    lo=pd.Series([10.,11.,13.,12.,15.,14.])
    hi=pd.Series([12.,13.,15.,15.,17.,17.])
    x=np.column_stack([np.ones(6),lo])
    coeff=np.linalg.lstsq(x,hi,rcond=None)[0]
    residual=hi-x@coeff
    expected_r2=1-float((residual**2).sum()/((hi-hi.mean())**2).sum())
    assert abs(rsrs_beta(hi,lo,6).iloc[-1]-coeff[1])<1e-12
    assert abs(rsrs_r2(hi,lo,6).iloc[-1]-expected_r2)<1e-12


def test_all_paper_rules_covered_without_custom_improvements():
    from rsrs.research_spec import strategy_manifest,LONG_STRATEGIES,SHORT_STRATEGIES
    manifest=strategy_manifest()['strategies']
    assert len(LONG_STRATEGIES)==10 and len(SHORT_STRATEGIES)==6 and len(manifest)==20
    assert not any('改进' in name or 'MA200' in name for name in manifest)
    assert all(manifest[name]['direction']=='long_short' for name in SHORT_STRATEGIES)


def test_backtest_rejects_invalid_index_and_exposure():
    for d,s in [(pd.DataFrame({'ret':[0.,.1]},index=[1,1]),pd.Series([0.,0.])),
                (pd.DataFrame({'ret':[0.,.1]}),pd.Series([0.,2.]))]:
        try:backtest(d,s)
        except ValueError:pass
        else:raise AssertionError('非法输入必须被拒绝')


if __name__=='__main__':
    tests=[v for k,v in list(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('[PASS]',test.__name__)
    print(f'{len(tests)}/{len(tests)} passed')

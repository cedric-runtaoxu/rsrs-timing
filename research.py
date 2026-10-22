"""RSRS 独立有效性研究。python research.py [--no-plots]

研究方案见 docs/research_protocol.md。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from rsrs.data import load_index, SYMBOLS
from rsrs.indicators import rsrs_family, rsrs_signal
from rsrs.strategies import ma_filter, ma_direction, long_short_position, volume_corr_filter, filtered_signal
from rsrs.backtest import backtest, annualized_return, annualized_vol, sharpe, max_drawdown
from rsrs.predictive import binned_stats
from rsrs.research_spec import SPEC, PAPER_STRATEGIES, LONG_STRATEGIES, SHORT_STRATEGIES, CONTROLS, BASELINES, ALL_STRATEGIES, strategy_manifest
from rsrs.research_checks import execution_trades, independent_account_check, raw_data_quality
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'report'/'research'
PERIODS={
    '原报告期':('2005-04-08','2017-04-28'),
    '发表后前段':('2017-05-02','2021-12-31'),
    '近期检验':('2022-01-01',SPEC.end),
    '发表后全部':('2017-05-02',SPEC.end),
    '2026年内':('2026-01-01',SPEC.end),
}
MAIN=['买入持有','标准分','右偏(16,300)','价格过滤·右偏','量过滤·右偏',
      '多空·价格过滤·右偏','多空·量过滤·右偏','MA20','BOLL(14,2)']


def signals(d, spec=SPEC):
    f=rsrs_family(d,spec.n,spec.m,min_periods=spec.min_periods,exclude_current=spec.exclude_current)
    f16=rsrs_family(d,spec.right_n,spec.right_m,min_periods=min(spec.min_periods,spec.right_m),exclude_current=spec.exclude_current)
    ma,down=ma_direction(d.close,spec.ma_window)
    vol=volume_corr_filter(d.volume,f.modified,spec.volume_window).shift(spec.volume_lag).fillna(False).astype(bool)
    sig={'买入持有':pd.Series(1.,index=d.index),
         '斜率':rsrs_signal(f.beta,spec.beta_buy,spec.beta_sell),
         '标准分':rsrs_signal(f.zscore,spec.buy,spec.sell),
         '修正标准分':rsrs_signal(f.modified,spec.buy,spec.sell),
         '右偏(16,300)':rsrs_signal(f16.right_skewed,spec.buy,spec.sell),
         '右偏(18,600)':rsrs_signal(f.right_skewed,spec.buy,spec.sell)}
    for label,col in [('标准分','zscore'),('修正','modified'),('右偏','right_skewed')]:
        sig['价格过滤·'+label]=filtered_signal(f[col],ma,spec.buy,spec.sell)
        sig['量过滤·'+label]=filtered_signal(f[col],vol,spec.buy,spec.sell)
        sig['多空·价格过滤·'+label]=long_short_position(f[col],ma,down,spec.buy,spec.sell)
        sig['多空·量过滤·'+label]=long_short_position(f[col],vol,vol,spec.buy,spec.sell)
    average=d.close.rolling(spec.ma_window,min_periods=spec.ma_window).mean()
    sig['MA20']=rsrs_signal(d.close-average,0,0)
    center=d.close.rolling(spec.boll_window,min_periods=spec.boll_window).mean()
    sd=d.close.rolling(spec.boll_window,min_periods=spec.boll_window).std(ddof=spec.boll_ddof)
    standardized=(d.close-center)/sd.replace(0,np.nan)
    sig['BOLL(14,2)']=rsrs_signal(standardized,spec.boll_width,-spec.boll_width)
    return sig,f


def run_strategy(d, signal, spec=SPEC):
    return backtest(d,signal,cost=spec.cost,lag=spec.lag,execution=spec.execution,short_cost=spec.short_cost)


def consistency_checks(d, full_signals, full_results, market):
    """在三个截点独立截断输入；后来数据不得改变已有信号、仓位或收益。"""
    rows=[]
    for cutoff in ['2017-04-28','2021-12-31','2024-12-31']:
        prefix=d.loc[:cutoff]
        partial_signals,_=signals(prefix)
        for name,signal in partial_signals.items():
            partial=run_strategy(prefix,signal)
            full=full_results[name]
            checks={'信号一致':np.allclose(signal,full_signals[name].loc[prefix.index],atol=1e-12,rtol=1e-12),
                    '损益仓位一致':np.allclose(partial.position,full.position.loc[prefix.index],atol=1e-12,rtol=1e-12),
                    '成交仓位一致':np.allclose(partial.execution_position,full.execution_position.loc[prefix.index],atol=1e-12,rtol=1e-12),
                    '收益一致':np.allclose(partial.ret,full.ret.loc[prefix.index],atol=1e-12,rtol=1e-12)}
            rows.append({'市场':market,'截止日':cutoff,'策略':name,**checks})
            if not all(checks.values()):
                raise AssertionError(f'延长样本改变已有计算：{market}/{cutoff}/{name}: {checks}')
    return rows


def metrics(ret,position,turnover):
    ret=ret.fillna(0)
    eq=(1+ret).cumprod()
    ann=annualized_return(ret)
    mdd=max_drawdown(eq)
    # 对连续调仓策略统计从空仓到有仓的次数，不能把每次调仓称为一笔完整交易。
    return {'实际起点':str(ret.index[0].date()),'实际终点':str(ret.index[-1].date()),
            '天数':len(ret),'累计收益':float(eq.iloc[-1]-1),'年化收益':ann,
            '年化波动':annualized_vol(ret),'夏普':sharpe(ret),'最大回撤':mdd,
            'Calmar':ann/abs(mdd) if mdd<0 else np.nan,
            '平均净仓位':position.mean(),'平均绝对仓位':position.abs().mean(),
            '多头天数':int((position>0).sum()),'空头天数':int((position<0).sum()),
            '空仓天数':int((position==0).sum()),'年均单边换手':turnover.sum()/len(ret)*252}


def frame_markdown(frame, formats=None):
    formats=formats or {}
    def fmt(k,v):
        if pd.isna(v): return '—'
        if k in formats: return formats[k].format(v)
        if isinstance(v,(float,np.floating)): return f'{v:.3f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(fmt(k,v) for k,v in row.items())+' |' for _,row in frame.iterrows()])


def sample_for_prediction(ind,close,start,end,h,entry_delay=0):
    dates=pd.Series(close.index,index=close.index)
    future=close.shift(-entry_delay-h)/close.shift(-entry_delay)-1
    d=pd.DataFrame({'x':ind,'y':future,'label_end':dates.shift(-entry_delay-h)})
    # 不允许期末的信号标签跨越分期边界。
    return d.loc[start:end].loc[lambda x:x.label_end<=pd.Timestamp(end)].dropna()


def prediction_tables(d,f):
    rows,bins,phases=[],[],[]
    for col in ['beta','zscore','modified','right_skewed','right_skewed_16_300']:
        hist=f.loc[:'2017-04-28',col].dropna()
        edges=np.r_[-np.inf,np.unique(hist.quantile([.2,.4,.6,.8]).to_numpy()),np.inf]
        for period,(start,end) in PERIODS.items():
            for h,entry_delay in product([1,5,10,20],[0,1]):
                p=sample_for_prediction(f[col],d.close,start,end,h,entry_delay)
                if len(p)<30: continue
                x=(p.x-p.x.mean())/p.x.std()
                for lag in ([10,20,60] if h==10 else [max(20,h-1)]):
                    fit=sm.OLS(p.y,sm.add_constant(x)).fit(cov_type='HAC',cov_kwds={'maxlags':lag})
                    ci=fit.conf_int().iloc[1]
                    rows.append({'分期':period,'指标':col,'预测天数':h,'标签起点延迟':entry_delay,'样本数':len(p),'HAC阶数':lag,
                                 'Pearson':p.x.corr(p.y),'Spearman':p.x.corr(p.y,method='spearman'),
                                 '每SD收益系数':fit.params.iloc[1],'HAC_t':fit.tvalues.iloc[1],
                                 'HAC_p探索':fit.pvalues.iloc[1],'系数CI下':ci.iloc[0],'系数CI上':ci.iloc[1]})
                if h==10 and entry_delay==0:
                    p=p.copy()
                    p['组']=pd.cut(p.x,edges,labels=False,include_lowest=True)+1
                    for group,g in p.groupby('组'):
                        bins.append({'分期':period,'指标':col,'组':group,'样本数':len(g),
                                     '指标均值':g.x.mean(),'未来10日均值':g.y.mean(),'上涨比例':(g.y>0).mean()})
                    for offset in range(h):
                        q=p.iloc[offset::h]
                        phases.append({'分期':period,'指标':col,'起点偏移':offset,'样本数':len(q),
                                       'Pearson':q.x.corr(q.y)})
    return pd.DataFrame(rows),pd.DataFrame(bins),pd.DataFrame(phases)


def bootstrap_mean_difference(a,b,block=60,reps=2000):
    """成对循环移动块bootstrap；估计日均超额收益的年化区间，不是CAGR。"""
    x=(a-b).dropna().to_numpy()
    rng=np.random.default_rng(20261022)
    n=len(x)
    sampled=[]
    for _ in range(reps):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        indexes=(starts[:,None]+np.arange(block))%n
        sampled.append(x[indexes.ravel()[:n]].mean()*252)
    lo,hi=np.quantile(sampled,[.025,.975])
    return {'日均超额年化':x.mean()*252,'CI下':lo,'CI上':hi,'块长':block,'重复':reps}


def plot_results(m,p,b,returns):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'figure.dpi':150,'axes.spines.top':False,
                         'axes.spines.right':False,'font.size':10})
    colors=['#526270','#187F9C','#A65722','#7665A6','#438D60']
    chosen=['买入持有','价格过滤·右偏','多空·价格过滤·右偏','量过滤·右偏','多空·量过滤·右偏']
    r=returns.loc['2017-05-02':,chosen]
    fig,ax=plt.subplots(2,1,figsize=(11,7.2),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for name,c in zip(chosen,colors):
        eq=(1+r[name]).cumprod()
        ax[0].plot(eq.index,eq,label=name,color=c,lw=1.4)
        ax[1].plot(eq.index,eq/eq.cummax().clip(lower=1)-1,color=c,lw=1)
    ax[0].set_title('沪深300｜原报告发表后：固定规则、次日收盘执行、单边成本0.1%',loc='left',pad=15)
    ax[0].set_ylabel('分期起始资金 = 1');ax[1].set_ylabel('回撤')
    ax[0].legend(ncol=3,fontsize=9,loc='upper left')
    for a in ax:a.grid(alpha=.18)
    fig.tight_layout();fig.savefig(OUT/'01_post_publication.png');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    periods=['原报告期','发表后前段','近期检验']
    for period,c in zip(periods,colors):
        q=p[(p['分期']==period)&(p['指标']=='right_skewed')&(p['标签起点延迟']==0)&((p['预测天数']!=10)|(p['HAC阶数']==20))]
        ax[0].plot(q['预测天数'],q.Pearson,'o-',label=period,color=c)
        q=b[(b['分期']==period)&(b['指标']=='right_skewed')]
        ax[1].plot(q['组'],q['未来10日均值']*100,'o-',label=period,color=c)
    ax[0].set_title('逐日相关性：右偏指标(18,600)');ax[0].set_xlabel('未来交易日数');ax[0].set_ylabel('Pearson相关系数')
    ax[1].set_title('使用2017年前固定五分位切点');ax[1].set_xlabel('低信号 → 高信号组');ax[1].set_ylabel('未来10日平均收益（%）')
    for a in ax:a.axhline(0,color='#777777',lw=.7);a.grid(alpha=.18);a.legend(fontsize=9)
    fig.suptitle('分箱趋势与逐日预测能力需要分开判断',x=.06,ha='left',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'02_predictive.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    q=m[(m['市场']=='沪深300')&(m['分期']=='发表后全部')&(m['策略'].isin(MAIN))].set_index('策略').loc[MAIN]
    x=np.arange(len(q));ax.bar(x-.18,q['年化收益']*100,.36,label='年化收益',color=colors[1]);ax.bar(x+.18,q['最大回撤']*100,.36,label='最大回撤',color=colors[2])
    ax.set_xticks(x,q.index,rotation=30,ha='right');ax.axhline(0,color='#777777',lw=.7);ax.set_ylabel('%')
    ax.set_title('沪深300发表后收益与回撤｜多头、多空与技术基线',loc='left',pad=14)
    ax.legend();ax.grid(axis='y',alpha=.18);fig.tight_layout();fig.savefig(OUT/'03_risk_return.png');plt.close(fig)



def main():
    from dataclasses import replace
    from rsrs.research_reporting import write_report
    ap=argparse.ArgumentParser()
    ap.add_argument('--no-plots',action='store_true')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=strategy_manifest()
    source_manifest=json.loads((ROOT/'data/extended/manifest.json').read_text())
    all_metrics=[];gross_metrics=[];yearly=[];consistency=[];accounts=[];quality=[]
    prediction=[];fixed_bins=[];phases=[];distributions=[];annual_ic=[];ledgers=[];common_window=[]
    for symbol in ['sh000300','sh000016','sh000905']:
        market=SYMBOLS[symbol]
        source=ROOT/'data'/'extended'/f'{symbol}.csv'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_manifest['symbols'][symbol]['sha256'], '行情快照校验值与来源清单不符'
        audit=raw_data_quality(source,market)
        assert all(audit[col]==0 for col in ['重复日期','价格缺失行','非正价格行','负成交量行']), '行情基础检查失败'
        quality.append(audit)
        d=load_index(symbol,end=SPEC.end,data_dir=source.parent,invalid_ohlc=SPEC.invalid_ohlc)
        sig,f=signals(d)
        results={name:run_strategy(d,sig[name]) for name in ALL_STRATEGIES}
        assert set(sig)==set(manifest['strategies']), '策略配置未覆盖实际运行策略'
        consistency.extend(consistency_checks(d,sig,results,market))
        f=f.assign(right_skewed_16_300=rsrs_family(d,SPEC.right_n,SPEC.right_m,
                   min_periods=SPEC.min_periods,exclude_current=SPEC.exclude_current).right_skewed)
        pred,bins,phase=prediction_tables(d,f)
        prediction.append(pred.assign(市场=market));fixed_bins.append(bins.assign(市场=market))
        phases.append(phase.assign(市场=market))
        for period,(start,end) in PERIODS.items():
            for col in ['beta','zscore','modified','right_skewed','right_skewed_16_300']:
                x=f.loc[start:end,col].dropna()
                distributions.append({'市场':market,'分期':period,'指标':col,'样本数':len(x),
                                      '均值':x.mean(),'标准差':x.std(),'偏度':x.skew(),'超额峰度':x.kurt(),
                                      '最小值':x.min(),'最大值':x.max()})
        for year in d.index.year.unique():
            for col in ['beta','zscore','modified','right_skewed','right_skewed_16_300']:
                q=sample_for_prediction(f[col],d.close,f'{year}-01-01',f'{year}-12-31',10,1)
                if len(q)>20:
                    annual_ic.append({'市场':market,'年份':year,'指标':col,'样本数':len(q),
                                      'Pearson':q.x.corr(q.y),'Spearman':q.x.corr(q.y,method='spearman')})
        for name,r in results.items():
            error=independent_account_check(d,sig[name],r,SPEC.cost,SPEC.lag,SPEC.short_cost)
            accounts.append({'市场':market,'策略':name,'最大收益误差':error,'通过':error<=1e-12})
            trades=execution_trades(d,r.execution_position)
            if len(trades):ledgers.append(trades.assign(市场=market,策略=name))
            prior=r.execution_position.shift(1).fillna(0)
            entries=(r.execution_position!=0)&(r.execution_position!=prior)
            exits=(prior!=0)&(r.execution_position!=prior)
            reversals=(r.execution_position*prior<0)
            for period,(start,end) in PERIODS.items():
                row=metrics(r.ret.loc[start:end],r.position.loc[start:end],r.turnover.loc[start:end])
                identifiers={'市场':market,'分期':period,'策略':name,
                             '策略类别':manifest['strategies'][name]['category'],
                             '配置标识':manifest['strategies'][name]['config_sha256']}
                row.update(identifiers)
                row.update({'单边成本':SPEC.cost,'假设空头年费':SPEC.short_cost})
                row.update({'建仓次数':int(entries.loc[start:end].sum()),
                            '平仓次数':int(exits.loc[start:end].sum()),
                            '反手次数':int(reversals.loc[start:end].sum())})
                bench=results['买入持有'].ret.loc[start:end]
                row['年化差']=row['年化收益']-annualized_return(bench)
                eq_rel=(1+r.ret.loc[start:end]).cumprod()/(1+bench).cumprod()
                row['相对净值年化']=float(eq_rel.iloc[-1]**(SPEC.annualization/len(bench))-1)
                all_metrics.append(row)
                gross=r.position*d.ret.fillna(0)
                gross_ids={k:v for k,v in identifiers.items() if k!='配置标识'}
                gross_ids.update({'参考主规则标识':identifiers['配置标识'],'单边成本':0.,'假设空头年费':0.})
                gross_metrics.append({**gross_ids,**metrics(gross.loc[start:end],r.position.loc[start:end],r.turnover.loc[start:end])})
            for year,g in r.ret.groupby(r.ret.index.year):
                yearly.append({'市场':market,'年份':year,'策略':name,
                               '累计收益':(1+g).prod()-1,'天数':len(g),'期末':str(g.index[-1].date())})
        # 同一市场所有规则都已具备信号并完成成交后的共同起点；仅重新统计，不重置账户。
        ready=max(f[col].first_valid_index() for col in ['beta','zscore','modified','right_skewed','right_skewed_16_300'])
        ready=d.index[min(d.index.get_loc(ready)+SPEC.lag+1,len(d)-1)]
        ready=max(ready,pd.Timestamp(PERIODS['原报告期'][0]))
        for name,r in results.items():
            common_window.append({'市场':market,'策略':name,
                                  **metrics(r.ret.loc[ready:'2017-04-28'],r.position.loc[ready:'2017-04-28'],r.turnover.loc[ready:'2017-04-28'])})
        destination=OUT/symbol
        destination.mkdir(exist_ok=True)
        for filename,field in [('daily_returns','ret'),('daily_positions','position'),
                               ('execution_positions','execution_position'),('daily_turnover','turnover')]:
            pd.DataFrame({k:getattr(r,field) for k,r in results.items()}).to_csv(destination/f'{filename}.csv',encoding='utf-8-sig',index_label='date')
        pd.DataFrame(sig).to_csv(destination/'signals.csv',encoding='utf-8-sig',index_label='date')
        f.to_csv(destination/'indicators.csv',encoding='utf-8-sig',index_label='date')
        if symbol=='sh000300':
            main_data,main_signals,main_f=d,sig,f
            returns=pd.DataFrame({k:r.ret for k,r in results.items()})
    m=pd.DataFrame(all_metrics)
    p=pd.concat(prediction,ignore_index=True)
    b=pd.concat(fixed_bins,ignore_index=True)
    # 每个分期统一对三市场×五指标×四期限×两标签起点的120个主检验做Holm校正。
    p['HAC_p_Holm']=np.nan
    for period in PERIODS:
        ix=p.index[(p['分期']==period)&(p['HAC阶数']==20)]
        p.loc[ix,'HAC_p_Holm']=multipletests(p.loc[ix,'HAC_p探索'],method='holm')[1]
    bin_corr=[]
    for period,(start,end) in PERIODS.items():
        for width in [.1,.2,.3,.5]:
            pair=sample_for_prediction(main_f.right_skewed,main_data.close,start,end,10)
            for min_n in [1,20]:
                g=binned_stats(main_f.right_skewed.loc[start:end],main_data.close.loc[:end],width=width,min_n=min_n)
                bin_corr.append({'市场':'沪深300','分期':period,'箱宽':width,'最少箱样本':min_n,
                                 '箱数':len(g),'保留样本数':g.n.sum(),
                                 '分箱相关':g.center.corr(g.exp_ret),'逐日相关':pair.x.corr(pair.y)})
    sensitivity=[]
    for cost,lag in product([0,.001,.002,.003],[1,2]):
        for name in ALL_STRATEGIES:
            r=backtest(main_data,main_signals[name],cost=cost,lag=lag,execution=SPEC.execution)
            for period in ['原报告期','发表后全部','近期检验']:
                start,end=PERIODS[period]
                sensitivity.append({'策略':name,'分期':period,'单边成本':cost,'执行延迟':lag,
                                    **metrics(r.ret.loc[start:end],r.position.loc[start:end],r.turnover.loc[start:end])})
    carry_sensitivity=[]
    for annual_fee in [0,.02,.05,.10]:
        for name in SHORT_STRATEGIES:
            r=backtest(main_data,main_signals[name],cost=SPEC.cost,lag=SPEC.lag,
                       execution=SPEC.execution,short_cost=annual_fee)
            for period in ['原报告期','发表后全部','近期检验']:
                start,end=PERIODS[period]
                carry_sensitivity.append({'策略':name,'分期':period,'假设空头年费':annual_fee,
                                          **metrics(r.ret.loc[start:end],r.position.loc[start:end],r.turnover.loc[start:end])})
    grid=[]
    # 参数或实施约定一次只改变一项；全部报告，不能用于回填最优主配置。
    choices=[('N','n',[16,18,20]),('M','m',[300,600,900]),('S','buy',[.5,.7,.9]),
             ('排除当期β','exclude_current',[False,True]),('最少有效斜率','min_periods',[2,10,600]),
             ('成交量窗口滞后','volume_lag',[0,1])]
    for label,field,values in choices:
        for value in values:
            changed={field:value}
            if field=='buy':changed['sell']=-value
            # 基础右偏参数不在这项共同参数试验内。
            variant=replace(SPEC,**changed)
            variant_sig,_=signals(main_data,variant)
            names=['量过滤·右偏','多空·量过滤·右偏'] if field=='volume_lag' else [
                   '价格过滤·右偏','多空·价格过滤·右偏']
            for name in names:
                r=run_strategy(main_data,variant_sig[name],variant)
                for period in ['原报告期','发表后全部','近期检验']:
                    start,end=PERIODS[period]
                    grid.append({'策略':name,'分期':period,'变动参数':label,'值':value,
                                 **metrics(r.ret.loc[start:end],r.position.loc[start:end],r.turnover.loc[start:end])})
    boot=[]
    comparisons=[(name,'买入持有') for name in MAIN if name!='买入持有']+[
        ('多空·价格过滤·右偏','价格过滤·右偏'),('多空·量过滤·右偏','量过滤·右偏')]
    for strategy,baseline in comparisons:
        for block in [30,60,120]:
            q=bootstrap_mean_difference(returns.loc['2017-05-02':,strategy],returns.loc['2017-05-02':,baseline],block)
            boot.append({'策略':strategy,'比较对象':baseline,**q})
    frames={'metrics':m,'metrics_gross':pd.DataFrame(gross_metrics),'predictive':p,'fixed_bins':b,
            'nonoverlap':pd.concat(phases,ignore_index=True),'distribution':pd.DataFrame(distributions),
            'yearly_ic':pd.DataFrame(annual_ic),'binned_correlation':pd.DataFrame(bin_corr),
            'cost_execution':pd.DataFrame(sensitivity),'short_carry_sensitivity':pd.DataFrame(carry_sensitivity),
            'parameter_sensitivity':pd.DataFrame(grid),'bootstrap':pd.DataFrame(boot),
            'consistency_checks':pd.DataFrame(consistency),'account_checks':pd.DataFrame(accounts),
            'data_quality':pd.DataFrame(quality),'trades':pd.concat(ledgers,ignore_index=True),
            'common_start':pd.DataFrame(common_window),'yearly':pd.DataFrame(yearly)}
    assert m.groupby('策略')['配置标识'].nunique().eq(1).all(), '跨时期或跨市场配置发生改变'
    (OUT/'strategy_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,frame in frames.items():frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
              'statsmodels':__import__('statsmodels').__version__,'scipy':__import__('scipy').__version__}
    meta={'study_as_of':'2026-10-22','calculated_at_utc':datetime.now(timezone.utc).isoformat(),'versions':versions,
          'source_manifest':json.loads((ROOT/'data/extended/manifest.json').read_text()),
          'source_sha256':{s:hashlib.sha256((ROOT/'data/extended'/f'{s}.csv').read_bytes()).hexdigest()
                           for s in ['sh000300','sh000016','sh000905']},
          'execution':SPEC.execution,'cost_one_way':SPEC.cost,'annualization':SPEC.annualization,
          'bootstrap_seed':20261022,'periods':PERIODS,'strategy_manifest':manifest,
          'protocol_sha256':hashlib.sha256((ROOT/'docs/research_protocol.md').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
          'document_sha256':{path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                             for path in sorted((ROOT/'docs').glob('*.md'))},
          'code_sha256':{path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                         for path in [Path(__file__),*sorted((ROOT/'rsrs').glob('*.py'))]}}
    (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(OUT,frames)
    if not args.no_plots:
        plot_results(m,p[p['市场']=='沪深300'],b[b['市场']=='沪深300'],returns)
    print(m[(m['市场']=='沪深300')&(m['策略'].isin(MAIN))][
        ['分期','策略','年化收益','夏普','最大回撤','平均净仓位','平均绝对仓位']].to_string(index=False))
    print(f'一致性 {len(consistency)} 组，独立核账 {len(accounts)} 组通过。输出：{OUT}')


if __name__=='__main__':main()

"""原研报逐节复现的统计口径；不以原文数值匹配程度选择参数。"""
from dataclasses import replace
import numpy as np
import pandas as pd
from .backtest import annualized_return, sharpe, max_drawdown, trade_table
from .research_spec import SPEC

PAPER_SPEC = replace(SPEC, cost=0., end='2017-04-28')
START = '2005-03-01'  # 原文只披露月份；可用数据不足时保留实际起点。
MAIN_END = '2017-04-28'
CROSS_END = '2017-03-31'
N_GRID = tuple(range(14, 26))  # 原图11图例及图12的实际散点，非横轴刻度10..26。
M_GRID = tuple(range(450, 801, 50))
COSTS = (0., .002, .003)  # 单边；对应无成本、双边0.4%、双边0.6%。
STAT_ROWS = ['年化收益率','夏普比率','最大回撤','持仓总天数','交易次数','平均持仓天数',
             '获利天数','亏损天数','胜率（按天）','平均盈利率（按天）','平均亏损率（按天）',
             '平均盈亏比（按天）','盈利次数','亏损次数','单次最大盈利','单次最大亏损',
             '胜率（按次）','平均盈利率（按次）','平均亏损率（按次）','平均盈亏比（按次）']


def period_stats(result, start=START, end=MAIN_END):
    """主表无成本；交易统计按区间内连续同向损益持仓段，含未平仓估值段。

    左边界已有仓位计一个结转段；这是统计分段而非新增建仓。另存完整真实成交账本。
    零收益日和零收益段计入胜率分母，单列以免天数/次数隐式丢失。
    """
    r=result.ret.loc[start:end]
    p=result.position.loc[r.index]
    gross=result.benchmark_ret.loc[r.index].fillna(0)
    held=(p*gross).loc[p.ne(0)]
    trades=trade_table(p,gross)
    t=trades['ret'] if len(trades) else pd.Series(dtype=float)
    wins,losses=held[held>0],held[held<0]
    tw,tl=t[t>0],t[t<0]
    ratio=lambda a,b: a.mean()/abs(b.mean()) if len(a) and len(b) else np.nan
    out={'年化收益率':annualized_return(r),'夏普比率':sharpe(r),
         '最大回撤':max_drawdown((1+r).cumprod()),'持仓总天数':len(held),'交易次数':len(t),
         '平均持仓天数':len(held)/len(t) if len(t) else np.nan,
         '获利天数':len(wins),'亏损天数':len(losses),'胜率（按天）':len(wins)/len(held) if len(held) else np.nan,
         '平均盈利率（按天）':wins.mean(),'平均亏损率（按天）':losses.mean(),
         '平均盈亏比（按天）':ratio(wins,losses),'盈利次数':len(tw),'亏损次数':len(tl),
         '单次最大盈利':t.max(),'单次最大亏损':t.min(),'胜率（按次）':len(tw)/len(t) if len(t) else np.nan,
         '平均盈利率（按次）':tw.mean(),'平均亏损率（按次）':tl.mean(),'平均盈亏比（按次）':ratio(tw,tl),
         '零收益持仓天数':int(held.eq(0).sum()),'零收益交易段数':int(t.eq(0).sum()),
         '期末净值':float((1+r).prod()),'累计收益':float((1+r).prod()-1),
         '实际起点':str(r.index[0].date()),'实际终点':str(r.index[-1].date()),
         '左端结转持仓':bool(p.iloc[0]!=0),'期末成交仓位':float(result.execution_position.loc[r.index[-1]])}
    return out


def distribution(series):
    x=series.dropna()
    return {'均值':x.mean(),'标准差':x.std(ddof=1),'偏度':x.skew(),'超额峰度':x.kurt(),'样本数':len(x)}


def grouped_prediction(indicator, close, bins=50, start=START, end=MAIN_END):
    """按指标水平汇总未来10日收益；仅使用标签终点不超出区间的完整配对。

    50个等宽箱是参照notebook作出的明确实施选择，不声称原文披露了50箱。
    分箱边界用原报告期全部可用指标确定；箱横坐标使用实际区间中点。
    不合并或删除小样本箱，报告样本数。空箱保留但不参加相关计算。
    """
    c=close.loc[:end]
    x=indicator.loc[start:end].dropna()
    if x.empty or x.nunique()<2:
        raise ValueError('分箱需要至少两个不同指标值')
    _,edges=pd.cut(x,bins,retbins=True)
    pairs=pd.DataFrame({'x':x,'y':c.shift(-10)/c-1}).loc[start:end].dropna()
    labels=pd.cut(pairs.x,edges,labels=False,include_lowest=True)
    rows=[]
    for k in range(bins):
        q=pairs.loc[labels==k]
        rows.append({'箱':k+1,'左端':edges[k],'右端':edges[k+1],
                     '箱中点':(edges[k]+edges[k+1])/2,'指标均值':q.x.mean(),'样本数':len(q),
                     '未来10日收益':q.y.mean(),'上涨概率':float((q.y>0).mean()) if len(q) else np.nan})
    return pd.DataFrame(rows)


def grouped_correlations(groups):
    rows=[]
    valid=groups.loc[groups['样本数']>0]
    for side in ['整体','左侧','右侧']:
        q=valid if side=='整体' else valid.loc[valid['箱中点']<0] if side=='左侧' else valid.loc[valid['箱中点']>0]
        rows.append({'侧别':side,'非空箱数':len(q),'样本数':int(q['样本数'].sum()),
                     '期望收益相关':q['箱中点'].corr(q['未来10日收益']) if len(q)>2 else np.nan,
                     '上涨概率相关':q['箱中点'].corr(q['上涨概率']) if len(q)>2 else np.nan})
    return pd.DataFrame(rows)

"""独立逐日核账、实际成交记录与原始行情检查。"""
import numpy as np
import pandas as pd


def execution_trades(d, executed):
    rows=[]
    state,entry=0.,None
    def record(end,closed):
        daily=d.ret.iloc[entry+1:end+1].fillna(0)
        rows.append({'建仓日':d.index[entry],'平仓日':d.index[end] if closed else pd.NaT,
                     '估值日':d.index[end],'方向':'多头' if state>0 else '空头',
                     '已平仓':closed,'持仓收益天数':len(daily),
                     '毛收益':float((1+state*daily).prod()-1)})
    for i,target in enumerate(executed):
        if target!=state:
            if state: record(i,True)
            state,entry=float(target),i
    if state: record(len(d)-1,False)
    return pd.DataFrame(rows)


def independent_account_check(d, signal, result, cost=.001, lag=1, short_cost=0):
    held=0.
    expected=[]
    for i,r in enumerate(d.ret.fillna(0)):
        target=float(signal.iloc[i-lag]) if i>=lag else 0.
        expected.append(held*r-cost*abs(target-held)-max(-held,0)*short_cost/252)
        held=target
    error=float(np.max(np.abs(np.asarray(expected)-result.ret.to_numpy())))
    if error>1e-12: raise AssertionError(f'独立核账失败: {error}')
    return error


def raw_data_quality(path, market):
    d=pd.read_csv(path)
    price=d[['open','high','low','close']].apply(pd.to_numeric,errors='coerce')
    bad=(price.high+.01<price[['open','close','low']].max(axis=1)) | (price.low-.01>price[['open','close','high']].min(axis=1))
    return {'市场':market,'行数':len(d),'起点':d.date.iloc[0],'终点':d.date.iloc[-1],
            '重复日期':int(d.date.duplicated().sum()),'价格缺失行':int(price.isna().any(axis=1).sum()),
            '非正价格行':int((price<=0).any(axis=1).sum()),'OHLC异常行':int(bad.sum()),
            'OHLC异常日期':';'.join(d.loc[bad,'date'].astype(str)),
            '负成交量行':int((pd.to_numeric(d.volume,errors='coerce')<0).sum())}

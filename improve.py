"""评价RSRS在2017—2026延长样本中的表现并检验改进方案。

运行：python improve.py
输出：report/improvement/
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import signals as report_signals, metrics, bootstrap_mean_difference
from rsrs.backtest import backtest, trade_table
from rsrs.data import load_index
from rsrs.improvement import (
    adaptive_trend_rsrs, volatility_damped_signal,
)
from rsrs.indicators import rsrs_family
from rsrs.research_spec import SPEC

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "report" / "improvement"
MARKETS = {"sh000300": "沪深300", "sh000016": "上证50", "sh000905": "中证500"}
PERIODS = {
    "原报告期": ("2005-04-08", "2017-04-28"),
    "延长样本全期": ("2017-05-02", "2026-10-21"),
    "延长样本前半段": ("2017-05-02", "2021-12-31"),
    "延长样本后半段": ("2022-01-01", "2026-10-21"),
}
STRATEGIES = ["买入持有", "原始右偏", "原始价格过滤右偏", "波动钝化RSRS",
              "纯趋势MA250", "状态自适应RSRS"]


def pct(x):
    return "—" if pd.isna(x) else f"{x:.2%}"


def num(x, n=3):
    return "—" if pd.isna(x) else f"{x:.{n}f}"


def markdown(frame, percent=(), decimals=()):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"]*len(cols)) + " |"]
    for _, row in frame.iterrows():
        values=[]
        for col in cols:
            value=row[col]
            if col in percent: values.append(pct(value))
            elif col in decimals: values.append(num(value))
            elif pd.isna(value): values.append("—")
            else: values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_signals(data):
    base, family = report_signals(data, SPEC)
    damped_signal, damped = volatility_damped_signal(data)
    improved = adaptive_trend_rsrs(data, family["right_skewed"])
    trend = (data.close > data.close.rolling(250, min_periods=250).mean()).astype(float)
    return {
        "买入持有": pd.Series(1., index=data.index),
        "原始右偏": base["右偏(18,600)"],
        "原始价格过滤右偏": base["价格过滤·右偏"],
        "波动钝化RSRS": damped_signal,
        "纯趋势MA250": trend,
        "状态自适应RSRS": improved.position,
    }, family, damped, improved


def period_metrics(result, start, end):
    idx = result.ret.loc[start:end].index
    return metrics(result.ret.loc[idx], result.position.loc[idx], result.turnover.loc[idx])


def prediction_pair(indicator, close, start, end):
    dates=pd.Series(close.index,index=close.index)
    # t日信号与次日收盘执行后的10日可获得区间对齐。
    future=close.shift(-11)/close.shift(-1)-1
    pair=pd.DataFrame({"指标":indicator,"未来10日收益":future,"标签终点":dates.shift(-11)})
    return pair.loc[start:end].loc[lambda x:x["标签终点"]<=pd.Timestamp(end)].dropna()


def diagnostics(data, family, results):
    rows=[]
    for period,(start,end) in PERIODS.items():
        pair=prediction_pair(family.right_skewed,data.close,start,end)
        rows.append({"分期":period,"样本数":len(pair),
                     "Pearson":pair["指标"].corr(pair["未来10日收益"]),
                     "Spearman":pair["指标"].corr(pair["未来10日收益"],method="spearman")})
    trade_rows=[]
    active_rows=[]
    for strategy,result in results.items():
        for period,(start,end) in PERIODS.items():
            idx=result.ret.loc[start:end].index
            pos=result.position.loc[idx]
            market=result.benchmark_ret.loc[idx]
            trades=trade_table(pos,market)
            trade_rows.append({"策略":strategy,"分期":period,"交易段数":len(trades),
                               "胜率":(trades.ret>0).mean() if len(trades) else np.nan,
                               "平均每段收益":trades.ret.mean() if len(trades) else np.nan,
                               "平均持有日":trades.days.mean() if len(trades) else np.nan})
            cash=1-pos
            active_rows.append({"策略":strategy,"分期":period,
                                "错过上涨日收益之和":float((cash*market.clip(lower=0)).sum()),
                                "规避下跌日损失之和":float((cash*(-market.clip(upper=0))).sum()),
                                "成本之和":float((result.turnover.loc[idx]*SPEC.cost).sum()),
                                "平均仓位":float(pos.mean())})
    return pd.DataFrame(rows),pd.DataFrame(trade_rows),pd.DataFrame(active_rows)


def sensitivity(data, indicator):
    rows=[]
    for rank_window in [600,756,1000]:
        for trend_window in [200,250,300]:
            for high,low in [(.7,.3),(.8,.2),(.9,.1)]:
                detail=adaptive_trend_rsrs(
                    data, indicator, rank_window=rank_window,
                    rank_min_periods=min(252,rank_window),
                    entry_quantile=high, exit_quantile=low,
                    trend_window=trend_window, trend_slope_gap=20,
                )
                result=backtest(data,detail.position,cost=SPEC.cost,lag=SPEC.lag,execution=SPEC.execution)
                for period in PERIODS:
                    start,end=PERIODS[period]
                    rows.append({"滚动窗口":rank_window,"趋势窗口":trend_window,
                                 "进入分位":high,"退出分位":low,"分期":period,
                                 **period_metrics(result,start,end)})
    return pd.DataFrame(rows)


def yearly_results(data, results):
    rows=[]
    for year in range(2017,2027):
        for name,result in results.items():
            r=result.ret[result.ret.index.year==year]
            if len(r): rows.append({"年份":year,"策略":name,"收益":float((1+r).prod()-1)})
    return pd.DataFrame(rows)


def plots(out, data, results, yearly, diagnosis, sensitivity_frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    for font in ["Microsoft YaHei","SimHei","Noto Sans CJK SC","DejaVu Sans"]:
        if font in fonts: plt.rcParams["font.sans-serif"]=[font]; break
    plt.rcParams.update({"axes.unicode_minus":False,"figure.dpi":130,"savefig.dpi":170,
                         "axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    colors={"买入持有":"#596773","原始右偏":"#B35A3C","原始价格过滤右偏":"#DB8D34",
            "波动钝化RSRS":"#7A6AA6","纯趋势MA250":"#3F8A8A","状态自适应RSRS":"#166B45"}
    start="2017-05-02"
    fig,ax=plt.subplots(2,1,figsize=(11,7.5),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    for name in STRATEGIES:
        r=results[name].ret.loc[start:]
        eq=(1+r).cumprod()
        ax[0].plot(eq.index,eq,label=name,color=colors[name],lw=2 if name=="状态自适应RSRS" else 1.2)
        ax[1].plot(eq.index,eq/eq.cummax().clip(lower=1)-1,color=colors[name],lw=1.5 if name=="状态自适应RSRS" else 1)
    ax[0].set_title("沪深300｜2017年后净值（次日收盘、单边成本0.1%）",loc="left",pad=12)
    ax[0].set_ylabel("净值");ax[1].set_ylabel("回撤");ax[1].yaxis.set_major_formatter(lambda x,p:f"{x:.0%}")
    ax[0].legend(ncol=3,frameon=False);fig.tight_layout();fig.savefig(out/"01_post2017_nav.png",bbox_inches="tight");plt.close(fig)

    q=yearly[yearly.策略.isin(["买入持有","原始价格过滤右偏","波动钝化RSRS","状态自适应RSRS"])].pivot(index="年份",columns="策略",values="收益")
    ax=q.plot(kind="bar",figsize=(11,5),color=[colors.get(c) for c in q.columns],width=.8)
    ax.axhline(0,color="#666",lw=.8);ax.set_title("年度收益对比",loc="left",pad=12);ax.set_ylabel("收益")
    ax.yaxis.set_major_formatter(lambda x,p:f"{x:.0%}");ax.legend(frameon=False,ncol=2);ax.figure.tight_layout();ax.figure.savefig(out/"02_yearly_returns.png",bbox_inches="tight");plt.close(ax.figure)

    fig,ax=plt.subplots(figsize=(9,4.6));x=np.arange(len(diagnosis))
    ax.bar(x-.18,diagnosis.Pearson,.36,label="Pearson",color="#337F9B")
    ax.bar(x+.18,diagnosis.Spearman,.36,label="Spearman",color="#9A6D3B")
    ax.axhline(0,color="#666",lw=.8);ax.set_xticks(x,diagnosis.分期,rotation=15);ax.set_ylabel("与未来10日收益相关")
    ax.set_title("右偏RSRS方向信息在延长样本中衰减",loc="left",pad=12);ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/"03_predictive_decay.png",bbox_inches="tight");plt.close(fig)

    q=sensitivity_frame[sensitivity_frame.分期=="延长样本全期"].copy()
    chosen=(q.滚动窗口.eq(756)&q.趋势窗口.eq(250)&q.进入分位.eq(.8))
    fig,ax=plt.subplots(figsize=(8.5,5));ax.scatter(q["最大回撤"],q["年化收益"],c="#799A8A",alpha=.65,label="相邻27组")
    ax.scatter(q.loc[chosen,"最大回撤"],q.loc[chosen,"年化收益"],c="#A33B34",s=90,label="主方案")
    ax.axhline(0,color="#777",lw=.8);ax.set_xlabel("最大回撤");ax.set_ylabel("年化收益");ax.xaxis.set_major_formatter(lambda x,p:f"{x:.0%}");ax.yaxis.set_major_formatter(lambda x,p:f"{x:.0%}")
    ax.set_title("2017—2026延长样本：相邻参数敏感性",loc="left",pad=12);ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/"04_sensitivity.png",bbox_inches="tight");plt.close(fig)


def write_report(out, performance, diagnosis, trades, active, sensitivity_frame, cross, bootstrap, benchmark):
    main=performance[(performance.市场=="沪深300") & performance.分期.isin(PERIODS)].copy()
    table=main[["分期","策略","累计收益","年化收益","夏普","最大回撤","平均绝对仓位","年均单边换手"]].copy()
    table["分期"]=pd.Categorical(table["分期"],categories=list(PERIODS),ordered=True)
    table["策略"]=pd.Categorical(table["策略"],categories=STRATEGIES,ordered=True)
    table=table.sort_values(["分期","策略"])
    extension=main[main.分期=="延长样本全期"].set_index("策略")
    latter=main[main.分期=="延长样本后半段"].set_index("策略")
    sens=sensitivity_frame[sensitivity_frame.分期=="延长样本全期"]
    stable=pd.DataFrame([{
        "相邻组合数":str(len(sens)),"正年化比例":(sens["年化收益"]>0).mean(),
        "年化收益中位数":sens["年化收益"].median(),"最大回撤中位数":sens["最大回撤"].median(),
        "最差最大回撤":sens["最大回撤"].min(),
    }])
    trade=trades[(trades.策略.isin(["原始右偏","波动钝化RSRS","状态自适应RSRS"])) &
                 (trades.分期.isin(PERIODS))]
    act=active[(active.策略.isin(["原始右偏","波动钝化RSRS","状态自适应RSRS"])) &
               (active.分期.isin(["延长样本全期","延长样本后半段"]))]
    cross_table=cross[["市场","策略","累计收益","年化收益","夏普","最大回撤","平均绝对仓位"]].copy()
    report=f"""# RSRS策略评价、失效诊断与改进

本报告在完成原研报复现后，评价原规则在**2017-05-02至2026-10-21**延长样本中的有效性，诊断表现变化的原因，并检验可解释的改进。原报告期仅用于建立参照；全部核心评价、参数敏感性、跨市场检验和统计检验都以完整的2017—2026延长样本为主。所有策略统一采用t日收盘形成信号、t+1日收盘执行、单边成本0.1%、现金收益0。

**证据边界：**改进规则是在观察到延长样本表现之后提出的，因此2017—2026是后验研究区间，并非盲样本外。计算不使用未来数据，策略条件在整个评价期保持不变；这证明规则在历史数据上的表现，不能直接等同于未来可实现收益。

## 1. 结论先行

1. **沪深300基准已按完整评价期重新核对。**首个收益观察日为{benchmark.loc[0,'首个收益观察日']}，以此前交易日收盘{benchmark.loc[0,'起算收盘']:.3f}为基数，期末收盘为{benchmark.loc[0,'期末收盘']:.3f}；累计收益{pct(benchmark.loc[0,'指数累计收益'])}、按252个交易日年化{pct(benchmark.loc[0,'指数年化收益'])}。该数值与回测表的连续买入持有账户一致，并非0.8%。
2. **原策略在延长样本中没有稳定跑赢市场。**买入持有年化{pct(extension.loc['买入持有','年化收益'])}；原始右偏RSRS年化{pct(extension.loc['原始右偏','年化收益'])}，但最大回撤{pct(extension.loc['原始右偏','最大回撤'])}；原始价格过滤右偏年化{pct(extension.loc['原始价格过滤右偏','年化收益'])}，低于基准。
3. **失效首先表现为指标方向信息衰减。**右偏RSRS与次日执行后未来10日收益的Pearson相关系数从原报告期的{diagnosis.loc[diagnosis.分期=='原报告期','Pearson'].iloc[0]:.3f}下降到延长样本全期的{diagnosis.loc[diagnosis.分期=='延长样本全期','Pearson'].iloc[0]:.3f}。当排序关系接近零时，调整固定阈值只能改变交易频率，无法恢复预测方向。
4. **状态自适应RSRS的优势主要是控制风险。**完整延长样本年化{pct(extension.loc['状态自适应RSRS','年化收益'])}、夏普{extension.loc['状态自适应RSRS','夏普']:.3f}、最大回撤{pct(extension.loc['状态自适应RSRS','最大回撤'])}；同期买入持有最大回撤{pct(extension.loc['买入持有','最大回撤'])}。收益提升有限，但回撤明显收窄。
5. **结果不能被表述为已经找到稳定的新因子。**改进策略完整延长样本平均仓位仅{pct(extension.loc['状态自适应RSRS','平均绝对仓位'])}；低回撤部分来自较长时间空仓。参数邻域和跨市场结果也要求谨慎解释。

![2017—2026净值与回撤](01_post2017_nav.png)

## 2. 基准收益核对

{markdown(benchmark,['指数累计收益','指数年化收益','回测买入持有累计收益','回测买入持有年化收益'],[])}

评价期首日的日收益来自上一个交易日收盘，因此核对表用2017-04-28收盘作为起算基数，与策略收益序列完全对齐。年化统一采用252个交易日口径。买入持有在评价期开始前已经处于持仓状态，期内没有换手成本；策略也同样保留此前历史形成的状态。这样既保留指标预热，也避免各策略在分段边界被人为重置。

## 3. 为什么原策略在延长样本中表现变差

### 3.1 指标的方向信息衰减

![预测关系衰减](03_predictive_decay.png)

{markdown(diagnosis,[],['Pearson','Spearman'])}

RSRS依赖“指标越高，未来收益越高”的排序关系。该关系在延长样本中接近零后，改阈值只能改变交易频率，无法恢复方向性。波动钝化可以减少弱信号触发，但不会创造新的预测信息。

### 3.2 固定绝对阈值不适应指标分布漂移

原规则长期固定±0.7。当波动、R²和beta分布改变时，±0.7所代表的历史极端程度也改变。结果可能是某些阶段长期不触发，另一些阶段频繁触发。改进方案因此使用只含过去数据的滚动经验分位数。

### 3.3 高低价结构不能单独代表市场趋势

同一种RSRS形态可出现在上涨、反弹、下跌中继和震荡。交易账本显示，延长样本中的部分持仓发生在长期趋势走弱阶段；仅过滤开仓、却不在长期趋势破坏时退出，会放大这种错误。

{markdown(trade,['胜率','平均每段收益'],['平均持有日'])}

### 3.4 空仓的收益与代价不对称

{markdown(act,['错过上涨日收益之和','规避下跌日损失之和','成本之和','平均仓位'])}

这里的日收益加总是归因量，不等于复合收益。它用于判断空仓规避的下跌，是否足以补偿错过的上涨与交易成本。

## 4. 改进方案：状态自适应RSRS

改进保留18/600右偏RSRS本体，不使用未来收益重估beta或拟合预测模型。规则为：

- **尺度校准：**把当日右偏RSRS转换为过去756个交易日约三年的经验分位数，最少252日才启用。
- **开仓：**指标处于前20%，收盘价高于MA250，并且MA250高于20日前。
- **平仓：**指标落入后20%，或收盘价跌破MA250；二者任一成立即退出。
- **执行：**条件都由当日及过去数据计算，次日收盘成交。参数在2017—2026全期及三个指数上保持不变。

滚动分位数用于处理指标量纲漂移，长期均线用于识别趋势状态，非对称退出用于减少长期下行中的错误持仓。波动钝化RSRS和纯MA250分别作为对照，以判断改善来自信号压缩、趋势过滤还是二者结合。

## 5. 统一口径的分期结果

{markdown(table,['累计收益','年化收益','最大回撤','平均绝对仓位'],['夏普','年均单边换手'])}

![年度收益](02_yearly_returns.png)

完整延长样本是唯一主要评价期；前后半段只用于解释路径，不改变策略条件。状态自适应RSRS的年化收益略高于买入持有，同时显著降低最大回撤；与纯MA250相比，最大回撤也更小。但低仓位是收益稳定的重要来源，不能把低回撤全部归因于RSRS预测能力。延长样本后半段，改进策略年化{pct(latter.loc['状态自适应RSRS','年化收益'])}，同期买入持有为{pct(latter.loc['买入持有','年化收益'])}。

## 6. 参数邻域与跨市场

主方案采用三年分位窗口、MA250和80/20分位。为披露参数选择风险，完整报告3×3×3=27个相邻组合；每组都在2017-05-02至2026-10-21上评价，没有把任何较晚年份另设为样本起点。

{markdown(stable,['正年化比例','年化收益中位数','最大回撤中位数','最差最大回撤'])}

![参数敏感性](04_sensitivity.png)

{markdown(cross_table,['累计收益','年化收益','最大回撤','平均绝对仓位'],['夏普'])}

跨市场结果并不一致，因此该方案不能直接概括为适用于全部宽基指数的通用规则。跨市场表全部使用2017-05-02至2026-10-21，并且不针对不同指数重新调参。

## 7. 统计不确定性

{markdown(bootstrap,['日均超额年化','CI下','CI上'],[])}

区间来自成对循环移动块bootstrap，比较改进策略与原始价格过滤右偏的日均收益差，报告30/60/120日块长。它不是CAGR置信区间；区间若包含零，表示当前样本不能把收益差与零显著区分。最大回撤的改善是历史路径事实，也不等于未来保证。

## 8. 评价与下一步

本研究说明原策略表现变化不只是成本造成，而与预测关系和市场状态变化有关；同时提出一套固定、无未来函数、在完整延长样本中改善回撤的规则。它尚未达到“可直接实盘”或“已通过独立样本外验证”的证据标准。

下一步应加入ETF可交易价格、股息、滑点和现金收益，并在2026年之后继续前瞻记录。若要提高收益，应优先研究仓位分级或跨资产配置，而不是继续在同一段历史上搜索更多参数。

## 复现文件

- `benchmark_reconciliation.csv`：沪深300首尾价格收益与回测买入持有账户的逐项核对。
- `performance.csv`：统一分期下的全部策略绩效。
- `diagnosis_ic.csv`、`trade_diagnostics.csv`、`active_return_attribution.csv`：失效诊断。
- `sensitivity.csv`：27组参数邻域完整结果。
- `cross_market.csv`：同配置跨市场迁移。
- `bootstrap.csv`：收益差的块bootstrap。
- `signals.csv`：沪深300逐日指标、过滤条件和仓位。
- `metadata.json`：数据、代码与配置哈希。
"""
    (out/"report.md").write_text(report,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data"/"extended")
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args();out=args.outdir;out.mkdir(parents=True,exist_ok=True)
    performance=[];cross=[];main_objects=None
    for symbol,market in MARKETS.items():
        data=load_index(symbol,data_dir=args.data_dir,invalid_ohlc=SPEC.invalid_ohlc).loc[:SPEC.end]
        signal_map,family,damped,improved=build_signals(data)
        results={name:backtest(data,signal,cost=SPEC.cost,lag=SPEC.lag,execution=SPEC.execution)
                 for name,signal in signal_map.items()}
        for name,result in results.items():
            for period,(start,end) in PERIODS.items():
                performance.append({"市场":market,"分期":period,"策略":name,
                                    **period_metrics(result,start,end)})
            q=period_metrics(result,*PERIODS["延长样本全期"])
            cross.append({"市场":market,"策略":name,**q})
        if symbol=="sh000300":
            diagnosis,trades,active=diagnostics(data,family,results)
            sensitivity_frame=sensitivity(data,family.right_skewed)
            yearly=yearly_results(data,results)
            signal_frame=pd.DataFrame({"close":data.close,"right_skewed":family.right_skewed,
                                       "damped_rsrs":damped.damped_rsrs,
                                       **{c:improved[c] for c in ["indicator_rank","trend_ma","price_above_trend","trend_up","entry_condition","exit_condition","position"]}})
            boot=[]
            for block in [30,60,120]:
                start,end=PERIODS["延长样本全期"]
                q=bootstrap_mean_difference(results["状态自适应RSRS"].ret.loc[start:end],
                                            results["原始价格过滤右偏"].ret.loc[start:end],block)
                boot.append({"比较":"状态自适应RSRS - 原始价格过滤右偏",**q})
            bootstrap=pd.DataFrame(boot)
            start,end=PERIODS["延长样本全期"]
            evaluation=data.loc[start:end]
            prior_close=data.loc[:start,"close"].iloc[-2]
            raw_cumulative=evaluation.close.iloc[-1]/prior_close-1
            raw_annualized=(1+raw_cumulative)**(252/len(evaluation))-1
            buyhold=period_metrics(results["买入持有"],start,end)
            benchmark=pd.DataFrame([{
                "首个收益观察日":evaluation.index[0].date().isoformat(),
                "起算基准日":data.loc[:start].index[-2].date().isoformat(),
                "结束日":evaluation.index[-1].date().isoformat(),
                "交易日数":len(evaluation),"起算收盘":prior_close,
                "期末收盘":evaluation.close.iloc[-1],"指数累计收益":raw_cumulative,
                "指数年化收益":raw_annualized,"回测买入持有累计收益":buyhold["累计收益"],
                "回测买入持有年化收益":buyhold["年化收益"],
            }])
            main_objects=(data,results,yearly,diagnosis,trades,active,sensitivity_frame,signal_frame,bootstrap,benchmark)
    performance=pd.DataFrame(performance);cross=pd.DataFrame(cross)
    data,results,yearly,diagnosis,trades,active,sensitivity_frame,signal_frame,bootstrap,benchmark=main_objects
    frames={"performance":performance,"diagnosis_ic":diagnosis,"trade_diagnostics":trades,
            "active_return_attribution":active,"sensitivity":sensitivity_frame,
            "cross_market":cross,"yearly":yearly,"signals":signal_frame,"bootstrap":bootstrap,
            "benchmark_reconciliation":benchmark}
    for name,frame in frames.items(): frame.to_csv(out/f"{name}.csv",index=False,encoding="utf-8-sig")
    if not args.no_plots: plots(out,data,results,yearly,diagnosis,sensitivity_frame)
    write_report(out,performance,diagnosis,trades,active,sensitivity_frame,cross,bootstrap,benchmark)
    config={"reference_period":["2005-04-08","2017-04-28"],
            "evaluation_period":["2017-05-02",SPEC.end],
            "execution":SPEC.execution,"cost_one_way":SPEC.cost,
            "volatility_damped":{"formula":"z*(R2**(2*volatility_rank))","n":18,"m":600,"thresholds":[.7,-.7]},
            "improved":{"base":"right_skewed(18,600)","rank_window":756,"rank_min_periods":252,
                        "entry_quantile":.8,"exit_quantile":.2,"trend_window":250,"trend_slope_gap":20}}
    meta={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"config":config,
          "source_sha256":{f"data/extended/{s}.csv":hashlib.sha256((args.data_dir/f"{s}.csv").read_bytes()).hexdigest() for s in MARKETS},
          "code_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                         for p in [Path(__file__),ROOT/"rsrs/improvement.py",ROOT/"tests/test_improvement.py"]}}
    (out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    generated={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in out.rglob("*") if p.is_file() and p.name!="generated_sha256.json"}
    (out/"generated_sha256.json").write_text(json.dumps(generated,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"完成：{out/'report.md'}")


if __name__=="__main__": main()

"""诊断RSRS发表后失效，并检验2019钝化基准与状态自适应改进。

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
    adaptive_trend_rsrs, fixed_2019_signal,
)
from rsrs.indicators import rsrs_family
from rsrs.research_spec import SPEC

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "report" / "improvement"
MARKETS = {"sh000300": "沪深300", "sh000016": "上证50", "sh000905": "中证500"}
PERIODS = {
    "开发期": ("2005-04-08", "2017-04-28"),
    "2019报告跟踪期": ("2017-05-02", "2019-11-13"),
    "2019报告发表后": ("2019-11-14", "2026-10-21"),
    "发表后全部": ("2017-05-02", "2026-10-21"),
    "近期": ("2022-01-01", "2026-10-21"),
}
STRATEGIES = ["买入持有", "原始右偏", "原始价格过滤右偏", "2019钝化公式",
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
    fixed_2019, damped = fixed_2019_signal(data)
    improved = adaptive_trend_rsrs(data, family["right_skewed"])
    trend = (data.close > data.close.rolling(250, min_periods=250).mean()).astype(float)
    return {
        "买入持有": pd.Series(1., index=data.index),
        "原始右偏": base["右偏(18,600)"],
        "原始价格过滤右偏": base["价格过滤·右偏"],
        "2019钝化公式": fixed_2019,
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
                for period in ["开发期","2019报告发表后","发表后全部","近期"]:
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
            "2019钝化公式":"#7A6AA6","纯趋势MA250":"#3F8A8A","状态自适应RSRS":"#166B45"}
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

    q=yearly[yearly.策略.isin(["买入持有","原始价格过滤右偏","2019钝化公式","状态自适应RSRS"])].pivot(index="年份",columns="策略",values="收益")
    ax=q.plot(kind="bar",figsize=(11,5),color=[colors.get(c) for c in q.columns],width=.8)
    ax.axhline(0,color="#666",lw=.8);ax.set_title("年度收益对比",loc="left",pad=12);ax.set_ylabel("收益")
    ax.yaxis.set_major_formatter(lambda x,p:f"{x:.0%}");ax.legend(frameon=False,ncol=2);ax.figure.tight_layout();ax.figure.savefig(out/"02_yearly_returns.png",bbox_inches="tight");plt.close(ax.figure)

    fig,ax=plt.subplots(figsize=(9,4.6));x=np.arange(len(diagnosis))
    ax.bar(x-.18,diagnosis.Pearson,.36,label="Pearson",color="#337F9B")
    ax.bar(x+.18,diagnosis.Spearman,.36,label="Spearman",color="#9A6D3B")
    ax.axhline(0,color="#666",lw=.8);ax.set_xticks(x,diagnosis.分期,rotation=15);ax.set_ylabel("与未来10日收益相关")
    ax.set_title("右偏RSRS方向信息在发表后衰减",loc="left",pad=12);ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/"03_predictive_decay.png",bbox_inches="tight");plt.close(fig)

    q=sensitivity_frame[sensitivity_frame.分期=="2019报告发表后"].copy()
    chosen=(q.滚动窗口.eq(756)&q.趋势窗口.eq(250)&q.进入分位.eq(.8))
    fig,ax=plt.subplots(figsize=(8.5,5));ax.scatter(q["最大回撤"],q["年化收益"],c="#799A8A",alpha=.65,label="相邻27组")
    ax.scatter(q.loc[chosen,"最大回撤"],q.loc[chosen,"年化收益"],c="#A33B34",s=90,label="主方案")
    ax.axhline(0,color="#777",lw=.8);ax.set_xlabel("最大回撤");ax.set_ylabel("年化收益");ax.xaxis.set_major_formatter(lambda x,p:f"{x:.0%}");ax.yaxis.set_major_formatter(lambda x,p:f"{x:.0%}")
    ax.set_title("2019报告发表后：相邻参数敏感性",loc="left",pad=12);ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/"04_sensitivity.png",bbox_inches="tight");plt.close(fig)


def write_report(out, performance, diagnosis, trades, active, sensitivity_frame, cross, bootstrap):
    main=performance[(performance.市场=="沪深300") & performance.分期.isin(PERIODS)].copy()
    table=main[["分期","策略","累计收益","年化收益","夏普","最大回撤","平均绝对仓位","年均单边换手"]]
    table["分期"]=pd.Categorical(table["分期"],categories=list(PERIODS),ordered=True)
    table["策略"]=pd.Categorical(table["策略"],categories=STRATEGIES,ordered=True)
    table=table.sort_values(["分期","策略"])
    post=main[main.分期=="2019报告发表后"].set_index("策略")
    recent=main[main.分期=="近期"].set_index("策略")
    sens=sensitivity_frame[sensitivity_frame.分期=="2019报告发表后"]
    stable=pd.DataFrame([{
        "相邻组合数":str(len(sens)),"正年化比例":(sens["年化收益"]>0).mean(),
        "年化收益中位数":sens["年化收益"].median(),"最大回撤中位数":sens["最大回撤"].median(),
        "最差最大回撤":sens["最大回撤"].min(),
    }])
    trade=trades[(trades.策略.isin(["原始右偏","2019钝化公式","状态自适应RSRS"])) &
                 (trades.分期.isin(["开发期","2019报告发表后","近期"]))]
    act=active[(active.策略.isin(["原始右偏","2019钝化公式","状态自适应RSRS"])) &
               (active.分期.isin(["2019报告发表后","近期"]))]
    bs=bootstrap.copy()
    cross_table=cross[["市场","策略","累计收益","年化收益","夏普","最大回撤","平均绝对仓位"]].copy()
    report=f"""# RSRS策略评价、失效诊断与后验改进

本报告承接原研报复现，但研究问题已经改变：不再检验“是否复现”，而是评价固定RSRS规则为什么在后续失效，并提出可解释的改进。所有策略仍采用t日收盘形成信号、t+1日收盘执行、单边成本0.1%、现金收益0。

**证据边界：**我们是在已经看到2017年后失效的情况下设计改进，参数选择也参考了历史结果，因此2017—2026是后验研究区间，不是事前注册的盲样本外。代码不存在未来函数，规则在各分期不变，但这只能证明历史计算可执行。2026年之后的新数据才构成真正的前瞻检验。

## 1. 结论先行

1. **主要失效来自预测关系衰减。**右偏RSRS对次日执行后未来10日收益的相关性在开发期为{diagnosis.loc[diagnosis.分期=='开发期','Pearson'].iloc[0]:.3f}，2019报告发表后为{diagnosis.loc[diagnosis.分期=='2019报告发表后','Pearson'].iloc[0]:.3f}。固定阈值面对的已不是同一种稳定映射。
2. **2019年的波动率钝化只解决震荡误触发的一部分。**固定原参数复现该公式后，2017—2019年表现改善；2019-11-14以后年化{pct(post.loc['2019钝化公式','年化收益'])}、最大回撤{pct(post.loc['2019钝化公式','最大回撤'])}，说明单独压缩高波动信号不足以应对长期结构变化。
3. **状态自适应RSRS在后验历史中的改善主要体现在风险收益质量。**2019报告发表后年化{pct(post.loc['状态自适应RSRS','年化收益'])}、夏普{post.loc['状态自适应RSRS','夏普']:.3f}、最大回撤{pct(post.loc['状态自适应RSRS','最大回撤'])}；同期买入持有年化{pct(post.loc['买入持有','年化收益'])}、最大回撤{pct(post.loc['买入持有','最大回撤'])}。2022年以来，改进策略年化{pct(recent.loc['状态自适应RSRS','年化收益'])}，同期市场{pct(recent.loc['买入持有','年化收益'])}。
4. **证据仍不足以宣称发现稳定的新因子。**改进策略平均仓位较低，收益差的区间可能因少持仓而被动降低；块bootstrap区间较宽，参数邻域表现也存在差异。更准确的结论是：长期趋势约束和滚动尺度校准显著降低了原规则的错误暴露，当前样本下改善了回撤和近期收益。

![发表后净值与回撤](01_post2017_nav.png)

## 2. 2019年报告提供了什么

2019年报告把问题定位为震荡行情下信号频繁切换、胜率下降。它测试三类加权回归：时间线性权重和指数权重总体无效，成交额加权对中证500、创业板更有帮助，对沪深300等大盘指数提升有限。较有效的方案是“钝化RSRS”：

`damped = z × (R²)^(2q)`，其中q是N日收益波动率在过去M日的分位数。波动越高，R²被提高到更大的幂，指标越靠近零，从而减少震荡期触发。

报告为沪深300重选了18/700/0.7，但该选择使用了截至2019年的数据。若从2017年开始评价，它含有未来信息。这里仅借用公式，继续固定2017年原参数18/600/±0.7。2019年报告发表后的区间是作者当时未见的数据，但对我们当前研究仍是已经观察到的历史。

## 3. 为什么原策略后来表现不好

### 3.1 指标的方向信息消失

![预测关系衰减](03_predictive_decay.png)

{markdown(diagnosis,[''],['Pearson','Spearman'])}

RSRS最初依赖“指标越高，未来收益越高”的排序关系。该关系接近零后，改阈值只能改变交易频率，无法恢复方向性。2019钝化将弱信号压回零，但不会创造新的预测信息。

### 3.2 固定绝对阈值不适应指标分布漂移

原规则长期固定±0.7。当波动、R²和beta分布改变时，±0.7所代表的历史极端程度也改变。结果可能是某些阶段长期不触发，另一些阶段频繁触发。改进方案因此使用只含过去数据的滚动经验分位数。

### 3.3 原规则把“高低价结构”直接当成市场方向

RSRS描述日内高低价关系，但同样的形态可出现在反弹、下跌中继和震荡。2019报告发现的震荡误判是其中一种表现。我们的交易账本还显示，2021年底至2023年的多笔持仓在长期下行趋势中持续亏损；仅过滤开仓、却不在长期趋势破坏时退出，会放大这种错误。

{markdown(trade,['胜率','平均每段收益'],['平均持有日'])}

### 3.4 表现落后来自“错过上涨”和“未充分规避下跌”的不对称

{markdown(act,['错过上涨日收益之和','规避下跌日损失之和','成本之和','平均仓位'])}

这里的日收益加总是归因量，不等于复合收益。它帮助回答策略为何落后：空仓能否规避的下跌，是否足以补偿被错过的上涨与成本。

## 4. 自有改进：状态自适应RSRS

改进保留18/600右偏RSRS本体，不用2017年后的收益重估beta或拟合预测模型。规则为：

- **尺度校准：**把当日右偏RSRS转换为过去756个交易日约三年的经验分位数，最少252日才启用。
- **开仓：**指标处于前20%，收盘价高于MA250，并且MA250高于20日前。
- **平仓：**指标落入后20%，或收盘价跌破MA250；二者任一成立即退出。
- **执行：**条件都由当日及过去数据计算，次日收盘成交。参数在所有后续时期和三个指数上保持不变。

这个结构分别解决三个问题：滚动分位数处理量纲漂移，长期均线区分趋势状态，非对称退出避免RSRS在长期下行中继续持仓。MA250单独列作控制，以判断改善是否只是普通趋势策略。

## 5. 全部分期结果

{markdown(table,['累计收益','年化收益','最大回撤','平均绝对仓位'],['夏普','年均单边换手'])}

![年度收益](02_yearly_returns.png)

状态自适应RSRS在2019报告发表后略高于买入持有，同时显著降低最大回撤；与纯MA250相比，最大回撤也明显更小，说明结果不是完全由均线贡献。但低仓位是收益稳定的重要来源，应把它理解成防御型择时，不应把低回撤全部归因于RSRS预测能力。

## 6. 参数邻域与跨市场

主方案选择三年分位窗口、MA250和80/20分位，具有“三年尺度、一年趋势、五等分极端区域”的经济解释，但并非事前预注册，选择过程参考了历史结果。为揭示数据挖掘风险，完整报告3×3×3=27个相邻组合。

{markdown(stable,['正年化比例','年化收益中位数','最大回撤中位数','最差最大回撤'])}

![参数敏感性](04_sensitivity.png)

{markdown(cross_table,['累计收益','年化收益','最大回撤','平均绝对仓位'],['夏普'])}

跨市场结果并不一致：在上证50上，改进策略的收益和回撤均优于买入持有及原规则；在中证500上，低仓位使其明显错过市场上涨，收益远低于买入持有和纯趋势策略。因此它更像大盘指数的防御型改进，不是可直接推广到全部宽基指数的通用规则。跨市场表是迁移检验，不针对各指数重新调参。

## 7. 统计不确定性

{markdown(bs,['日均超额年化','CI下','CI上'],[])}

区间来自成对循环移动块bootstrap，比较改进策略与原始价格过滤右偏的日均收益差，报告30/60/120日块长。它不是CAGR置信区间；区间若包含零，表示当前样本不能把收益差与零显著区分。最大回撤的改善是历史路径事实，也不等于未来保证。

## 8. 评价与下一步

这次改进达到了两个目标：明确解释了原策略失效不只是成本，而是预测关系和市场状态共同变化；提出一套固定且无未来函数、在后验历史中改善回撤和近期收益的规则。它没有达到“可直接实盘”或“已通过独立样本外验证”的证据标准。

下一步应按时间推进做walk-forward：每年只使用此前数据更新分位数，保留参数不变；加入ETF可交易价格、股息、滑点和现金收益；对2026年之后数据继续盲测。若需要进一步提高收益，应优先研究仓位分级或跨资产配置，而不是继续在同一段历史上搜索更多N/M/S组合。

## 复现文件

- `performance.csv`：所有分期和策略绩效。
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
            q=period_metrics(result,*PERIODS["2019报告发表后"])
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
                q=bootstrap_mean_difference(results["状态自适应RSRS"].ret.loc[PERIODS["2019报告发表后"][0]:],
                                            results["原始价格过滤右偏"].ret.loc[PERIODS["2019报告发表后"][0]:],block)
                boot.append({"比较":"状态自适应RSRS - 原始价格过滤右偏",**q})
            bootstrap=pd.DataFrame(boot)
            main_objects=(data,results,yearly,diagnosis,trades,active,sensitivity_frame,signal_frame,bootstrap)
    performance=pd.DataFrame(performance);cross=pd.DataFrame(cross)
    data,results,yearly,diagnosis,trades,active,sensitivity_frame,signal_frame,bootstrap=main_objects
    frames={"performance":performance,"diagnosis_ic":diagnosis,"trade_diagnostics":trades,
            "active_return_attribution":active,"sensitivity":sensitivity_frame,
            "cross_market":cross,"yearly":yearly,"signals":signal_frame,"bootstrap":bootstrap}
    for name,frame in frames.items(): frame.to_csv(out/f"{name}.csv",index=False,encoding="utf-8-sig")
    if not args.no_plots: plots(out,data,results,yearly,diagnosis,sensitivity_frame)
    write_report(out,performance,diagnosis,trades,active,sensitivity_frame,cross,bootstrap)
    config={"development_end":"2017-04-28","evaluation_end":SPEC.end,
            "execution":SPEC.execution,"cost_one_way":SPEC.cost,
            "author_benchmark":{"formula":"z*(R2**(2*volatility_rank))","n":18,"m":600,"thresholds":[.7,-.7]},
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

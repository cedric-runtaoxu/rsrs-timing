"""按2017年光大RSRS研报的1—4章、图1—32、表1—10运行复现。

python paper.py              # 计算、生成全部图表与章节报告
python paper.py --no-plots   # 只计算；不创建图片，不把图片标成已生成
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research import signals
from rsrs.data import load_index, SYMBOLS
from rsrs.backtest import backtest
from rsrs.indicators import rsrs_family, rsrs_signal
from rsrs.research_checks import execution_trades, independent_account_check
from rsrs.research_spec import LONG_STRATEGIES, strategy_manifest
from rsrs.paper_study import (PAPER_SPEC, START, MAIN_END, CROSS_END, N_GRID, M_GRID, COSTS,
                              STAT_ROWS, period_stats, distribution, grouped_prediction, grouped_correlations)

ROOT=Path(__file__).resolve().parent
FIGURES={
 1:'布林带上下轨突破策略',2:'传统均线策略',3:'牛市高斜率示意',4:'熊市高斜率示意',
 5:'牛市低斜率示意',6:'熊市低斜率示意',7:'历史斜率分布',8:'各时期斜率均值',
 9:'斜率和标准分策略净值',10:'斜率策略交易成本',11:'斜率N参数净值路径',12:'斜率N参数期末净值',
 13:'标准分策略交易成本',14:'标准分分布',15:'修正标准分分布',
 16:'标准分未来10日上涨概率',17:'标准分未来10日期望收益',18:'修正标准分未来10日上涨概率',
 19:'修正标准分未来10日期望收益',20:'右偏标准分分布',21:'右偏标准分未来10日上涨概率',
 22:'右偏标准分未来10日期望收益',23:'三种标准分策略净值',24:'右偏标准分策略交易成本',
 25:'价格过滤三种标准分净值',26:'价格过滤右偏交易成本',27:'成交量过滤三种标准分净值',
 28:'标准分M参数敏感性',29:'沪深300价格过滤右偏多空',30:'上证50价格过滤右偏多头',
 31:'中证500价格过滤右偏多头',32:'六种主要策略净值汇总（按原图内容）'}
TABLES={1:'斜率低阶矩',2:'斜率与标准分交易统计',3:'标准分与修正标准分分布',4:'三种标准分交易统计',
        5:'价格过滤交易统计',6:'成交量过滤交易统计',7:'标准分M参数敏感性',8:'多空与跨市场交易统计',
        9:'十种策略规则汇总',10:'六种主要策略绩效汇总'}


class Study:
    def __init__(self, out, data_dir, plots=True):
        self.out=Path(out);self.data_dir=Path(data_dir);self.plots=plots
        for folder in ['figures','tables','data']:(self.out/folder).mkdir(parents=True,exist_ok=True)
        self.tables={};self.figure_data={};self.results={};self.market_data={};self.families={};self.signal_sets={}
        self.audit=[];self.coverage=[]
        if plots:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
            installed={x.name for x in font_manager.fontManager.ttflist}
            for font in ['Microsoft YaHei','SimHei','Noto Sans CJK SC','WenQuanYi Zen Hei']:
                if font in installed:
                    plt.rcParams['font.sans-serif']=[font];break
            plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
                                 'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':160})
            self.plt=plt

    def csv(self,name,frame,folder='data',index=False):
        frame.to_csv(self.out/folder/f'{name}.csv',index=index,encoding='utf-8-sig')

    def table(self,n,frame):
        self.tables[n]=frame
        self.csv(f'table_{n:02d}',frame,'tables')
        self.coverage.append({'类别':'表','编号':n,'原文内容':TABLES[n],'状态':'已计算',
                              '结果文件':f'tables/table_{n:02d}.csv','口径说明':'本地数据及明确实施约定；非原文数字复制'})

    def figure(self,n,frame,kind='line',ylabel='净值（初始资金1）',foot=None):
        self.figure_data[n]=frame
        self.csv(f'fig_{n:02d}',frame,index=True)
        schematic=n in [3,4,5,6]
        approx=n in [16,17,18,19,21,22]
        explanation='概念示意，合成路径，不是历史实证' if schematic else '50个等宽箱，原文未披露精确分箱' if approx else '同主题本地重算'
        self.coverage.append({'类别':'图','编号':n,'原文内容':FIGURES[n],
                              '状态':'示意重绘' if schematic and self.plots else '已重算绘图' if self.plots else '仅数据',
                              '结果文件':f'figures/fig_{n:02d}.png' if self.plots else f'data/fig_{n:02d}.csv',
                              '口径说明':explanation})
        if not self.plots:return
        fig,ax=self.plt.subplots(figsize=(10.8,4.7))
        if kind=='hist':
            ax.hist(frame.iloc[:,0].dropna(),bins=50,color='#28748c',edgecolor='white',linewidth=.3)
            ax.set_xlabel('指标值')
        elif kind=='bar':
            ax.bar(frame.index,frame.iloc[:,0],width=.065,color='#28748c')
            ax.set_xlabel('指标分箱中点')
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            for col in frame:
                ax.plot(frame.index,frame[col],label=col,lw=1.15,marker='o' if kind=='points' else None,
                        markersize=3)
            ax.legend(loc='upper left',fontsize=8,ncol=4 if len(frame.columns)>6 else 3)
            if kind=='points':ax.set_xlabel(frame.index.name or '参数')
            if schematic:ax.set_xlabel('示意交易日序号')
        ax.set_title(f'图{n}  {FIGURES[n]}',loc='left',pad=13)
        ax.set_ylabel(ylabel);ax.grid(alpha=.18)
        if kind!='hist' and isinstance(frame.index,pd.DatetimeIndex):
            import matplotlib.dates as md
            ax.xaxis.set_major_locator(md.YearLocator(2));ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
        fig.text(.07,.015,foot or explanation,fontsize=8,color='#555555')
        fig.tight_layout(rect=(0,.045,1,1))
        fig.savefig(self.out/'figures'/f'fig_{n:02d}.png');self.plt.close(fig)

    def result(self,symbol,name,cost=0):
        key=(symbol,name,cost)
        if key not in self.results:
            d=self.market_data[symbol]
            self.results[key]=backtest(d,self.signal_sets[symbol][name],cost=cost,execution=PAPER_SPEC.execution,lag=PAPER_SPEC.lag)
            err=independent_account_check(d,self.signal_sets[symbol][name],self.results[key],cost=cost,lag=PAPER_SPEC.lag)
            self.audit.append({'市场':SYMBOLS[symbol],'策略':name,'单边成本':cost,'独立核账误差':err})
        return self.results[key]

    def nav(self,names,symbol='sh000300',end=MAIN_END):
        return pd.DataFrame({name:(1+self.result(symbol,name).ret.loc[START:end]).cumprod() for name in names})

    def stats_table(self,n,names,symbol='sh000300',end=MAIN_END):
        z=pd.DataFrame({name:period_stats(self.result(symbol,name),end=end) for name in names}).loc[STAT_ROWS]
        self.table(n,z.rename_axis('统计量').reset_index())

    def cost_figure(self,n,name,symbol='sh000300',end=MAIN_END):
        f=self.nav(['买入持有'],symbol,end)
        for cost in COSTS:
            lab='策略无成本' if cost==0 else f'策略双边{cost*2:.1%}'
            f[lab]=(1+self.result(symbol,name,cost).ret.loc[START:end]).cumprod()
        self.figure(n,f,foot=f'{SYMBOLS[symbol]} | {f.index[0].date()}—{f.index[-1].date()} | 次日收盘成交；费用为两次单边费之和')

    def run(self):
        for symbol in ['sh000300','sh000016','sh000905']:
            d=load_index(symbol,end=MAIN_END,data_dir=self.data_dir,invalid_ohlc='mask')
            self.market_data[symbol]=d
            sig,f=signals(d,PAPER_SPEC);self.signal_sets[symbol]=sig;self.families[symbol]=f
            self.csv(f'{symbol}_indicators',f,index=True);self.csv(f'{symbol}_signals',pd.DataFrame(sig),index=True)
        d=self.market_data['sh000300'];f=self.families['sh000300'].loc[START:MAIN_END]
        # 第1章：传统阈值策略与高低价关系示意。
        self.cost_figure(1,'BOLL(14,2)');self.cost_figure(2,'MA20')
        for n,direction,beta in [(3,1,1.4),(4,-1,1.4),(5,1,.6),(6,-1,.6)]:
            low=100+direction*np.arange(18)*.5
            high=106+beta*(low-100)
            self.figure(n,pd.DataFrame({'最低价':low,'最高价':high},index=np.arange(1,19)),ylabel='合成价格',
                        foot=f'概念示意：β={beta}，仅解释相对变化；不表示真实市场或必然预测结果。')
        # 第2.1节：斜率分布、滚动均值与阈值依据。
        self.figure(7,f[['beta']].rename(columns={'beta':'斜率'}),'hist','频数')
        self.table(1,pd.DataFrame([distribution(f.beta)],index=['斜率']).rename_axis('指标').reset_index())
        mean=self.families['sh000300'].beta.shift(1).rolling(250,min_periods=250).mean().loc[START:MAIN_END]
        self.figure(8,mean.to_frame('此前250日斜率均值'),ylabel='斜率均值')
        # 第2.2节：基础策略、完整交易统计、费用与原图N网格。
        self.figure(9,self.nav(['买入持有','斜率','标准分']))
        self.stats_table(2,['斜率','标准分'])
        self.cost_figure(10,'斜率')
        scan=self.nav(['买入持有']);nrows=[]
        for n in N_GRID:
            fam=rsrs_family(d,n=n,m=600,min_periods=PAPER_SPEC.min_periods)
            bt=backtest(d,rsrs_signal(fam.beta,1.,.8),cost=0)
            scan[f'N={n}']=(1+bt.ret.loc[START:MAIN_END]).cumprod()
            nrows.append({'N':n,**period_stats(bt)})
        self.nscan=pd.DataFrame(nrows);self.csv('n_sensitivity',self.nscan)
        self.figure(11,scan);self.figure(12,self.nscan.set_index('N')[['期末净值']],'points')
        self.cost_figure(13,'标准分')
        # 第2.3节：分布修正、原文式按水平汇总预测性、三指标策略。
        for n,col in [(14,'zscore'),(15,'modified'),(20,'right_skewed')]:
            self.figure(n,f[[col]],'hist','频数')
        self.table(3,pd.DataFrame([distribution(f.zscore),distribution(f.modified)],index=['标准分','修正标准分']).rename_axis('指标').reset_index())
        corr=[];bin_sensitivity=[]
        for name,col,nums in [('标准分','zscore',(16,17)),('修正标准分','modified',(18,19)),('右偏标准分','right_skewed',(21,22))]:
            g=grouped_prediction(f[col],d.close);self.csv(f'prediction_{col}',g)
            cr=grouped_correlations(g);cr.insert(0,'指标',name);corr.append(cr)
            self.figure(nums[0],g.set_index('箱中点')[['上涨概率']],'bar','上涨概率',foot='N18 M600；50等宽箱；完整未来10日标签；样本数见同名预测数据表。')
            self.figure(nums[1],g.set_index('箱中点')[['未来10日收益']],'bar','未来10日收益',foot='分箱条件均值，不是策略收益；空箱留空，稀疏尾部不隐藏。')
            for count in [30,50,70]:
                cg=grouped_correlations(grouped_prediction(f[col],d.close,bins=count))
                cg.insert(0,'指标',name);cg.insert(1,'等宽箱数',count);bin_sensitivity.append(cg)
        self.predictive=pd.concat(corr,ignore_index=True);self.csv('prediction_correlations',self.predictive)
        self.csv('prediction_bin_sensitivity',pd.concat(bin_sensitivity,ignore_index=True))
        self.figure(23,self.nav(['买入持有','标准分','修正标准分','右偏(16,300)']))
        self.stats_table(4,['标准分','修正标准分','右偏(16,300)']);self.cost_figure(24,'右偏(16,300)')
        self.cases()
        # 第3.1及3.2节：价格、成交量过滤。共同参数18/600/0.7。
        price=['价格过滤·'+s for s in ['标准分','修正','右偏']]
        volume=['量过滤·'+s for s in ['标准分','修正','右偏']]
        self.figure(25,self.nav(['买入持有']+price));self.stats_table(5,price);self.cost_figure(26,'价格过滤·右偏')
        self.figure(27,self.nav(['买入持有']+volume));self.stats_table(6,volume)
        # 表7的单指标600值与原文标准分对应：固定标准分族，不混合三指标平均。
        mrows=[]
        for m in M_GRID:
            sig,_=signals(d,replace(PAPER_SPEC,m=m))
            row={'M':m}
            for name,label in [('标准分','单指标净值'),('价格过滤·标准分','价格优化净值'),('量过滤·标准分','相关性优化净值')]:
                bt=backtest(d,sig[name],cost=0)
                row[label]=float((1+bt.ret.loc[START:MAIN_END]).prod())
            mrows.append(row)
        self.table(7,pd.DataFrame(mrows));self.figure(28,self.tables[7].set_index('M'),'points')
        # 第3.3节：原文只展示一组多空和两个跨市场多头。
        combos=[('sh000300','价格过滤·右偏','沪深300多头'),('sh000300','多空·价格过滤·右偏','沪深300多空'),
                ('sh000016','价格过滤·右偏','上证50多头'),('sh000905','价格过滤·右偏','中证500多头')]
        t8=pd.DataFrame({label:period_stats(self.result(s,n),end=CROSS_END) for s,n,label in combos}).loc[STAT_ROWS]
        self.table(8,t8.rename_axis('统计量').reset_index())
        self.cross_cost=[]
        for symbol,name,label in combos:
            for cost in COSTS:
                self.cross_cost.append({'市场版本':label,'单边成本':cost,**period_stats(self.result(symbol,name,cost),end=CROSS_END)})
        self.csv('cross_market_costs',pd.DataFrame(self.cross_cost))
        self.cost_figure(29,'多空·价格过滤·右偏',end=CROSS_END)
        self.cost_figure(30,'价格过滤·右偏','sh000016',CROSS_END)
        self.cost_figure(31,'价格过滤·右偏','sh000905',CROSS_END)
        rules=[]
        for name in LONG_STRATEGIES:
            ind='β' if name=='斜率' else 'z×R²×β' if '右偏' in name else 'z×R²' if '修正' in name else 'z'
            filt='MA20(t−1)>MA20(t−3)' if '价格过滤' in name else 'corr(前10日量,修正标准分)>0' if '量过滤' in name else '无'
            rules.append({'策略':name,'指标':ind,'N/M':'18/不适用' if name=='斜率' else '16/300' if name=='右偏(16,300)' else '18/600',
                          '开/平阈值':'1.0/0.8' if name=='斜率' else '0.7/−0.7','开仓过滤':filt})
        self.table(9,pd.DataFrame(rules))
        summary=['斜率','标准分','修正标准分','右偏(16,300)','价格过滤·右偏','量过滤·右偏']
        self.figure(32,self.nav(summary),foot='原图32标题写成交量修正成本，图例实际为六策略汇总；此图按图例重建。')
        z=pd.DataFrame({n:period_stats(self.result('sh000300',n)) for n in summary})
        self.table(10,z.loc[['年化收益率','夏普比率','最大回撤','胜率（按次）','平均盈亏比（按次）']].rename_axis('统计量').reset_index())
        # 图32标题所指对象另存费用数据，正文明确标题/内容不一致。
        self.csv('volume_modified_costs',pd.DataFrame([{'单边成本':c,**period_stats(self.result('sh000300','量过滤·修正',c))} for c in COSTS]))
        self.export_audit()
        from rsrs.paper_reporting import write_report
        write_report(self)
        generated={p.relative_to(self.out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in self.out.rglob('*') if p.is_file() and p.name!='generated_sha256.json'}
        (self.out/'generated_sha256.json').write_text(json.dumps(generated,ensure_ascii=False,indent=2),encoding='utf-8')

    def cases(self):
        # 2008所有标准分持仓段 + 2009—2014逐年净值；不挑选单笔有利交易。
        bt=self.result('sh000300','标准分')
        ledger=execution_trades(self.market_data['sh000300'],bt.execution_position)
        case=ledger[(ledger['建仓日']<='2008-12-31')&(ledger['估值日']>='2008-01-01')].copy()
        case['当段毛收益']=case.pop('毛收益')
        d=self.market_data['sh000300'];f=self.families['sh000300']
        for label,col in [('开仓信号标准分','zscore'),('开仓信号斜率','beta'),('开仓信号R²','r2')]:
            case[label]=[f[col].iloc[d.index.get_loc(t)-PAPER_SPEC.lag] for t in case['建仓日']]
        self.case2008=case;self.csv('case_2008_trades',case)
        names=['买入持有','斜率','标准分','修正标准分','右偏(16,300)','价格过滤·标准分','量过滤·标准分']
        colors=dict(zip(names,['#4c6071','#d58d32','#298a51','#b83f4c','#8561a8','#377f9c','#a17735']))
        daily=pd.DataFrame({n:self.result('sh000300',n).ret for n in names})
        annual=[]
        for year in range(2009,2015):
            annual.append({'年份':year,**{n:float((1+daily.loc[str(year),n]).prod()-1) for n in names}})
        self.sideways=pd.DataFrame(annual);self.csv('case_2009_2014_yearly',self.sideways)
        self.case_stats=pd.DataFrame([{'区间':period,'策略':n,**period_stats(self.result('sh000300',n),start,end)}
                    for period,start,end in [('2008','2008-01-01','2008-12-31'),('2009—2014','2009-01-01','2014-12-31')]
                    for n in names]);self.csv('case_period_stats',self.case_stats)
        if self.plots:
            for period,start,end in [('2008','2008-01-01','2008-12-31'),('2009_2014','2009-01-01','2014-12-31')]:
                eq=(1+daily.loc[start:end]).cumprod()
                fig,axes=self.plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
                eq.plot(ax=axes[0],lw=1.1,color=[colors[n] for n in names]);axes[0].set_ylabel('区间净值');axes[0].legend(fontsize=8,ncol=3)
                for n in ['标准分','价格过滤·标准分','量过滤·标准分']:
                    axes[1].step(d.loc[start:end].index,self.result('sh000300',n).position.loc[start:end],where='pre',label=n,alpha=.7,color=colors[n])
                axes[1].set_ylabel('承担当日损益的仓位');axes[1].legend(fontsize=8,ncol=3)
                fig.suptitle(f'{period.replace("_","—")} 行情复盘 | 无交易费 | 次日收盘成交')
                fig.tight_layout();fig.savefig(self.out/'figures'/f'case_{period}.png');self.plt.close(fig)

    def export_audit(self):
        stats=[];ledgers=[];checks=[]
        for (symbol,name,cost),bt in self.results.items():
            stats.append({'市场':SYMBOLS[symbol],'策略':name,'单边成本':cost,**period_stats(bt)})
            if cost==0:
                tr=execution_trades(self.market_data[symbol],bt.execution_position)
                tr.insert(0,'策略',name);tr.insert(0,'市场',SYMBOLS[symbol]);ledgers.append(tr)
        self.all_stats=pd.DataFrame(stats);self.csv('all_stats',self.all_stats)
        self.csv('execution_trades',pd.concat(ledgers,ignore_index=True))
        self.csv('account_checks',pd.DataFrame(self.audit))
        for symbol,d in self.market_data.items():
            sig,_=signals(d.loc[:CROSS_END],PAPER_SPEC)
            for name in self.signal_sets[symbol]:
                same=bool(np.allclose(sig[name],self.signal_sets[symbol][name].loc[:CROSS_END],atol=1e-12))
                if not same:raise AssertionError(f'截断重算失败: {symbol}/{name}')
                checks.append({'市场':SYMBOLS[symbol],'策略':name,'截止日':CROSS_END,'历史信号一致':same})
        self.csv('prefix_checks',pd.DataFrame(checks))
        # 机器可读图表覆盖清单，不将数据缺口或示意图伪装为完全数值复刻。
        self.coverage=pd.DataFrame(self.coverage).sort_values(['类别','编号'])
        self.csv('coverage',self.coverage,folder='.')
        files={p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name:hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [self.data_dir/f'{s}.csv' for s in self.market_data]}
        metadata={'calculated_at_utc':datetime.now(timezone.utc).isoformat(),'spec':asdict(PAPER_SPEC),
                  'main_requested':[START,MAIN_END],'cross_requested':[START,CROSS_END],
                  'n_grid':N_GRID,'m_grid':M_GRID,'costs_single_side':COSTS,'prediction_bins':50,
                  'source_sha256':files,'plots_generated':self.plots,
                  'code_hash_normalization':'CRLF normalized to LF',
                  'code_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                 for p in [ROOT/'paper.py',ROOT/'research.py',*sorted((ROOT/'rsrs').glob('*.py'))]},
                  'table7_indicator':'zscore; paper columns not explicit, inferred from baseline column',
                  'figure32':'follow six-series legend; title mismatch documented'}
        (self.out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        (self.out/'strategy_manifest.json').write_text(json.dumps(strategy_manifest(PAPER_SPEC),ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--outdir',type=Path,default=ROOT/'report'/'paper')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data'/'extended')
    a=p.parse_args();study=Study(a.outdir,a.data_dir,not a.no_plots);study.run()
    print(f'完成：32个图主题、10张表；报告 {a.outdir / "report.md"}')


if __name__=='__main__':main()

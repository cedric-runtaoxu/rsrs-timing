"""所有时期、市场共用的不可变配置；分期只改变统计窗口。"""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class ResearchSpec:
    n: int = 18
    m: int = 600
    right_n: int = 16
    right_m: int = 300
    min_periods: int = 10
    exclude_current: bool = False
    buy: float = .7
    sell: float = -.7
    beta_buy: float = 1.
    beta_sell: float = .8
    ma_window: int = 20
    volume_window: int = 10
    volume_lag: int = 1
    boll_window: int = 14
    boll_width: float = 2.
    boll_ddof: int = 0
    execution: str = 'next_close'
    lag: int = 1
    cost: float = .001
    short_cost: float = 0.
    annualization: int = 252
    invalid_ohlc: str = 'mask'
    end: str = '2026-10-21'


SPEC = ResearchSpec()
LONG_STRATEGIES = ['斜率', '标准分', '修正标准分', '右偏(16,300)',
                   '价格过滤·标准分', '量过滤·标准分', '价格过滤·修正',
                   '量过滤·修正', '价格过滤·右偏', '量过滤·右偏']
SHORT_STRATEGIES = ['多空·'+kind+'·'+label for kind in ['价格过滤','量过滤']
                    for label in ['标准分','修正','右偏']]
PAPER_STRATEGIES = LONG_STRATEGIES + SHORT_STRATEGIES
CONTROLS = ['右偏(18,600)']
BASELINES = ['买入持有', 'MA20', 'BOLL(14,2)']
ALL_STRATEGIES = BASELINES + LONG_STRATEGIES + CONTROLS + SHORT_STRATEGIES


def strategy_manifest(spec=SPEC):
    """每条规则的参数、交易语义和证据来源都参与配置校验。"""
    rows = {}
    for name in ALL_STRATEGIES:
        short = name in SHORT_STRATEGIES
        n,m = (spec.right_n,spec.right_m) if name=='右偏(16,300)' else (spec.n,spec.m)
        category = ('研报多空规则' if short else '研报多头策略' if name in LONG_STRATEGIES
                    else '同参数对照' if name in CONTROLS else '比较基线')
        indicator = ('beta' if name=='斜率' else 'zscore' if '标准分' in name and '修正' not in name
                     else 'modified' if '修正' in name else 'right_skewed')
        filt = 'ma_direction' if '价格过滤' in name else 'volume_corr_positive' if '量过滤' in name else 'none'
        p = dict(category=category,indicator=indicator,n=n,m=None if name=='斜率' else m,
                 min_periods=spec.min_periods,exclude_current=spec.exclude_current,z_ddof=1,
                 buy=spec.beta_buy if name=='斜率' else spec.buy,
                 sell=spec.beta_sell if name=='斜率' else spec.sell,
                 entry_filter=filt,ma_window=spec.ma_window,ma_offsets=[1,3],
                 volume_window=spec.volume_window,volume_indicator='modified',volume_lag=spec.volume_lag,
                 direction='long_short' if short else 'long_only',
                 long_entry='indicator>buy AND long_filter',long_exit='indicator<sell',
                 short_entry='indicator<sell AND short_filter' if short else None,
                 short_exit='indicator>buy' if short else None,
                 short_filter=('MA(t-1)<MA(t-3)' if filt=='ma_direction' else 'volume_corr>0') if short else None,
                 filter_exits=False,neutral_or_missing_indicator='hold_existing_position',
                 missing_filter='disallow_new_entry',threshold_rule='strict_level_not_crossing',
                 reversal='close_then_open_if_allowed; turnover=2',
                 execution=spec.execution,lag=spec.lag,cost=spec.cost,short_cost=spec.short_cost,
                 price_return_model='daily_signed_index_exposure',
                 annualization=spec.annualization,cash_return=0.,risk_free=0.,initial_position=0.,
                 reset_on_period_boundary=False,force_liquidation_at_end=False,invalid_ohlc=spec.invalid_ohlc)
        p['paper_evidence'] = ('第3.3节图29、表8单列多空效果' if name=='多空·价格过滤·右偏'
                               else '第3.1/3.2节做空规则的实现，未逐个披露对应绩效' if short
                               else '第2章/第3章' if name in LONG_STRATEGIES
                               else '同参数控制' if name in CONTROLS else '买入持有或第1.1节技术基线')
        if name in BASELINES:
            p.update(indicator=None,n=None,m=None,buy=None,sell=None,entry_filter='none',
                     long_entry='always' if name=='买入持有' else 'close>MA20' if name=='MA20' else 'close>upper_band',
                     long_exit='never' if name=='买入持有' else 'close<MA20' if name=='MA20' else 'close<lower_band')
            if name=='BOLL(14,2)':
                p.update(boll_window=spec.boll_window,boll_width=spec.boll_width,boll_ddof=spec.boll_ddof)
        canonical=json.dumps(p,sort_keys=True,ensure_ascii=False,separators=(',',':'))
        p['config_sha256']=hashlib.sha256(canonical.encode()).hexdigest()
        rows[name]=p
    return {'common_spec':asdict(spec),'strategies':rows}

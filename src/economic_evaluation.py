"""Execution-aware portfolios, equal-weight benchmarks, and active-risk audit."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'

def max_drawdown(returns):
    wealth=np.cumprod(1+np.asarray(returns,float));peaks=np.maximum.accumulate(np.r_[1.,wealth])
    return float(np.max(1-np.r_[1.,wealth]/peaks))

def summarize(r,periods=252/5):
    if r.empty:return {}
    wealth=float(np.prod(1+r.net_return));vol=r.net_return.std(ddof=1)
    return {'n_periods':len(r),'annual_return':wealth**(periods/len(r))-1 if wealth>0 else -1.,
        'annual_volatility':float(vol*np.sqrt(periods)),
        'sharpe':float(r.net_return.mean()/vol*np.sqrt(periods)) if vol>0 else np.nan,
        'max_drawdown':max_drawdown(r.net_return),'average_turnover':float(r.turnover.mean()),
        'total_return':wealth-1,'positive_period_rate':float((r.net_return>0).mean())}

def one_run(g,cost_bps=10,topk=3,style='long_short'):
    dates=np.sort(g.date.unique())[::5];prev=pd.Series(dtype=float);records=[]
    for date in dates:
        cs=g[g.date==date].sort_values('code')
        if style!='equal_weight' and len(cs)<2*topk:continue
        ranked=cs.sort_values('p')
        if style=='long_short':
            selected=pd.concat([ranked.head(topk),ranked.tail(topk)])
            weights=pd.Series(np.r_[np.full(topk,-1/topk),np.full(topk,1/topk)],index=selected.code)
        elif style=='long_only':
            selected=ranked.tail(topk);weights=pd.Series(np.full(topk,1/topk),index=selected.code)
        else:
            selected=cs;weights=pd.Series(np.full(len(cs),1/len(cs)),index=selected.code)
        union=weights.index.union(prev.index)
        turnover=float(.5*np.abs(weights.reindex(union).fillna(0)-prev.reindex(union).fillna(0)).sum()) if len(prev) else 1.
        gross=float(np.sum(weights.values*selected.exec_ret.values));net=gross-cost_bps/10000*turnover
        records.append({'date':date,'gross_return':gross,'net_return':net,'turnover':turnover});prev=weights
    r=pd.DataFrame(records);return r,summarize(r)

def buy_hold(panel,formation_dates,cost_bps):
    dates=np.sort(panel.date.unique());first=formation_dates[0];last=formation_dates[-1]
    entry=dates[np.searchsorted(dates,first,side='right')];exit=dates[min(np.searchsorted(dates,last)+6,len(dates)-1)]
    p=panel[(panel.date>=entry)&(panel.date<=exit)].pivot(index='date',columns='code',values='close').ffill().dropna(axis=1)
    w=np.full(p.shape[1],1/p.shape[1]);wealth=(1-cost_bps/10000)*(p/p.iloc[0]).mul(w,axis=1).sum(axis=1)
    ret=wealth.pct_change().fillna(0);curve=pd.DataFrame({'date':wealth.index,'net_return':ret.values,'gross_return':ret.values,'turnover':0.})
    summary=summarize(curve,252);summary['average_turnover']=1/len(curve);summary['total_return']=wealth.iloc[-1]-1
    return curve,summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--horizon',type=int,default=5);ap.add_argument('--input',default='')
    ap.add_argument('--suffix',default='');args=ap.parse_args()
    if args.horizon!=5:raise ValueError('The non-overlapping backtest is defined for h=5')
    path=Path(args.input) if args.input else OUT/'predictions_h5.csv';pred=pd.read_csv(path,parse_dates=['date'])
    execution=[];panels={}
    for market in ['CN','TH']:
        d=pd.read_csv(ROOT/'data'/f'{market.lower()}_processed.csv',parse_dates=['date']).sort_values(['code','date'])
        d['exec_ret']=d.groupby('code').close.shift(-6)/d.groupby('code').close.shift(-1)-1
        panels[market]=d;execution.append(d[['date','code','market','exec_ret']])
    execution=pd.concat(execution,ignore_index=True)
    pred=pred.merge(execution,on=['date','code','market'],how='left',validate='many_to_one').dropna(subset=['exec_ret'])
    all_curves=[];rows=[];curve_map={};bench_summary={};bench_curve={};buy_rows=[]
    costs=[0,5,10,25,50]
    for market in ['CN','TH']:
        template=pred[(pred.market==market)&(pred.method=='adwinDTARC')]
        for seed in sorted(template.seed.unique()):
            g=template[template.seed==seed]
            for cost in costs:
                curve,summary=one_run(g,cost_bps=cost,style='equal_weight');key=(market,seed,cost)
                bench_summary[key]=summary;bench_curve[key]=curve
                rows.append({'market':market,'method':'equal_weight','seed':seed,'style':'benchmark','cost_bps':cost}|summary|
                    {'annual_excess_return':0.,'information_ratio':np.nan,'market_beta':1.,'annual_alpha':0.})
                c=curve.assign(market=market,method='equal_weight',seed=seed,style='benchmark',cost_bps=cost)
                c['wealth']=(1+c.net_return).cumprod();c['drawdown']=1-c.wealth/c.wealth.cummax().clip(lower=1);all_curves.append(c)
                curve_map[(market,'equal_weight',seed,'benchmark',cost)]=curve
                formation=np.sort(g.date.unique())[::5]
                bh,bs=buy_hold(panels[market],formation,cost)
                buy_rows.append({'market':market,'method':'buy_hold','seed':seed,'style':'buy_hold','cost_bps':cost}|bs)
    for keys,g in pred.groupby(['market','method','seed']):
        for style in ['long_short','long_only']:
            for cost in costs:
                curve,summary=one_run(g,cost_bps=cost,style=style);market,method,seed=keys
                b=bench_curve[market,seed,cost];bs=bench_summary[market,seed,cost]
                aligned=curve.merge(b[['date','net_return']],on='date',suffixes=('','_benchmark'))
                active=aligned.net_return-aligned.net_return_benchmark;sd=active.std(ddof=1)
                cov=np.cov(aligned.net_return,aligned.net_return_benchmark,ddof=1)[0,1];var=aligned.net_return_benchmark.var(ddof=1)
                beta=cov/var if var>0 else np.nan;alpha=(aligned.net_return.mean()-beta*aligned.net_return_benchmark.mean())*(252/5)
                extra={'annual_excess_return':summary['annual_return']-bs['annual_return'],
                    'information_ratio':float(active.mean()/sd*np.sqrt(252/5)) if sd>0 else np.nan,
                    'market_beta':beta,'annual_alpha':alpha}
                rows.append(dict(zip(['market','method','seed'],keys))|{'style':style,'cost_bps':cost}|summary|extra)
                curve_map[(market,method,seed,style,cost)]=curve
                if not curve.empty:
                    curve=curve.assign(market=market,method=method,seed=seed,style=style,cost_bps=cost)
                    curve['wealth']=(1+curve.net_return).cumprod();curve['drawdown']=1-curve.wealth/curve.wealth.cummax().clip(lower=1)
                    all_curves.append(curve)
    suffix=f'_{args.suffix}' if args.suffix else ''
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/f'economic_metrics_h5{suffix}.csv',index=False)
    pd.concat(all_curves,ignore_index=True).to_csv(OUT/f'economic_curves_h5{suffix}.csv',index=False)
    pd.DataFrame(buy_rows).to_csv(OUT/f'economic_buy_hold_h5{suffix}.csv',index=False)
    show=metrics[(metrics.cost_bps==10)&(metrics['style']=='long_short')]
    print(show.groupby(['market','method'])[['annual_return','annual_excess_return','sharpe','market_beta','max_drawdown']].mean().round(4))
    print('\nEqual-weight and buy-and-hold benchmarks at 10 bps:')
    print(metrics[(metrics.cost_bps==10)&(metrics.method=='equal_weight')].groupby('market')[['annual_return','sharpe','max_drawdown']].mean().round(4))
    print(pd.DataFrame(buy_rows).query('cost_bps==10').groupby('market')[['annual_return','sharpe','max_drawdown']].mean().round(4))
if __name__=='__main__':main()

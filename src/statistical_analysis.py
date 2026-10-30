"""Prespecified paired bootstrap, block sensitivity, and multiplicity control."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import matthews_corrcoef,roc_auc_score,brier_score_loss,accuracy_score

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
PRIMARY={'static','adwinER','periodicER'}

def metrics(y,p):
    yh=(p>=.5).astype(int)
    return {'mcc':matthews_corrcoef(y,yh),'auc':roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan,
            'brier':brier_score_loss(y,p),'accuracy':accuracy_score(y,yh)}

def blocks(n,b,rng):
    starts=rng.integers(0,max(1,n-b+1),size=int(np.ceil(n/b)))
    return np.concatenate([np.arange(s,min(s+b,n)) for s in starts])[:n]

def prepare(pred,market,seed,methods):
    g=pred[(pred.market==market)&(pred.seed==seed)]
    base=g[g.method==methods[0]][['date','code','y']].copy()
    for method in methods:
        p=g[g.method==method][['date','code','p']].rename(columns={'p':method})
        base=base.merge(p,on=['date','code'],how='inner',validate='one_to_one')
    return base

def holm(p):
    p=np.asarray(p,float);m=len(p);order=np.argsort(p);adj=np.empty(m);running=0
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*p[idx]);adj[idx]=min(1,running)
    return adj

def bootstrap(args):
    pred=pd.read_csv(OUT/args.input,parse_dates=['date']);rows=[]
    for block in args.blocks:
        rng=np.random.default_rng(args.seed+block)
        for market in ['CN','TH']:
            for comparator in ['adwinER','static','periodicER','phDTARC']:
                methods=['adwinDTARC',comparator]
                paired={s:prepare(pred,market,s,methods) for s in sorted(pred.seed.unique())}
                dates={s:np.sort(d.date.unique()) for s,d in paired.items()};lookups={}
                for s,d in paired.items():
                    pos=d.date.map({x:i for i,x in enumerate(dates[s])}).values
                    lookups[s]={i:np.where(pos==i)[0] for i in range(len(dates[s]))}
                observed=[]
                for s,d in paired.items():
                    a=metrics(d.y.values,d[methods[0]].values);b=metrics(d.y.values,d[methods[1]].values)
                    observed.append({m:a[m]-b[m] for m in a})
                obs=pd.DataFrame(observed).mean().to_dict();boot={m:[] for m in obs}
                for _ in range(args.replicates):
                    per={m:[] for m in obs}
                    for s,d in paired.items():
                        sampled=blocks(len(dates[s]),block,rng)
                        idx=np.concatenate([lookups[s][i] for i in sampled])
                        y=d.y.values[idx];a=metrics(y,d[methods[0]].values[idx]);b=metrics(y,d[methods[1]].values[idx])
                        for m in obs:per[m].append(a[m]-b[m])
                    for m in obs:boot[m].append(np.nanmean(per[m]))
                for metric,vals in boot.items():
                    vals=np.asarray(vals);lo,hi=np.nanpercentile(vals,[2.5,97.5])
                    p=min(1.,2*min((np.sum(vals<=0)+1)/(len(vals)+1),(np.sum(vals>=0)+1)/(len(vals)+1)))
                    rows.append({'market':market,'comparison':f'adwinDTARC - {comparator}','comparator':comparator,
                        'family':'primary' if comparator in PRIMARY else 'exploratory','metric':metric,
                        'mean_difference':obs[metric],'ci_2.5':lo,'ci_97.5':hi,'bootstrap_p':p,
                        'replicates':args.replicates,'block_days':block})
                print('bootstrap',block,market,comparator,flush=True)
    out=pd.DataFrame(rows);out['holm_p']=np.nan
    mask=(out.block_days==5)&(out.family=='primary')
    out.loc[mask,'holm_p']=holm(out.loc[mask,'bootstrap_p'].values)
    out.to_csv(OUT/args.output,index=False)
    out[out.block_days==5].to_csv(OUT/'paired_block_bootstrap_h5.csv',index=False)
    print(out.round(5).to_string(index=False))

def ablation_ci():
    d=pd.read_csv(OUT/'metrics_ablation_h5.csv');rows=[]
    for market in ['CN','TH']:
        for variant in sorted(d.method.unique()):
            if variant=='adwinDTARC':continue
            for metric in ['accuracy','macro_f1','mcc','auc','brier']:
                wide=d[d.market==market].pivot(index='seed',columns='method',values=metric)
                delta=wide[variant]-wide['adwinDTARC'];n=delta.notna().sum();mean=delta.mean();sd=delta.std(ddof=1)
                half=student_t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else np.nan
                rows.append({'market':market,'comparison':f'{variant} - adwinDTARC','metric':metric,'n_seeds':n,
                    'mean_difference':mean,'seed_sd':sd,'ci_2.5':mean-half,'ci_97.5':mean+half})
    out=pd.DataFrame(rows);out.to_csv(OUT/'ablation_paired_seed_ci.csv',index=False);print(out.round(5).to_string(index=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['bootstrap','ablation'],required=True)
    ap.add_argument('--replicates',type=int,default=500);ap.add_argument('--blocks',type=lambda s:list(map(int,s.split(','))),default=[5,10,20])
    ap.add_argument('--seed',type=int,default=20261028);ap.add_argument('--input',default='predictions_h5.csv')
    ap.add_argument('--output',default='paired_block_bootstrap_sensitivity_h5.csv');args=ap.parse_args()
    bootstrap(args) if args.mode=='bootstrap' else ablation_ci()
if __name__=='__main__':main()

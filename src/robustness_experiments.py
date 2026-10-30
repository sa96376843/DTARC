"""Additional experiments requested in technical review; no test-set retuning."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef,roc_auc_score,brier_score_loss
from experiment import (ROOT,DATA,OUT,initial_fit,run_online,static_predictions,
                        evaluate,predict,transform,purge_label_boundary)

INFO=json.loads((DATA/'dataset_summary.json').read_text());FEATURES=INFO['features']
CFG=json.loads((OUT/'best_config.json').read_text())['best']
SEEDS=[11,22,33,44,55]

def load(path,market):
    d=pd.read_csv(DATA/path,parse_dates=['date']);d['market']=market
    return d

def base_data(name):
    if name=='primary':return {m:load(f'{m.lower()}_processed.csv',m) for m in ['CN','TH']}
    if name=='loo':return {m:load(f'{m.lower()}_processed_loo.csv',m) for m in ['CN','TH']}
    if name=='exec':
        out={}
        for m in ['CN','TH']:
            d=load(f'{m.lower()}_processed_exec.csv',m)
            d['y_5']=d.y_exec5;d['fwd_ret_5']=d.fwd_ret_exec5;out[m]=d
        return out
    return {'CN':load(f'cn_processed_{name}.csv','CN')}

def run_suite(name,methods,filename,seeds=SEEDS):
    data=base_data(name);preds=[];events=[];runtime=[]
    for m,d in data.items():
        for seed in seeds:
            base=initial_fit(d[d.split=='train'].dropna(subset=['y_5']),FEATURES,5,seed,
                             purge_extra=1 if name=='exec' else 0)
            for method in methods:
                if method=='static':p,e,r=static_predictions(method,d,base,FEATURES,5,seed,'test')
                else:
                    # The no-provenance ablation still uses the ADWIN detector;
                    # only its replay quotas are replaced by uniform sampling.
                    runner_method = 'adwinDTARC' if method=='DTARC_no_provenance' else method
                    p,e,r=run_online(runner_method,d,base,FEATURES,5,seed,CFG,'test',
                                     variant='no_provenance' if method=='DTARC_no_provenance' else 'full',
                                     delay_batches=2 if name=='exec' else None)
                    if method=='DTARC_no_provenance':p['method']=method
                p['robustness']=name;preds.append(p)
                if len(e):events.append(e.assign(market=m,seed=seed,method=method,h=5))
                runtime.append(r|{'market':m,'seed':seed,'method':method,'h':5})
            print(name,m,seed,flush=True)
    pred=pd.concat(preds,ignore_index=True);pred.to_csv(OUT/f'predictions_{filename}.csv',index=False)
    evaluate(pred).to_csv(OUT/f'metrics_{filename}.csv',index=False)
    pd.DataFrame(runtime).to_csv(OUT/f'runtime_{filename}.csv',index=False)
    if events:pd.concat(events).to_csv(OUT/f'drift_events_{filename}.csv',index=False)
    print(evaluate(pred).groupby(['market','method'])[['mcc','auc','brier']].mean().round(5))

def mechanism():
    data=base_data('primary');settings=[]
    for factor,values in [('adwin_delta',[.001,.05,.2,.5]),('abrupt_threshold',[.25,.5,1.]),('prox',[0.,.0001,.001,.01])]:
        for value in values:settings.append((factor,value,CFG|{factor:value}))
    preds=[];events=[];runtime=[]
    for m,d in data.items():
        for seed in SEEDS:
            base=initial_fit(d[d.split=='train'].dropna(subset=['y_5']),FEATURES,5,seed)
            for factor,value,cfg in settings:
                p,e,r=run_online('adwinDTARC',d,base,FEATURES,5,seed,cfg,'test')
                p['factor']=factor;p['value']=value;preds.append(p)
                if len(e):events.append(e.assign(market=m,seed=seed,method='adwinDTARC',h=5,factor=factor,value=value))
                runtime.append(r|{'market':m,'seed':seed,'method':'adwinDTARC','h':5,'factor':factor,'value':value})
            print('mechanism',m,seed,flush=True)
    pred=pd.concat(preds,ignore_index=True);pred.to_csv(OUT/'predictions_mechanism_h5.csv',index=False)
    from sklearn.metrics import accuracy_score,f1_score
    rows=[]
    for keys,g in pred.groupby(['market','split','h','method','seed','factor','value']):
        y,p=g.y.values,g.p.values;yh=(p>=.5).astype(int)
        rows.append(dict(zip(['market','split','h','method','seed','factor','value'],keys))|{
            'n':len(g),'accuracy':accuracy_score(y,yh),'macro_f1':f1_score(y,yh,average='macro'),
            'mcc':matthews_corrcoef(y,yh),'auc':roc_auc_score(y,p),'brier':brier_score_loss(y,p)})
    pd.DataFrame(rows).to_csv(OUT/'metrics_mechanism_h5.csv',index=False)
    pd.DataFrame(runtime).to_csv(OUT/'runtime_mechanism_h5.csv',index=False)
    pd.concat(events,ignore_index=True).to_csv(OUT/'drift_events_mechanism_h5.csv',index=False)

def protocol_degradation():
    d=load('cn_processed.csv','CN');h=5;seed=42
    train=d[d.split=='train'].dropna(subset=['y_5']);preds=[]
    # Correct and no-purge share the proper delayed online stream.
    for name,purge,premature in [('correct',True,False),('no_purge',False,False),('premature_labels',True,True)]:
        base=initial_fit(train,FEATURES,h,seed,purge_boundary=purge)
        p,e,r=run_online('adwinDTARC',d,base,FEATURES,h,seed,CFG,'validation',allow_premature=premature)
        p['method']=name;preds.append(p)
    # Random row split over 2023--2025: deliberately invalid for temporal deployment.
    rng=np.random.default_rng(seed);pre=d[d.split!='test'].dropna(subset=['y_5']).copy()
    hold=rng.random(len(pre))<.2;tr=pre.loc[~hold];te=pre.loc[hold].copy()
    base=initial_fit(tr,FEATURES,h,seed,purge_boundary=False)
    probs=predict(base['model'],transform(te,base,FEATURES))
    p=te[['date','code','y_5','fwd_ret_5']].rename(columns={'y_5':'y','fwd_ret_5':'fwd_ret'})
    p['p']=probs;p['method']='random_row_split';p['market']='CN';p['seed']=seed;p['split']='validation';p['h']=h;preds.append(p)
    pred=pd.concat(preds,ignore_index=True);pred.to_csv(OUT/'predictions_protocol_degradation.csv',index=False)
    evaluate(pred).to_csv(OUT/'metrics_protocol_degradation.csv',index=False)
    print(evaluate(pred)[['method','accuracy','mcc','auc','brier']])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,
        choices=['adjustment','loo','exec','matched','mechanism','protocol']);args=ap.parse_args()
    if args.mode=='adjustment':
        for kind in ['sina_qianfuquan','sina_houfuquan']:
            run_suite(kind,['static','periodicER','adwinER','adwinDTARC'],kind)
    elif args.mode=='loo':run_suite('loo',['static','adwinER','adwinDTARC'],'loo')
    elif args.mode=='exec':run_suite('exec',['static','periodicER','adwinER','adwinDTARC'],'execution_aligned')
    elif args.mode=='matched':run_suite('primary',['adwinERstrat','DTARC_no_provenance'],'matched_replay')
    elif args.mode=='mechanism':mechanism()
    else:protocol_degradation()

if __name__=='__main__':main()

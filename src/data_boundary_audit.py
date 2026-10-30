"""Audit 2025/2026 boundary distributions for stitched and single-source panels."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'

def psi(a,b,bins=10):
    a=pd.Series(a).dropna().values;b=pd.Series(b).dropna().values
    if len(a)<10 or len(b)<10:return np.nan
    edges=np.unique(np.quantile(a,np.linspace(0,1,bins+1)))
    if len(edges)<3:return 0.
    edges[0]=-np.inf;edges[-1]=np.inf
    pa=np.clip(np.histogram(a,edges)[0]/len(a),1e-6,None);pb=np.clip(np.histogram(b,edges)[0]/len(b),1e-6,None)
    return float(np.sum((pb-pa)*np.log(pb/pa)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='cn_raw_baostock.csv');ap.add_argument('--suffix',default='stitched');args=ap.parse_args()
    d=pd.read_csv(ROOT/'data'/args.input,parse_dates=['date']).sort_values(['code','date']).copy()
    d['ret']=d.groupby('code').close.pct_change();d['amihud']=d.ret.abs()/(d.amount/1e8).replace(0,np.nan)
    rows=[];windows=[]
    for code,g in d.groupby('code'):
        g=g.sort_values('date').reset_index(drop=True);first=g.index[g.date.dt.year==2026][0]
        pre=g.iloc[max(0,first-20):first];post=g.iloc[first:first+20]
        boundary=g.iloc[max(0,first-20):first+20].copy();boundary['window_position']=np.arange(len(boundary))-len(pre)
        boundary['source_segment']=np.where(boundary.date.dt.year<2026,'pre_boundary','post_boundary');windows.append(boundary)
        rows.append({'code':code,'last_2025_date':pre.date.max(),'first_2026_date':post.date.min(),'boundary_return':g.loc[first,'ret'],
            'pre_return_mean':pre.ret.mean(),'post_return_mean':post.ret.mean(),'pre_return_sd':pre.ret.std(),'post_return_sd':post.ret.std(),
            'return_psi':psi(pre.ret,post.ret),'pre_volume_mean':pre.volume.mean(),'post_volume_mean':post.volume.mean(),
            'volume_log_ratio':np.log(post.volume.mean()/pre.volume.mean()),'pre_amihud_median':pre.amihud.median(),
            'post_amihud_median':post.amihud.median(),'pre_n':len(pre),'post_n':len(post)})
    audit=pd.DataFrame(rows);audit.to_csv(OUT/f'boundary_audit_{args.suffix}.csv',index=False)
    win=pd.concat(windows,ignore_index=True);win.to_csv(OUT/f'boundary_windows_{args.suffix}.csv',index=False)
    extreme=win.loc[win.ret.abs().nlargest(20).index,['date','code','ret','volume','amount','window_position']]
    extreme.sort_values('ret',key=lambda s:s.abs(),ascending=False).to_csv(OUT/f'boundary_extreme_returns_{args.suffix}.csv',index=False)
    summary=pd.DataFrame([{'n_stocks':len(audit),'median_return_psi':audit.return_psi.median(),'max_return_psi':audit.return_psi.max(),
        'median_abs_boundary_return':audit.boundary_return.abs().median(),'max_abs_boundary_return':audit.boundary_return.abs().max(),
        'median_volume_log_ratio':audit.volume_log_ratio.median()}])
    summary.to_csv(OUT/f'boundary_summary_{args.suffix}.csv',index=False)
    print(summary.round(5).to_string(index=False));print(audit.round(5).to_string(index=False))
if __name__=='__main__':main()

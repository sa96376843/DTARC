"""Prepare single-source, leave-one-out, and execution-aligned audit datasets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from preprocess import per_stock, add_market_context

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'
INFO=json.loads((DATA/'dataset_summary.json').read_text());FEATURES=INFO['features']

def finish(d):
    d=d[d.date>='2023-12-11'].copy()
    d['split']=np.select([d.date<='2024-12-31',d.date<='2025-12-31'],['train','validation'],default='test')
    return d

def refresh_primary_summary():
    """Describe the final on-disk primary panels, after source replacement."""
    summary_path=DATA/'dataset_summary.json'
    info=json.loads(summary_path.read_text())
    for market in ['CN','TH']:
        d=pd.read_csv(DATA/f'{market.lower()}_processed.csv',parse_dates=['date'],float_precision='round_trip')
        info['markets'][market]={
            'stocks':int(d.code.nunique()),
            'start':str(d.date.min().date()),
            'end':str(d.date.max().date()),
            'observations':int(len(d)),
            'features':len(info['features']),
            'splits':{k:int(v) for k,v in d.split.value_counts().items()},
            'up_rate_h5':float(d.y_5.mean()),
        }
    summary_path.write_text(json.dumps(info,indent=2))

def main():
    for kind in ['qianfuquan','houfuquan']:
        raw=pd.read_csv(DATA/f'cn_raw_sina_{kind}.csv',parse_dates=['date'])
        d=add_market_context(pd.concat([per_stock(g) for _,g in raw.groupby('code')],ignore_index=True))
        d['market']='CN';processed=finish(d)
        processed.to_csv(DATA/f'cn_processed_sina_{kind}.csv',index=False)
        if kind=='qianfuquan':
            # Primary CN panel: one vendor and one adjustment convention for
            # the complete train/validation/test period. The former stitched
            # panel is retained only as a failed-protocol audit artifact.
            processed.to_csv(DATA/'cn_processed.csv',index=False)
    for market in ['CN','TH']:
        d=pd.read_csv(DATA/f'{market.lower()}_processed.csv',parse_dates=['date'],float_precision='round_trip')
        # Leave-one-out same-date context: the stock's own return is excluded.
        add_market_context(d, leave_one_out=True).to_csv(DATA/f'{market.lower()}_processed_loo.csv',index=False)
        # Execution-aligned target: enter at t+1 close and exit at t+6 close.
        d=d.sort_values(['code','date']).copy()
        g=d.groupby('code')['close']
        d['fwd_ret_exec5']=g.shift(-6)/g.shift(-1)-1
        d['label_available_5']=d.groupby('code')['date'].shift(-6)
        d['y_exec5']=(d.fwd_ret_exec5>0).astype(float)
        d.loc[d.fwd_ret_exec5.isna(),'y_exec5']=np.nan
        d.to_csv(DATA/f'{market.lower()}_processed_exec.csv',index=False)
    refresh_primary_summary()
    print('robustness datasets written')

if __name__=='__main__':main()

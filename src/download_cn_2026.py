"""Append 2026 A-share bars from Sina raw K-lines and qianfuquan series.

The ratio between Sina's adjusted and raw close is applied to all OHLC fields.
This preserves within-day price geometry while removing corporate-action jumps.
"""
from pathlib import Path
import json, re, subprocess, time
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'
CODES=['sh.600519','sh.600036','sh.601318','sh.600030','sh.601166','sh.600276',
       'sz.000001','sz.000333','sz.000651','sz.002415','sz.300750','sz.002594']

def curl(url):
    return subprocess.check_output(['curl','--retry','4','--retry-all-errors','--max-time','90','-s',
        '-A','Mozilla/5.0','-e','https://finance.sina.com.cn/',url])

def main():
    old=pd.read_csv(DATA/'cn_raw_baostock.csv')
    old=old[old.date<='2025-12-31'].copy(); frames=[]
    for code in CODES:
        symbol=code.replace('.','')
        u=(f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
           f'CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen=200')
        bars=json.loads(curl(u).decode('utf-8-sig'))
        d=pd.DataFrame(bars).rename(columns={'day':'date'})
        d=d[(d.date>='2026-01-01')&(d.date<='2026-09-30')].copy()
        q=curl(f'https://finance.sina.com.cn/realstock/company/{symbol}/qianfuquan.js').decode('utf-8-sig')
        adj=dict((a.replace('_','-'),float(b)) for a,b in re.findall(r'_(\d{4}_\d{2}_\d{2}):"([0-9.]+)"',q))
        d['adj_close']=d.date.map(adj)
        if d.adj_close.isna().any(): raise RuntimeError(f'Missing adjusted values: {code}')
        for c in ['open','high','low','close','volume']: d[c]=pd.to_numeric(d[c])
        ratio=d.adj_close/d.close
        for c in ['open','high','low','close']: d[c]=d[c]*ratio
        d['code']=code; d['preclose']=d.close.shift(1); d['amount']=d.close*d.volume
        d['turn']=None; d['adjustflag']='sina_qfq_ratio'
        frames.append(d[old.columns]); print(code,len(d),d.date.min(),d.date.max()); time.sleep(.4)
    out=pd.concat([old]+frames,ignore_index=True)
    out.to_csv(DATA/'cn_raw_baostock.csv',index=False)
    print('combined',out.shape,out.date.min(),out.date.max())

if __name__=='__main__': main()

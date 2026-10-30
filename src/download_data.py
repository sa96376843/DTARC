"""Download and archive the real daily OHLCV data used in the study."""
from pathlib import Path
import time, json, hashlib
import requests
import pandas as pd
import baostock as bs

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
START,END='2016-01-01','2026-09-30'
CN=['sh.600519','sh.600036','sh.601318','sh.600030','sh.601166','sh.600276',
    'sz.000001','sz.000333','sz.000651','sz.002415','sz.300750','sz.002594']
TH=['PTT','PTTEP','PTTGC','KBANK','SCB','BBL','KTB','ADVANC','AOT','CPALL','BDMS','SCC']

def download_cn():
    lg=bs.login()
    if lg.error_code!='0': raise RuntimeError(lg.error_msg)
    frames=[]
    fields='date,code,open,high,low,close,preclose,volume,amount,turn,adjustflag'
    for code in CN:
        rs=bs.query_history_k_data_plus(code,fields,start_date=START,end_date=END,
                                       frequency='d',adjustflag='2')
        if rs.error_code!='0': raise RuntimeError((code,rs.error_msg))
        rows=[]
        while rs.next(): rows.append(rs.get_row_data())
        d=pd.DataFrame(rows,columns=rs.fields)
        frames.append(d); print(code,len(d))
    bs.logout()
    out=pd.concat(frames,ignore_index=True)
    out.to_csv(DATA/'cn_raw_baostock.csv',index=False)

def download_th():
    """SiamChart's public UDF endpoint currently serves a rolling history."""
    import subprocess
    now=int(time.time()); p1=now-360*86400
    frames=[]
    for symbol in TH:
        url=(f'https://siamchart.com/query/history?symbol={symbol}&resolution=D'
             f'&from={p1}&to={now}')
        command=['curl','--retry','5','--retry-all-errors','--max-time','90','-s',
            '-A','Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36',
            '-H','Accept: application/json, text/javascript, */*; q=0.01',
            '-H','X-Requested-With: XMLHttpRequest','-e',
            f'https://siamchart.com/stock-chart/{symbol}/',url]
        raw=None
        for attempt in range(4):
            try:
                raw=subprocess.check_output(command); break
            except subprocess.CalledProcessError:
                time.sleep(3*(attempt+1))
        if raw is None: raise RuntimeError(f'Failed to download {symbol}')
        x=json.loads(raw.decode('utf-8-sig'))
        d=pd.DataFrame({'date':pd.to_datetime(x['t'],unit='s',utc=True).date,
                        'code':symbol,'open':x['o'],'high':x['h'],'low':x['l'],
                        'close':x['c'],'volume':x.get('v',[None]*len(x['t']))})
        d=d[(pd.to_datetime(d['date'])>=pd.Timestamp(START)) & (pd.to_datetime(d['date'])<=pd.Timestamp(END))]
        frames.append(d); print(symbol,len(d),d['date'].min(),d['date'].max()); time.sleep(0.5)
    pd.concat(frames,ignore_index=True).to_csv(DATA/'th_raw_siamchart.csv',index=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--market',choices=['cn','th','all'],default='all')
    a=p.parse_args()
    if a.market in ['cn','all']: download_cn()
    if a.market in ['th','all']: download_th()
    for name in ['cn_raw_baostock.csv','th_raw_siamchart.csv']:
        f=DATA/name
        if f.exists(): print(name,hashlib.sha256(f.read_bytes()).hexdigest())

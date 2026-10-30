"""Robust parallel archival of SiamChart's public rolling daily history."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, subprocess, time
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'
SYMBOLS=['PTT','PTTEP','PTTGC','KBANK','SCB','BBL','KTB','ADVANC','AOT','CPALL','BDMS','SCC']

def fetch(symbol):
    now=int(time.time()); start=now-360*86400
    url=f'https://siamchart.com/query/history?symbol={symbol}&resolution=D&from={start}&to={now}'
    cmd=['curl','--retry','2','--retry-all-errors','--max-time','60','-s',
         '-A','Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36',
         '-H','Accept: application/json, text/javascript, */*; q=0.01',
         '-H','X-Requested-With: XMLHttpRequest','-e',f'https://siamchart.com/stock-chart/{symbol}/',url]
    raw=subprocess.check_output(cmd)
    x=json.loads(raw.decode('utf-8-sig'))
    d=pd.DataFrame({'date':pd.to_datetime(x['t'],unit='s',utc=True).date,'code':symbol,
                    'open':x['o'],'high':x['h'],'low':x['l'],'close':x['c'],
                    'volume':x.get('v',[None]*len(x['t']))})
    d=d[(pd.to_datetime(d.date)>='2023-11-13')&(pd.to_datetime(d.date)<='2026-09-30')]
    d.to_csv(DATA/f'th_{symbol}.csv',index=False)
    return symbol,d

def main():
    frames=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        jobs={ex.submit(fetch,s):s for s in SYMBOLS}
        for job in as_completed(jobs):
            s,d=job.result(); frames.append(d); print(s,len(d),d.date.min(),d.date.max(),flush=True)
    out=pd.concat(frames,ignore_index=True).sort_values(['code','date'])
    out.to_csv(DATA/'th_raw_siamchart.csv',index=False)
    print('combined',out.shape,out.date.min(),out.date.max())
if __name__=='__main__': main()

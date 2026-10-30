"""Download a single-source Sina panel with both adjustment conventions for audit."""
from pathlib import Path
import json,re,subprocess,time
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'
CODES=['sh.600519','sh.600036','sh.601318','sh.600030','sh.601166','sh.600276',
       'sz.000001','sz.000333','sz.000651','sz.002415','sz.300750','sz.002594']

def curl(url):
    return subprocess.check_output(['curl','--retry','3','--retry-all-errors','--max-time','60','-s',
        '-A','Mozilla/5.0','-e','https://finance.sina.com.cn/',url])

def main():
    for kind in ['qianfuquan','houfuquan']:
        frames=[]
        for code in CODES:
            symbol=code.replace('.','')
            u=(f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
               f'CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen=1023')
            bars=json.loads(curl(u).decode('utf-8-sig'))
            d=pd.DataFrame(bars).rename(columns={'day':'date'})
            d=d[(d.date>='2023-11-13')&(d.date<='2026-09-30')].copy()
            raw=curl(f'https://finance.sina.com.cn/realstock/company/{symbol}/{kind}.js').decode('utf-8-sig')
            adj=dict((a.replace('_','-'),float(b)) for a,b in re.findall(r'_(\d{4}_\d{2}_\d{2}):"([0-9.]+)"',raw))
            d['adj_close']=d.date.map(adj)
            if d.adj_close.isna().any():raise RuntimeError(f'Missing adjusted values {code} {kind}')
            for c in ['open','high','low','close','volume']:d[c]=pd.to_numeric(d[c])
            d['raw_close']=d.close
            ratio=d.adj_close/d.close
            for c in ['open','high','low','close']:d[c]=d[c]*ratio
            d['code']=code;d['preclose']=d.close.shift(1);d['amount']=d.close*d.volume
            d['turn']=None;d['adjustflag']=f'sina_{kind}'
            frames.append(d[['date','code','open','high','low','close','preclose','volume','amount','turn','adjustflag','raw_close']])
            print(kind,code,len(d),flush=True);time.sleep(.2)
        pd.concat(frames,ignore_index=True).to_csv(DATA/f'cn_raw_sina_{kind}.csv',index=False)

if __name__=='__main__':main()

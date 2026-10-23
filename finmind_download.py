#!/usr/bin/env python3
"""FinMind 逐檔資料下載與快取：合法節流、斷點續跑。"""
import argparse
import os
import re
import time
from pathlib import Path
import pandas as pd
import requests
BASE="https://api.finmindtrade.com/api/v4/data"

def get_json(params,token=None,timeout=45,retry=4):
    head={"User-Agent":"army-ma-research/1.0"}
    if token: head["Authorization"]=f"Bearer {token}"
    for attempt in range(retry):
        try:
            r=requests.get(BASE,params=params,headers=head,timeout=timeout)
            if r.status_code==429:
                time.sleep(min(180,30*(attempt+1)))
                continue
            r.raise_for_status()
            payload=r.json()
            if payload.get("status")!=200:
                raise ValueError(f"FinMind status={payload.get('status')}: {payload.get('msg')}")
            return payload.get("data",[])
        except (requests.RequestException,ValueError) as e:
            if attempt==retry-1: raise
            print(f"重試：{e}",flush=True)
            time.sleep(3*(attempt+1))
    raise RuntimeError("API 已超過重試限制")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("data"))
    p.add_argument("--start",default="2017-01-01")
    p.add_argument("--end",default="2026-10-22")
    p.add_argument("--max-stocks",type=int,default=0)
    p.add_argument("--stocks",nargs="*",help="指定代號，省略則取市場股票清單")
    p.add_argument("--sleep",type=float,default=15.0)
    p.add_argument("--refresh",action="store_true")
    p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--shard-count",type=int,default=1)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.shard_count<1 or not 0<=args.shard_index<args.shard_count:
        p.error("分片參數錯誤：0 <= shard-index < shard-count")
    token=os.environ.get("FINMIND_TOKEN")
    if args.stocks:
        ids=sorted(set(args.stocks))
    else:
        info=get_json({"dataset":"TaiwanStockInfo"},token)
        ids=sorted(pd.DataFrame(info).stock_id.astype(str).unique().tolist())
    ids=[i for i in ids if re.fullmatch(r"[1-9][0-9]{3}",i)]
    if args.max_stocks: ids=ids[:args.max_stocks]
    ids=ids[args.shard_index::args.shard_count]
    print(f"分片 {args.shard_index+1}/{args.shard_count}，股票數 {len(ids)}",flush=True)
    fail=[]
    done=0
    skip=0
    for j,symbol in enumerate(ids,1):
        f=args.out/f"{symbol}.csv"
        if f.exists() and f.stat().st_size>1000 and not args.refresh:
            skip+=1
            continue
        try:
            dat=get_json({"dataset":"TaiwanStockPriceAdj","data_id":symbol,"start_date":args.start,"end_date":args.end},token)
            if dat:
                pd.DataFrame(dat).to_csv(f,index=False,encoding="utf-8-sig")
                done+=1
        except Exception as e:
            fail.append({"stock":symbol,"error":str(e)})
            print(f"{j}/{len(ids)} {symbol} 失敗：{e}",flush=True)
        if j%20==0:
            print(f"進度 {j}/{len(ids)}，新檔 {done}，快取 {skip}",flush=True)
        time.sleep(max(0,args.sleep))
    if fail: pd.DataFrame(fail).to_csv(args.out/"download_errors.csv",index=False,encoding="utf-8-sig")
    print(f"完成：新檔 {done}，已有 {skip}，失敗 {len(fail)}")
    if fail: raise SystemExit(f"{len(fail)} 檔失敗，請檢查 download_errors.csv，續跑可保留已下載快取")

if __name__=="__main__":
    main()

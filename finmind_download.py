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
            try:
                payload=r.json()
            except ValueError:
                payload={}
            message=str(payload.get("msg") or payload.get("detail") or r.text[:400])
            if r.status_code in (400,401,403,404):
                raise ValueError(f"FinMind 請求遭拒絕 HTTP {r.status_code}: {message}")
            if r.status_code in (402,429) or r.status_code>=500:
                raise requests.HTTPError(f"FinMind 暫時不可用 HTTP {r.status_code}: {message}",response=r)
            r.raise_for_status()
            if payload.get("status")!=200:
                if payload.get("status") in (400,401,403):
                    raise ValueError(f"FinMind 權限或參數錯誤：{message}")
                raise requests.HTTPError(f"FinMind status={payload.get('status')}: {message}")
            return payload.get("data",[])
        except ValueError:
            raise
        except requests.RequestException as e:
            if attempt==retry-1: raise
            print(f"暫時錯誤，準備重試 {attempt+1}/{retry}: {e}",flush=True)
            time.sleep(min(120,15*(attempt+1)))
    raise RuntimeError("API 已超過重試限制")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("data"))
    p.add_argument("--start",default="2017-01-01")
    p.add_argument("--end",default="2026-10-22")
    p.add_argument("--max-stocks",type=int,default=0)
    p.add_argument("--stocks",nargs="*",help="指定代號，省略則取市場股票清單")
    p.add_argument("--sleep",type=float,default=15.0)
    p.add_argument("--dataset",choices=["TaiwanStockPrice","TaiwanStockPriceAdj"],default="TaiwanStockPrice",help="預設免費原始價格；還原股價需要對應會員權限")
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
    print(f"使用資料集 {args.dataset}" + ("（未還原除權息）" if args.dataset=="TaiwanStockPrice" else "（還原股價）"),flush=True)
    fail=[]
    done=0
    skip=0
    for j,symbol in enumerate(ids,1):
        f=args.out/f"{symbol}.csv"
        if f.exists() and f.stat().st_size>1000 and not args.refresh:
            skip+=1
            continue
        try:
            dat=get_json({"dataset":args.dataset,"data_id":symbol,"start_date":args.start,"end_date":args.end},token)
            if dat:
                pd.DataFrame(dat).to_csv(f,index=False,encoding="utf-8-sig")
                done+=1
            else:
                raise ValueError(f"{symbol} 沒有歷史日K，不能當作已下載")
        except Exception as e:
            fail.append({"stock":symbol,"error":str(e)})
            print(f"{j}/{len(ids)} {symbol} 失敗：{e}",flush=True)
        if j%20==0:
            print(f"進度 {j}/{len(ids)}，新檔 {done}，快取 {skip}",flush=True)
        time.sleep(max(0,args.sleep))
    import json
    (args.out/f"data_source_shard_{args.shard_index}.json").write_text(json.dumps({"dataset":args.dataset,"adjusted":args.dataset=="TaiwanStockPriceAdj","start":args.start,"end":args.end,"requested_count":len(ids),"downloaded_count":done,"cached_count":skip,"failed_count":len(fail),"warning":"原始價格未還原除權息；拆股等公司行動可能扭曲訊號與報酬，請勿直接據此宣稱最佳軍線"},ensure_ascii=False,indent=2),encoding="utf-8")
    if fail: pd.DataFrame(fail).to_csv(args.out/"download_errors.csv",index=False,encoding="utf-8-sig")
    print(f"完成：新檔 {done}，已有 {skip}，失敗 {len(fail)}")
    if fail: raise SystemExit(f"{len(fail)} 檔失敗，請檢查 download_errors.csv，續跑可保留已下載快取")

if __name__=="__main__":
    main()

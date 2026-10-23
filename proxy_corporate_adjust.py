#!/usr/bin/env python3
"""使用免費參考事件做回溯比例還原的研究性版本。不是官方付費還原日 K。"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from finmind_download import get_json

SHARED=[
    ("TaiwanStockSplitPrice","before_price","after_price",1),
    ("TaiwanStockParValueChange","before_close","after_ref_close",2),
]
BY_STOCK=("TaiwanStockDividendResult","before_price","reference_price",3)

def clean_events(frame:pd.DataFrame):
    if frame.empty: return frame
    x=frame.copy()
    x["date"]=pd.to_datetime(x.date,errors="coerce")
    x["before"]=pd.to_numeric(x.before,errors="coerce")
    x["after"]=pd.to_numeric(x.after,errors="coerce")
    x["factor"]=x["after"]/x.before.replace(0,np.nan)
    x=x.dropna(subset=["date","factor"])
    x=x[(x.factor>0.02)&(x.factor<=3.0)]
    ambiguous=[]
    for (stock,date),g in x.groupby(["stock","date"]):
        if len(g)>1 and (g.factor.max()/g.factor.min()>1.02):
            ambiguous.append((stock,date))
    if ambiguous:
        raise ValueError(f"同一日有不同公司行動調整比率：{ambiguous[:10]}；必須人工核對，避免錯算")
    # 同一分割事件常同時出現在「分割參考價」與「面額變更」；只套用一次。
    return x.sort_values("priority",ascending=False).drop_duplicates(["stock","date"]).sort_values(["stock","date"])

def adjusted_close(df:pd.DataFrame,events:pd.DataFrame,stock:str):
    d=df.copy()
    dates=pd.to_datetime(d["date"].astype(str).str.slice(0,10))
    ratio=np.ones(len(d),dtype=float)
    my=events[events.stock==str(stock)] if len(events) else events
    for e in my.itertuples():
        ratio[dates<e.date]*=float(e.factor)
    d["adj_close"]=pd.to_numeric(d.close,errors="coerce")*ratio
    return d, float(ratio.min()),float(ratio.max())

def run(input_dir:Path,output_dir:Path,ids:list[str],start:str,end:str,sleep_s:float):
    output_dir.mkdir(parents=True,exist_ok=True)
    token=os.environ.get("FINMIND_TOKEN")
    ev=[];errors=[];report=[]
    for name,b,a,priority in SHARED:
        try:
            values=get_json({"dataset":name,"start_date":start,"end_date":end},token)
            for r in values:
                if str(r.get("stock_id")) in ids:
                    ev.append({"stock":str(r["stock_id"]),"date":r.get("date"),"before":r.get(b),
                               "after":r.get(a),"dataset":name,"priority":priority})
        except Exception as exc:
            errors.append({"dataset":name,"stock":"ALL","error":str(exc)})
        time.sleep(sleep_s)
    name,b,a,priority=BY_STOCK
    failures=set()
    for i,stock in enumerate(ids,1):
        try:
            values=get_json({"dataset":name,"data_id":stock,"start_date":start,"end_date":end},token)
            for r in values:
                ev.append({"stock":stock,"date":r.get("date"),"before":r.get(b),
                           "after":r.get(a),"dataset":name,"priority":priority})
        except Exception as exc:
            errors.append({"dataset":name,"stock":stock,"error":str(exc)})
            failures.add(stock)
        if i%10==0: print(f"公司行動資料進度 {i}/{len(ids)}，失敗股票 {len(failures)}",flush=True)
        time.sleep(sleep_s)
    raw_events=pd.DataFrame(ev,columns=["stock","date","before","after","dataset","priority"])
    events=clean_events(raw_events)
    events.to_csv(output_dir/"applied_corporate_events.csv",index=False,encoding="utf-8-sig")
    ok=0
    for symbol in ids:
        src=input_dir/f"{symbol}.csv"
        if symbol in failures or not src.exists():
            report.append({"stock":symbol,"status":"missing_events_or_prices"})
            continue
        try:
            frame=pd.read_csv(src)
            if "adj_close" in frame.columns: raise ValueError("輸入已含 adj_close，禁止重複還原")
            after,minimum,maximum=adjusted_close(frame,events,symbol)
            if after["adj_close"].isna().any() or (after["adj_close"]<=0).any():
                raise ValueError("調整價含無效值")
            after.to_csv(output_dir/f"{symbol}.csv",index=False,encoding="utf-8-sig")
            report.append({"stock":symbol,"status":"adjusted_proxy","events":len(events[events.stock==symbol]),
                           "min_factor":minimum,"max_factor":maximum})
            ok+=1
        except Exception as exc:
            errors.append({"dataset":"price_adjustment","stock":symbol,"error":str(exc)})
            report.append({"stock":symbol,"status":"adjustment_failed"})
    pd.DataFrame(report).to_csv(output_dir/"adjustment_coverage.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(errors,columns=["dataset","stock","error"]).to_csv(output_dir/"adjustment_errors.csv",index=False,encoding="utf-8-sig")
    manifest={"requested":len(ids),"adjusted":ok,"action_events":len(events),"errors":len(errors),
        "adjustment":"公開公司行動比率回溯近似值；不是 FinMind 官方還原股價",
        "missing":"減資參考價端點在現有權限下會 HTTP 400，無法確認完整公司行動",
        "warning":"不能將此代理調整報酬宣稱為經審計的實盤績效"}
    (output_dir/"adjustment_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False),flush=True)
    if ok<len(ids): raise RuntimeError(f"有 {len(ids)-ok} 檔未能完成代理還原，請檢查 coverage/errors")
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data"))
    p.add_argument("--out",type=Path,default=Path("proxy_adjusted_data"))
    p.add_argument("--stocks",nargs="+",required=True)
    p.add_argument("--start",default="2017-01-01")
    p.add_argument("--end",default="2026-10-22")
    p.add_argument("--sleep",type=float,default=3)
    a=p.parse_args()
    run(a.data,a.out,a.stocks,a.start,a.end,a.sleep)

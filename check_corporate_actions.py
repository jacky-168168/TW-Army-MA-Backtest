#!/usr/bin/env python3
"""免費公開公司行動交叉核對；僅找疑點，不產生未驗證的還原價績效。"""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from finmind_download import get_json

SHARED={
    "TaiwanStockSplitPrice":("before_price","after_price"),
    "TaiwanStockParValueChange":("before_close","after_ref_close"),
    "TaiwanStockCapitalReductionReferencePrice":("ClosingPriceonTheLastTradingDay","PostReductionReferencePrice")
}
PER_STOCK={"TaiwanStockDividendResult":("before_price","reference_price")}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--extremes",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("audit-actions"))
    p.add_argument("--limit-stocks",type=int,default=30)
    p.add_argument("--start",default="2017-01-01")
    p.add_argument("--end",default="2026-10-22")
    p.add_argument("--sleep",type=float,default=3)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.extremes)
    x=raw[raw.period=="test"].copy()
    if x.empty: x=raw.copy()
    x["abs_return"]=x.return_pct.abs()
    stocks=x.groupby("stock").abs_return.max().sort_values(ascending=False).head(a.limit_stocks).index.astype(str).tolist()
    events=[];failure=[]
    def fetch(name,symbol=None):
        params={"dataset":name,"start_date":a.start,"end_date":a.end}
        if symbol is not None: params["data_id"]=symbol
        try:
            d=get_json(params)
            if not isinstance(d,list): raise ValueError("API 非列表型態")
            return d
        except Exception as exc:
            failure.append({"dataset":name,"stock":symbol,"error":str(exc)})
            return []
    for dataset,(b,after) in SHARED.items():
        rows=fetch(dataset)
        for r in rows:
            sym=str(r.get("stock_id",""))
            if sym not in stocks: continue
            events.append({"stock":sym,"date":r.get("date"),"dataset":dataset,
                           "before":r.get(b),"after_reference":r.get(after)})
        time.sleep(a.sleep)
    for sym in stocks:
        for dataset,(b,after) in PER_STOCK.items():
            for r in fetch(dataset,sym):
                events.append({"stock":sym,"date":r.get("date"),"dataset":dataset,
                               "before":r.get(b),"after_reference":r.get(after)})
            time.sleep(a.sleep)
    e=pd.DataFrame(events,columns=["stock","date","dataset","before","after_reference"])
    if len(e):
        e.date=pd.to_datetime(e.date,errors="coerce")
        e["before"]=pd.to_numeric(e.before,errors="coerce")
        e["after_reference"]=pd.to_numeric(e.after_reference,errors="coerce")
        e=e.dropna(subset=["date"]).sort_values(["stock","date","dataset"])
        e["reference_factor"]=e.after_reference/e.before.replace(0,float("nan"))
        e.to_csv(a.output/"reference_events.csv",index=False,encoding="utf-8-sig")
    related=[]
    for _,t in x.iterrows():
        q=e[(e.stock==str(t.stock))&(e.date>=pd.Timestamp(t.entry_date))&(e.date<=pd.Timestamp(t.exit_date))] if len(e) else e
        types="、".join(sorted(q.dataset.unique())) if len(q) else ""
        related.append({"stock":t.stock,"ma":t.ma,"entry_date":t.entry_date,"exit_date":t.exit_date,
                        "return_pct":t.return_pct,"events_in_holding":len(q),
                        "event_categories":types,"requires_price_audit":True})
    pd.DataFrame(related).to_csv(a.output/"extreme_trades_events.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(failure,columns=["dataset","stock","error"]).to_csv(a.output/"data_errors.csv",index=False,encoding="utf-8-sig")
    meta={"checked_test_extreme_stocks":len(stocks),"events_found":len(e),"api_failures":len(failure),
          "event_overlap_trades":sum(a["events_in_holding"]>0 for a in related),
          "limitations":["交叉核對結果無法單獨證明極端交易有效或錯誤",
                         "公司行動資料覆蓋不完整；未取得付費級完整還原日 K",
                         "即使無已知公司行動也要核查漲跌停、價格錯誤及調整日期"]}
    (a.output/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False),flush=True)
    if len(failure)>=len(stocks)+len(SHARED):
        raise RuntimeError("所有公司行動資料請求均失敗，不可宣稱審核通過")

if __name__=="__main__":
    main()

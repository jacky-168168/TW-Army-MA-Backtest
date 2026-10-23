#!/usr/bin/env python3
"""不使用單筆進出場勝率，獨立衡量軍線的二日反應與貼線反轉規律。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from army_ma import Config,PERIODS,read_stock_csv

EVENTS={"fake_down": "假跌破", "true_up":"真突破", "fake_up":"假突破", "true_down":"真跌破"}
def events_for_stock(stock,df,ma_len,cfg):
    z=df.reset_index(drop=True)
    ma=z.close.rolling(ma_len,min_periods=ma_len).mean()
    up1=z.close.shift(1).gt(ma.shift(1)) & z.close.shift(2).le(ma.shift(2))
    down1=z.close.shift(1).lt(ma.shift(1)) & z.close.shift(2).ge(ma.shift(2))
    typ=pd.Series("",index=z.index)
    typ.loc[down1 & z.low.ge(z.low.shift(1))]="fake_down"
    typ.loc[up1 & z.high.gt(z.high.shift(1))]="true_up"
    typ.loc[up1 & z.high.le(z.high.shift(1))]="fake_up"
    typ.loc[down1 & z.low.lt(z.low.shift(1))]="true_down"
    allowed=(z.turnover.rolling(20,min_periods=20).mean().shift(1)>=cfg.min_turnover) & ma.notna()
    # 事件研究不假設同日收盤進場；以次日開盤作為反應期的起始價格。
    zdate=z.date
    out=[]
    for h in (5,10):
        entry=z.open.shift(-1)
        exit_price=z.close.shift(-h)
        raw_ret=exit_price/entry-1
        # 僅是方向性價格反應，並非實際空單損益，不含融券費或漲跌停成交限制。
        direction=np.where(typ.isin(["fake_down","true_up"]),1,-1)
        directional=np.where(direction>0,raw_ret,-raw_ret)
        cost=2*cfg.fee+cfg.sell_tax+2*cfg.slippage
        net=100*(directional-cost)
        eligible=allowed & typ.ne("") & (entry>0) & (exit_price>0)
        for period,(start,end) in PERIODS.items():
            # 嚴禁使用驗證期價格當作訓練期事件的持有期結果。
            in_period=(zdate>=pd.Timestamp(start))&(zdate<=pd.Timestamp(end))
            exit_in_period=(zdate.shift(-h)<=pd.Timestamp(end))
            picks=np.flatnonzero(eligible&in_period&exit_in_period)
            for i in picks:
                out.append((stock,ma_len,period,str(typ.iloc[i]),h,float(net[i])))
    return out

def run(data:Path,output:Path,first:int=16,last:int=70):
    cfg=Config()
    output.mkdir(parents=True,exist_ok=True)
    out=[];valid=0
    for p in sorted(data.glob("*.csv")):
        if not (p.stem.isdigit() and len(p.stem)==4): continue
        try:
            df,src=read_stock_csv(p)
            if len(df)<130: continue
        except Exception as e:
            print(f"無法讀取 {p.name}: {e}",flush=True)
            continue
        valid+=1
        for m in range(first,last+1):
            out.extend(events_for_stock(p.stem,df,m,cfg))
    if not out: raise RuntimeError("沒有符合期間與流動性條件的二日法則事件")
    events=pd.DataFrame(out,columns=["stock","ma","period","event","hold_days","directional_net_pct"])
    agg=events.groupby(["ma","period","event","hold_days"]).directional_net_pct.agg(
        observations="size",win_rate=lambda x:100*(x>0).mean(),
        mean="mean",median="median").reset_index()
    agg["win_rate"]=agg.win_rate.round(2)
    agg[["mean","median"]]=agg[["mean","median"]].round(3)
    agg.to_csv(output/"two_day_event_statistics.csv",index=False,encoding="utf-8-sig")
    stocks=events.groupby(["ma","period","event","hold_days"]).stock.nunique().rename("unique_stocks").reset_index()
    agg=agg.merge(stocks,on=["ma","period","event","hold_days"])
    agg.to_csv(output/"two_day_event_by_stock_coverage.csv",index=False,encoding="utf-8-sig")
    print(f"共 {valid} 檔有足夠歷史資料，事件紀錄 {len(events)} 筆；依 MA／時段／類型／持有日輸出",flush=True)
    print(agg[(agg.period=="test")&(agg.ma.isin([18,20,25,50,55,61]))&(agg.hold_days==5)].to_string(index=False),flush=True)
    (output/"NOTICE.txt").write_text(
        "事件研究是第二日確認後，次日開盤到第5或第10個交易日收盤的方向性價格反應。\n"
        "不等同原本的停損交易回測、不含放空借券限制，採原始未還原價，結果不可直接用作投資。\n"
        "訓練／驗證／樣本外事件切分確保未使用其他期間的未來價格。\n",
        encoding="utf-8")
    return agg

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("event-study"))
    p.add_argument("--first-ma",type=int,default=16)
    p.add_argument("--last-ma",type=int,default=70)
    a=p.parse_args()
    if not (16<=a.first_ma<=a.last_ma<=70): raise ValueError("MA 範圍應在 16～70")
    run(a.data,a.output,a.first_ma,a.last_ma)

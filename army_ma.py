#!/usr/bin/env python3
"""台股高流動性共通軍線：SMA16–70／二日法則／樣本外驗證。"""
from __future__ import annotations
import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrap(fn): return fn
        return wrap

@dataclass(frozen=True)
class Config:
    min_price: float = 0.0
    min_turnover: float = 50_000_000.0
    turnover_days: int = 20
    min_history: int = 80
    fee: float = 0.001425
    sell_tax: float = 0.003
    slippage: float = 0.001
    max_loss: float = 0.04
    min_stop_distance: float = 0.01
    structure_stop: bool = True
    noise_filter: bool = True
    noise_len: int = 20
    noise_max: int = 6
    trend_filter: bool = True
    slope_len: int = 5
    breakout_entry: bool = True
    exit_mode: str = "original"
    min_trades_stock: int = 5
    min_win_rate: float = 0.55
    min_profit_factor: float = 1.10
    min_universe_stocks: int = 100
    min_period_trades: int = 300

PERIODS = {"train": ("2018-01-01", "2022-12-30"),
           "validation": ("2023-01-01", "2024-12-31"),
           "test": ("2025-01-01", "2026-10-22")}
EVENT_NAMES = {1:"假跌破買",2:"真突破買",3:"真跌破賣",4:"假突破賣",5:"4%／結構停損",6:"期末結算"}

def read_stock_csv(path: Path) -> tuple[pd.DataFrame, str]:
    raw = pd.read_csv(path, low_memory=False)
    cols = {c.lower().strip().replace(" ", "_"): c for c in raw.columns}
    def find(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    dcol = find("date","datetime")
    o,h,l,c = find("open"),find("high","max"),find("low","min"),find("close")
    if not all([dcol,o,h,l,c]):
        raise ValueError(f"{path.name}: 無法辨識 OHLC/日期欄位: {list(raw.columns)}")
    df = pd.DataFrame({"date":pd.to_datetime(raw[dcol].astype(str).str.slice(0,10),errors="coerce"),
                       "raw_close":pd.to_numeric(raw[c],errors="coerce"),
                       "open":pd.to_numeric(raw[o], errors="coerce"),
                       "high":pd.to_numeric(raw[h],errors="coerce"),
                       "low":pd.to_numeric(raw[l], errors="coerce"),
                       "close":pd.to_numeric(raw[c],errors="coerce")})
    amt,vol=find("trading_money","turnover_twd","amount","value"),find("volume","trading_volume")
    if amt:
        df["turnover"]=pd.to_numeric(raw[amt],errors="coerce")
        src="來源成交金額"
    elif vol:
        df["turnover"]=pd.to_numeric(raw[c],errors="coerce")*pd.to_numeric(raw[vol],errors="coerce")
        src="收盤價×成交股數估算"
    else:
        raise ValueError(f"{path.name}: 缺少成交金額或成交股數")
    adj=find("adj_close","adjclose")
    if adj:
        factor=pd.to_numeric(raw[adj],errors="coerce")/pd.to_numeric(raw[c],errors="coerce").replace(0,np.nan)
        for name in ["open","high","low","close"]: df[name]*=factor
        src+="；股價採 Adj Close 等比還原"
    else:
        src+="；需確認來源 OHLC 已還原"
    df=df.dropna(subset=["date","open","high","low","close","turnover"])
    df=df[(df[["open","high","low","close"]]>0).all(axis=1)&(df.turnover>=0)]
    df=df[(df.high>=df[["open","low","close"]].max(axis=1)-1e-7)&
          (df.low<=df[["open","high","close"]].min(axis=1)+1e-7)]
    df=df.sort_values("date").drop_duplicates("date",keep="last").reset_index(drop=True)
    return df,src

def signals(df:pd.DataFrame,ma_len:int,cfg:Config)->pd.DataFrame:
    z=df.copy()
    ma=z.close.rolling(ma_len,min_periods=ma_len).mean()
    cross_up=(z.close>ma)&(z.close.shift(1)<=ma.shift(1))
    cross_dn=(z.close<ma)&(z.close.shift(1)>=ma.shift(1))
    cross_cnt=(cross_up|cross_dn).astype(int).rolling(cfg.noise_len,min_periods=cfg.noise_len).sum()
    broken_dn=(z.close.shift(1)<ma.shift(1))&(z.close.shift(2)>=ma.shift(2))
    broken_up=(z.close.shift(1)>ma.shift(1))&(z.close.shift(2)<=ma.shift(2))
    false_dn=broken_dn&(z.low>=z.low.shift(1))
    true_dn=broken_dn&(z.low<z.low.shift(1))
    false_up=broken_up&(z.high<=z.high.shift(1))
    true_up=broken_up&(z.high>z.high.shift(1))
    liquid=z.turnover.rolling(cfg.turnover_days,min_periods=cfg.turnover_days).mean().shift(1)>=cfg.min_turnover
    allowed=(liquid&(z.raw_close>=cfg.min_price)&(z.index>=cfg.min_history)&ma.notna())
    if cfg.noise_filter: allowed &= cross_cnt.lt(cfg.noise_max)
    if cfg.trend_filter: allowed &= ma>ma.shift(cfg.slope_len)
    entry_a=allowed&false_dn
    entry_b=allowed&true_up&cfg.breakout_entry
    z["ma"]=ma
    z["liq"]=liquid.fillna(False)
    z["signal_entry"]=np.where(entry_a,1,np.where(entry_b,2,0)).astype(np.int8)
    z["signal_exit"]=np.where(true_dn,3,np.where(false_up&(cfg.exit_mode=="full"),4,0)).astype(np.int8)
    z["structure_low"]=z.low.rolling(3,min_periods=3).min()
    return z

@njit(cache=True)
def _fast_simulate(opens,lows,closes,structural_lows,entries,exits,begin,last,
                   fee,sell_tax,slippage,max_loss,min_stop_distance,structure_stop):
    trades=np.zeros((last-begin+2,9),dtype=np.float64)
    count=0
    cash=1.0;shares=0.0;entry_cash=0.0;entry_px=0.0;stop=0.0
    entry_idx=-1;entry_code=0;in_position=False
    pending_entry=0;pending_exit=0;signal_i=-1
    peak=1.0;max_dd=0.0
    for i in range(begin,last+1):
        op=opens[i];lo=lows[i];cl=closes[i]
        if in_position and pending_exit>0:
            px=op*(1-slippage)
            cash=shares*px*(1-fee-sell_tax)
            trades[count,0]=entry_idx;trades[count,1]=i
            trades[count,2]=entry_px;trades[count,3]=px
            trades[count,4]=entry_code;trades[count,5]=pending_exit
            trades[count,6]=entry_cash;trades[count,7]=cash;trades[count,8]=stop
            count+=1;shares=0.0;in_position=False
        if not in_position and pending_entry>0:
            px=op*(1+slippage)
            shares=cash/(px*(1+fee));entry_cash=cash;cash=0.0
            stop=px*(1-max_loss)
            if structure_stop and np.isfinite(structural_lows[signal_i]):
                stop=max(stop,structural_lows[signal_i])
            stop=min(stop,px*(1-min_stop_distance))
            entry_idx=i;entry_px=px;entry_code=pending_entry;in_position=True
        pending_entry=0;pending_exit=0
        if in_position and lo<=stop:
            px=min(op,stop)*(1-slippage)
            cash=shares*px*(1-fee-sell_tax)
            trades[count,0]=entry_idx;trades[count,1]=i
            trades[count,2]=entry_px;trades[count,3]=px
            trades[count,4]=entry_code;trades[count,5]=5
            trades[count,6]=entry_cash;trades[count,7]=cash;trades[count,8]=stop
            count+=1;shares=0.0;in_position=False
        equity=shares*cl*(1-fee-sell_tax) if in_position else cash
        peak=max(peak,equity);max_dd=min(max_dd,equity/peak-1)
        if i<last:
            if in_position: pending_exit=exits[i]
            else:
                pending_entry=entries[i];signal_i=i
    if in_position:
        px=closes[last]*(1-slippage)
        cash=shares*px*(1-fee-sell_tax)
        trades[count,0]=entry_idx;trades[count,1]=last
        trades[count,2]=entry_px;trades[count,3]=px
        trades[count,4]=entry_code;trades[count,5]=6
        trades[count,6]=entry_cash;trades[count,7]=cash;trades[count,8]=stop
        count+=1
    return trades[:count],max_dd

def backtest_one(stock:str,z:pd.DataFrame,period:str,cfg:Config)->tuple[list[dict],dict]:
    start,end=map(pd.Timestamp,PERIODS[period])
    sel=np.flatnonzero((z.date>=start)&(z.date<=end))
    if len(sel)<30:
        return [],{"stock":stock,"period":period,"bars":len(sel),"eligible_days":0,"trades":0,"max_drawdown_pct":np.nan}
    begin,last=int(sel[0]),int(sel[-1])
    eligible_days=int(z.liq.iloc[begin:last+1].sum())
    values,drawdown=_fast_simulate(z.open.to_numpy(dtype=np.float64),z.low.to_numpy(dtype=np.float64),
        z.close.to_numpy(dtype=np.float64),z.structure_low.to_numpy(dtype=np.float64),
        z.signal_entry.to_numpy(dtype=np.int64),z.signal_exit.to_numpy(dtype=np.int64),
        begin,last,cfg.fee,cfg.sell_tax,cfg.slippage,cfg.max_loss,cfg.min_stop_distance,cfg.structure_stop)
    dates=z.date.dt.strftime("%Y-%m-%d").to_numpy()
    rows=[]
    for v in values:
        x,y=int(v[0]),int(v[1]);entry_cash,exit_cash=v[6],v[7]
        rows.append({"stock":stock,"period":period,"entry_date":dates[x],"exit_date":dates[y],
                     "entry_price":round(v[2],5),"exit_price":round(v[3],5),
                     "entry_signal":EVENT_NAMES[int(v[4])],"exit_reason":EVENT_NAMES[int(v[5])],
                     "return_pct":round(100*(exit_cash/entry_cash-1),6),"stop_price":round(v[8],5)})
    return rows,{"stock":stock,"period":period,"bars":len(sel),"eligible_days":eligible_days,
                 "trades":len(rows),"max_drawdown_pct":100*drawdown}

def evaluate(trades:pd.DataFrame,stats:pd.DataFrame,ma:int,period:str,cfg:Config)->dict:
    d=trades[trades.period==period] if not trades.empty else pd.DataFrame()
    s=stats[stats.period==period] if not stats.empty else pd.DataFrame()
    stock_count=int((s.eligible_days>0).sum()) if not s.empty else 0
    used=int(d.stock.nunique()) if not d.empty else 0
    if d.empty:
        return {"ma":ma,"period":period,"stocks_eligible":stock_count,"stocks_traded":0,"trades":0,
                "win_rate_pct":np.nan,"avg_trade_pct":np.nan,"profit_factor":np.nan,
                "median_stock_return_pct":np.nan,"stocks_pass":0,"coverage_pct":0.0,
                "median_stock_max_dd_pct":np.nan}
    x=d.return_pct.astype(float);wins=x[x>0];losses=x[x<0]
    pf=float(wins.sum()/(-losses.sum())) if losses.sum()<0 else (float("inf") if wins.sum()>0 else np.nan)
    grouped=d.groupby("stock").return_pct.agg(["size","mean",lambda v:float(v.gt(0).mean()),
        lambda v:float(v[v>0].sum()/(-v[v<0].sum())) if v.lt(0).any() else np.inf])
    grouped.columns=["count","avg","wr","pf"]
    passed=((grouped["count"]>=cfg.min_trades_stock)&(grouped.wr>=cfg.min_win_rate)&
            (grouped.pf>=cfg.min_profit_factor)&(grouped.avg>0))
    coverage=100*int(passed.sum())/stock_count if stock_count else 0.0
    return {"ma":ma,"period":period,"stocks_eligible":stock_count,"stocks_traded":used,"trades":len(d),
            "win_rate_pct":round(100*len(wins)/len(x),3),"avg_trade_pct":round(float(x.mean()),4),
            "profit_factor":round(pf,4),"median_stock_return_pct":round(float(grouped.avg.median()),4),
            "stocks_pass":int(passed.sum()),"coverage_pct":round(coverage,3),
            "median_stock_max_dd_pct":round(float(s.loc[s.trades>0,"max_drawdown_pct"].median()),4)}

def select_ma(summary:pd.DataFrame,min_universe_stocks:int=100,min_period_trades:int=300)->pd.DataFrame:
    """不讀 test，僅依 train 與 validation 挑選參數。"""
    clean=summary[summary.period.isin(["train","validation"])].copy()
    if "stocks_eligible" in clean.columns and (clean.groupby("period").stocks_eligible.max()<min_universe_stocks).any():
        return pd.DataFrame()
    if (clean.groupby("period").trades.max()<min_period_trades).any():
        return pd.DataFrame()
    a=clean.pivot(index="ma",columns="period",values=["coverage_pct","profit_factor","stocks_pass","trades"])
    if len(a)==0: return pd.DataFrame()
    a.columns=["_".join(c) for c in a.columns];a=a.reset_index()
    for col in ["coverage_pct_train","coverage_pct_validation","profit_factor_train","profit_factor_validation"]:
        if col not in a: a[col]=0.0
    a["robust_coverage"]=a[["coverage_pct_train","coverage_pct_validation"]].min(axis=1)
    a=a.sort_values("ma").reset_index(drop=True)
    a["neighbor_robustness"]=[float(a.loc[abs(a.ma-m)<=2,"robust_coverage"].median()) for m in a.ma]
    a["rank_score"]=a.robust_coverage*.7+a.neighbor_robustness*.3
    a=a.sort_values(["rank_score","robust_coverage","ma"],ascending=[False,False,True]).reset_index(drop=True)
    a["selected_top3"]=False
    a.loc[:min(2,len(a)-1),"selected_top3"]=True
    return a

def run(data_dir:Path,out_dir:Path,cfg:Config,first_ma:int=16,last_ma:int=70,only_stocks:int=0):
    out_dir.mkdir(parents=True,exist_ok=True)
    candidates=sorted(p for p in data_dir.glob("*.csv") if re.fullmatch(r"[1-9][0-9]{3}",p.stem))
    if only_stocks: candidates=candidates[:only_stocks]
    if not candidates: raise RuntimeError(f"{data_dir} 沒有四位數股票 CSV，請先下載資料")
    input_report=[];loaded=[]
    for path in candidates:
        try:
            df,src=read_stock_csv(path)
            if len(df)>=cfg.min_history+25:
                loaded.append((path.stem,df))
                input_report.append({"stock":path.stem,"bars":len(df),"first_date":str(df.date.iloc[0].date()),
                                     "last_date":str(df.date.iloc[-1].date()),"turnover_source":src,"status":"ok"})
            else:
                input_report.append({"stock":path.stem,"bars":len(df),"status":"insufficient_history"})
        except Exception as exc:
            input_report.append({"stock":path.stem,"bars":0,"status":f"error: {str(exc)[:180]}"})
    pd.DataFrame(input_report).to_csv(out_dir/"input_audit.csv",index=False,encoding="utf-8-sig")
    if not loaded: raise RuntimeError("資料全部無效，請檢查 input_audit.csv")
    ma_summaries=[];all_trades=[];all_stats=[]
    for ma in range(first_ma,last_ma+1):
        trades=[];stats=[]
        for stock,df in loaded:
            z=signals(df,ma,cfg)
            for period in PERIODS:
                ts,st=backtest_one(stock,z,period,cfg)
                trades.extend(ts);stats.append(st)
        trades_df=pd.DataFrame(trades);stats_df=pd.DataFrame(stats)
        for period in PERIODS: ma_summaries.append(evaluate(trades_df,stats_df,ma,period,cfg))
        for t in trades: t["ma"]=ma
        for s in stats: s["ma"]=ma
        all_trades.extend(trades);all_stats.extend(stats)
        if (ma-first_ma)%5==0 or ma==last_ma:
            print(f"MA {ma}/{last_ma}: {len(loaded)} 檔 × 3 時段完成",flush=True)
    summary=pd.DataFrame(ma_summaries)
    ranking=select_ma(summary,cfg.min_universe_stocks,cfg.min_period_trades)
    summary.to_csv(out_dir/"ma_summary_all.csv",index=False,encoding="utf-8-sig")
    ranking.to_csv(out_dir/"ma_ranking_pretest.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(all_trades).to_csv(out_dir/"trades_all.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(all_stats).to_csv(out_dir/"stock_period_summary.csv",index=False,encoding="utf-8-sig")
    chosen=ranking.loc[ranking.selected_top3,"ma"].tolist() if not ranking.empty else []
    oos=summary[(summary.period=="test")&summary.ma.isin(chosen)].copy()
    oos.to_csv(out_dir/"oos_selected_only.csv",index=False,encoding="utf-8-sig")
    metadata={"config":asdict(cfg),"periods":PERIODS,"all_ma":list(range(first_ma,last_ma+1)),
              "input_stocks":len(candidates),"valid_stocks":len(loaded),
              "data_end_max":str(max(d.date.iloc[-1] for _,d in loaded).date()),
              "selected_using_train_and_validation_only":chosen,
              "research_status":"候選參數已選出" if chosen else "樣本數未達門檻，不進行最佳參數宣稱",
              "notes":["樣本外不參與挑參數","股票數與歷史下市股票完整度需檢驗",
                       "4% 停損會因跳空或滑價突破","還原股價與證券種類需確認",
                       "獨立股票交易分析，不是多檔同時持倉的總組合回測"]}
    (out_dir/"run_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print("預選 MA:",chosen,"\n樣本外結果：\n",oos.to_string(index=False),flush=True)
    return summary,ranking,oos

def parse_args():
    p=argparse.ArgumentParser(description="高流動性台股共通軍線實證回測")
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("results"))
    p.add_argument("--first-ma",type=int,default=16)
    p.add_argument("--last-ma",type=int,default=70)
    p.add_argument("--only-stocks",type=int,default=0)
    p.add_argument("--min-turnover",type=float,default=50_000_000)
    p.add_argument("--min-study-stocks",type=int,default=100)
    p.add_argument("--min-study-trades",type=int,default=300)
    p.add_argument("--exit-mode",choices=["original","full"],default="original")
    p.add_argument("--no-noise",action="store_true")
    p.add_argument("--no-trend",action="store_true")
    p.add_argument("--no-structure",action="store_true")
    p.add_argument("--fee",type=float,default=.001425)
    p.add_argument("--slippage",type=float,default=.001)
    return p.parse_args()

if __name__=="__main__":
    opts=parse_args()
    if not (16<=opts.first_ma<=opts.last_ma<=70): raise SystemExit("MA 只允許 16–70")
    config=Config(min_turnover=opts.min_turnover,min_universe_stocks=opts.min_study_stocks,
                  min_period_trades=opts.min_study_trades,exit_mode=opts.exit_mode,
                  noise_filter=not opts.no_noise,trend_filter=not opts.no_trend,
                  structure_stop=not opts.no_structure,fee=opts.fee,slippage=opts.slippage)
    run(opts.data,opts.output,config,opts.first_ma,opts.last_ma,opts.only_stocks)

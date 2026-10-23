from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from army_ma import Config,signals,backtest_one,select_ma,read_stock_csv

BASE=Config(min_price=1,min_turnover=10,turnover_days=2,min_history=2,noise_filter=False,trend_filter=False,structure_stop=False)

def candle_fixture():
    d=pd.bdate_range("2018-01-01",periods=120)
    return pd.DataFrame({"date":d,"open":100.0,"high":101.0,"low":99.0,"close":100.0,
                         "turnover":100_000_000.0,"raw_close":100.0})

def test_two_day_fake_breakdown():
    df=candle_fixture();df.loc[:82,"close"]=101.0;df.loc[:82,"high"]=102.0
    df.loc[83,["close","low"]]=[98.,96.]
    df.loc[84,["close","low"]]=[98.5,97.]
    z=signals(df,16,BASE)
    assert z.signal_entry.iloc[84]==1
    assert z.signal_entry.iloc[83]==0

def test_signal_confirmation_next_open_and_costs():
    z=candle_fixture();z["signal_entry"]=0;z["signal_exit"]=0;z["structure_low"]=97.;z["liq"]=True
    z.loc[90,"signal_entry"]=1
    z.loc[91,["open","high","low","close"]]=[100,103,99,102]
    z.loc[92,"signal_exit"]=3
    z.loc[93,["open","high","low","close"]]=[103,104,102,103]
    rows,_=backtest_one("2330",z,"train",BASE)
    assert len(rows)==1 and rows[0]["entry_date"]==str(z.date.iloc[91].date())
    assert rows[0]["exit_date"]==str(z.date.iloc[93].date())
    assert rows[0]["entry_price"]==100.1
    assert rows[0]["return_pct"]<2.3

def test_gap_beyond_stop_exceeds_four_percent():
    z=candle_fixture();z["signal_entry"]=0;z["signal_exit"]=0;z["structure_low"]=97.;z["liq"]=True
    z.loc[90,"signal_entry"]=1
    z.loc[91,["open","high","low","close"]]=[100,102,99,100]
    z.loc[92,["open","high","low","close"]]=[90,93,88,91]
    rows,_=backtest_one("2317",z,"train",BASE)
    assert rows[0]["exit_reason"]=="4%／結構停損"
    assert rows[0]["exit_date"]==str(z.date.iloc[92].date())
    assert rows[0]["return_pct"]<-10

def test_fake_breakout_sell_only_full():
    df=candle_fixture();df.loc[:82,"close"]=99.0;df.loc[82,"close"]=98.0
    df.loc[83,["close","high","low"]]=[101,103,98.]
    df.loc[84,["close","high","low"]]=[100,102,99.]
    assert signals(df,16,replace(BASE,exit_mode="full")).signal_exit.iloc[84]==4
    assert signals(df,16,replace(BASE,exit_mode="original")).signal_exit.iloc[84]==0

def test_adjusted_csv_and_turnover_raw(tmp_path):
    p=tmp_path/"2330.csv"
    pd.DataFrame({"Date":["2024-01-02","2024-01-03"],"Open":[100,100],"High":[105,105],
        "Low":[95,95],"Close":[100,100],"Adj Close":[50,50],"Volume":[1000,2000]}).to_csv(p,index=False)
    df,desc=read_stock_csv(p)
    assert df.close.tolist()==[50.,50.] and df.turnover.tolist()==[100000.,200000.]
    assert "估算" in desc

def test_selection_never_reads_test():
    rows=[]
    for ma in [16,17]:
        for period in ["train","validation","test"]:
            rows.append({"ma":ma,"period":period,"coverage_pct":20 if ma==16 else 10,
                         "profit_factor":1.5,"stocks_pass":10,"trades":50})
    a=pd.DataFrame(rows);baseline=select_ma(a,min_universe_stocks=0,min_period_trades=0)
    a.loc[(a.period=="test")&(a.ma==17),"coverage_pct"]=100
    changed=select_ma(a,min_universe_stocks=0,min_period_trades=0)
    assert baseline.ma.tolist()==changed.ma.tolist()

def test_stops_have_no_future_leak_and_no_same_bar_entry():
    z=candle_fixture();z["signal_entry"]=0;z.loc[100,"signal_entry"]=1
    z["signal_exit"]=0;z["structure_low"]=90.;z["liq"]=True
    rows,_=backtest_one("2454",z,"train",BASE)
    assert rows[0]["entry_date"]==str(z.date.iloc[101].date())

def test_small_universe_does_not_claim_best_ma():
    a=pd.DataFrame([{"ma":ma,"period":period,"coverage_pct":50,"profit_factor":1.5,
                     "stocks_pass":4,"trades":20,"stocks_eligible":8}
         for ma in [16,17] for period in ["train","validation","test"]])
    assert select_ma(a).empty

def test_finmind_400_does_not_retry(monkeypatch):
    from finmind_download import get_json
    import requests
    calls=[]
    def fake_get(*args,**kwargs):
        calls.append(1)
        response=requests.Response()
        response.status_code=400
        response._content=b'{"status":400,"msg":"Your level is free. Please update your user level."}'
        response.url="https://api.finmindtrade.com/api/v4/data"
        return response
    monkeypatch.setattr(requests,"get",fake_get)
    with pytest.raises(ValueError,match="HTTP 400"):
        get_json({"dataset":"TaiwanStockPriceAdj","data_id":"2330","start_date":"2020-01-01"})
    assert len(calls)==1

def test_finmind_free_price_response(monkeypatch):
    from finmind_download import get_json
    import requests
    calls=[]
    def fake_get(*args,**kwargs):
        calls.append(kwargs["params"]["dataset"])
        response=requests.Response()
        response.status_code=200
        response._content=b'{"status":200,"data":[{"date":"2026-01-02","open":100,"max":101,"min":99,"close":100,"Trading_money":80000000}]}'
        response.url="https://api.finmindtrade.com/api/v4/data"
        return response
    monkeypatch.setattr(requests,"get",fake_get)
    data=get_json({"dataset":"TaiwanStockPrice","data_id":"2330","start_date":"2026-01-01"})
    assert data[0]["Trading_money"]==80000000
    assert calls==["TaiwanStockPrice"]

def test_duplicate_split_and_par_value_are_applied_only_once():
    from proxy_corporate_adjust import clean_events,adjusted_close
    e=pd.DataFrame([
        {"stock":"2327","date":"2025-08-25","before":546,"after":136.5,
         "dataset":"TaiwanStockSplitPrice","priority":1},
        {"stock":"2327","date":"2025-08-25","before":546,"after":136.5,
         "dataset":"TaiwanStockParValueChange","priority":2}])
    events=clean_events(e)
    assert len(events)==1
    assert events.factor.iloc[0]==pytest.approx(.25)
    prices=pd.DataFrame({"date":["2025-08-22","2025-08-25"],
                         "close":[546.0,136.5]})
    adjusted,minimum,maximum=adjusted_close(prices,events,"2327")
    assert adjusted.adj_close.tolist()==pytest.approx([136.5,136.5])

def test_conflicting_same_day_corporate_action_stops_adjustment():
    from proxy_corporate_adjust import clean_events
    e=pd.DataFrame([
        {"stock":"4763","date":"2025-06-30","before":885,"after":88.5,
         "dataset":"TaiwanStockSplitPrice","priority":1},
        {"stock":"4763","date":"2025-06-30","before":885,"after":442.5,
         "dataset":"TaiwanStockDividendResult","priority":3}])
    with pytest.raises(ValueError,match="不同公司行動調整比率"):
        clean_events(e)

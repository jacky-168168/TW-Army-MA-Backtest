"""建立合成日 K，僅供程式壓力測試，不是真實行情。"""
from pathlib import Path
import numpy as np
import pandas as pd
out=Path(__file__).parent/"demo_data"
out.mkdir(exist_ok=True)
days=pd.bdate_range("2017-01-02","2026-10-20")
rng=np.random.default_rng(20261023)
for j in range(8):
    returns=rng.normal(.00045,.018,len(days))
    shock=np.zeros(len(days));shock[220::160]=-.12
    returns+=shock
    close=(35+j*10)*np.exp(returns.cumsum())
    op=close*np.exp(rng.normal(0,.006,len(days)))
    high=np.maximum(op,close)*(1+abs(rng.normal(.01,.004,len(days))))
    low=np.minimum(op,close)*(1-abs(rng.normal(.01,.004,len(days))))
    turnover=rng.lognormal(np.log(120_000_000+j*10_000_000),.4,len(days))
    d=pd.DataFrame({"date":days,"open":op,"max":high,"min":low,"close":close,
                    "Trading_money":turnover,"Trading_Volume":turnover/close})
    d.to_csv(out/f"{2300+j}.csv",index=False)
print("8 檔模擬股票已產生（合成資料，非真實台股）")

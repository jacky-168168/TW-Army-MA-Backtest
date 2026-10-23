#!/usr/bin/env python3
"""研究報表穩健性審核：不重跑價格、不利用樣本外資料挑選參數。"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

PERIODS=("train","validation","test")
def summarize(g):
    x=pd.to_numeric(g.return_pct,errors="coerce").dropna().to_numpy(dtype=float)
    if not len(x): return {}
    lo,hi=np.quantile(x,[.05,.95])
    trimmed=x[(x>=lo)&(x<=hi)]
    per=g.groupby("stock").return_pct.agg(["count","mean"])
    pos=x[x>0]
    gross_gain=pos.sum()
    return {"trades":len(x),"stocks_traded":g.stock.nunique(),
        "win_rate_pct":100*np.mean(x>0),"avg_trade_pct":np.mean(x),
        "median_trade_pct":np.median(x),"trimmed_5pct_avg":np.mean(trimmed),
        "winsor_5pct_avg":np.mean(np.clip(x,lo,hi)),
        "positive_stock_pct":100*np.mean(per["mean"]>0),
        "median_stock_avg_pct":per["mean"].median(),
        "stocks_with_5_trades":int((per["count"]>=5).sum()),
        "positive_stocks_with_5_trades_pct":(100*np.mean(per.loc[per["count"]>=5,"mean"]>0)
                                            if (per["count"]>=5).any() else np.nan),
        "over_100pct_trades":int((x>100).sum()),
        "below_minus_20pct_trades":int((x<-20).sum()),
        "top_1pct_wins_share_pct":(100*float(np.sort(pos)[-max(1,int(np.ceil(len(x)*.01))):].sum())/gross_gain
                                    if gross_gain>0 else np.nan),
        "gross_positive_pct_sum":gross_gain}

def audit(trades:pd.DataFrame,output:Path,source:str="未還原原始日 K"):
    required={"ma","period","stock","return_pct"}
    if not required.issubset(trades.columns):
        raise ValueError(f"缺少欄位: {required-set(trades.columns)}")
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for (ma,period),group in trades.groupby(["ma","period"],sort=True):
        if period not in PERIODS: continue
        rows.append({"ma":int(ma),"period":period,**summarize(group)})
    d=pd.DataFrame(rows)
    if d.empty: raise ValueError("找不到有效回測交易")
    d.to_csv(output/"robustness_all_ma.csv",index=False,encoding="utf-8-sig")
    in_sample=d[d.period!="test"]
    cols=["positive_stock_pct","trimmed_5pct_avg","median_stock_avg_pct"]
    wide=in_sample.pivot(index="ma",columns="period",values=cols)
    if not ({"train","validation"}.issubset(in_sample.period.unique())): raise ValueError("訓練／驗證期不足")
    wide.columns=["_".join(c) for c in wide.columns]
    wide=wide.reset_index()
    wide["min_positive_stock_pct"]=wide[["positive_stock_pct_train","positive_stock_pct_validation"]].min(axis=1)
    wide["min_trimmed_avg"]=wide[["trimmed_5pct_avg_train","trimmed_5pct_avg_validation"]].min(axis=1)
    wide["min_median_stock_avg"]=wide[["median_stock_avg_pct_train","median_stock_avg_pct_validation"]].min(axis=1)
    wide["pretest_pass"]=(wide.min_positive_stock_pct>=50)&(wide.min_trimmed_avg>0)&(wide.min_median_stock_avg>0)
    # 先看跨股票正期望值覆蓋，然後才看截尾交易均值；不得使用 test 排名。
    wide=wide.sort_values(["pretest_pass","min_positive_stock_pct","min_trimmed_avg","ma"],
                          ascending=[False,False,False,True])
    wide.to_csv(output/"robustness_pretest_rank.csv",index=False,encoding="utf-8-sig")
    pick=wide.head(3).ma.tolist()
    oos=d[(d.period=="test")&d.ma.isin(pick)]
    oos.to_csv(output/"robustness_oos_top3.csv",index=False,encoding="utf-8-sig")
    extremes=trades[(trades.return_pct>100)|(trades.return_pct<-20)].copy()
    extremes=extremes.sort_values("return_pct",ascending=False)
    extremes.to_csv(output/"extreme_trades_for_review.csv",index=False,encoding="utf-8-sig")
    qualified=int(wide.pretest_pass.sum())
    lines=[
        "# 台股軍線參數穩健性審核",
        "",
        f"- 資料來源：{source}。這是衍生交易的統計審核，**不是已修正除權息的重跑結果**。",
        "- 参数只使用訓練 2018–2022／驗證 2023–2024 期間排名；2025–2026 樣本外不參與挑選。",
        f"- 通過研究級跨期門檻的均線：**{qualified}／{len(wide)} 組**。門檻：訓練與驗證期各有至少 50% 交易股票呈正平均報酬、各有正截尾平均報酬、各有正股票中位平均報酬。",
        f"- 依跨股票初步分數選出的前三候選（即使沒通過門檻，也僅供診斷）：**{pick}**。",
        "- 僅有 150 檔非隨機先導股票與未還原價格；不得推論全市場普適性。",
        "- 每檔股票在不同 MA 的交易不是獨立樣本；平均交易報酬不等於可同時持倉的投資組合績效。",
        "- 極端交易報酬須逐筆核對除權息、減資、分割、非正常交易價格。",
        "",
        "## 候選參數樣本外診斷",
        "| MA | 交易數 | 勝率 % | 平均報酬 % | 中位數 % | 5% 截尾均值 % | 正報酬股票比例 % |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _,r in oos.sort_values("ma").iterrows():
        lines.append(f"| {int(r.ma)} | {int(r.trades)} | {r.win_rate_pct:.2f} | {r.avg_trade_pct:.2f} | {r.median_trade_pct:.2f} | {r.trimmed_5pct_avg:.2f} | {r.positive_stock_pct:.1f} |")
    lines+=["","## 研究結論",
            "訓練與驗證期都符合研究級穩健門檻，且排除公司行動疑點並重新測試後，才能開始宣稱存在可通用軍線。"]
    (output/"audit_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines[:9]),flush=True)
    return d,wide

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--trades",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("audit-results"))
    p.add_argument("--source",default="FinMind TaiwanStockPrice（未還原）")
    a=p.parse_args()
    audit(pd.read_csv(a.trades,low_memory=False),a.output,a.source)

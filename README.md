# 台股高流動性共通軍線回測

研究 SMA16–70 對高流動性台股的普適性，依原始 Pine Script「二日法則」進出場，並檢查交易成本和樣本外績效。

**研究原型，不是已驗證的投資策略，亦不得把合成測試當作真實台股回測。**

## 執行

1. GitHub → Actions → **台股共通軍線批次回測** → Run workflow。
2. 選擇 `sample` 下載 10 檔真實股票（不需 FinMind Token），驗證資料與回測是否正常。
3. 準備全市場研究時，在 Settings → Secrets and variables → Actions 新增 `FINMIND_TOKEN`，然後選 `full`。
4. 從該次 Actions 的 Artifacts 下載 `army-ma-backtest-results`，包含完整 SMA 報表、逐筆交易及樣本外結果。

也可在本機執行：
```bash
pip install -r requirements.txt
python -m pytest -q
python finmind_download.py --stocks 2317 2330 2454 --out data --start 2017-01-01
python army_ma.py --data data --output results
```

## 研究條件

- 股票：上市櫃 4 位數代號（正式研究須核對歷史證券類別、下市股票與存活者偏差）。
- 流動性：前一交易日前的 20 日平均成交額至少 5,000 萬元。
- SMA：16～70 日，趨勢斜率回看 5 日；20 日穿越軍線達 6 次則禁止新進場。
- 買入：第一日收盤跌破軍線且第二日未創新低；或第一日收盤突破、第二日創新高；訊號次一交易日開盤買入。
- 賣出：真跌破，或選配假突破；另有以買入價為基準的 4%／結構停損。
- 費用：買賣手續費各 0.1425%、賣出交易稅 0.3%、雙向滑價各 0.1%；跳空可超過 4% 停損。
- 訓練 2018–2022、驗證 2023–2024、樣本外 2025–2026-10-22。樣本外不得參與選參數。
- 經過股票數、交易次數門檻後，才會挑選候選軍線。沒有足夠數據就不宣稱最佳 MA。
- 若下載 API 受到限制，工作流程不應默默當成全市場資料齊全。

公開儲存庫不可提交 API Token、登入憑證或付費原始行情。Token 僅存 GitHub Actions Secrets。

研究方向：優先比較「最多不同股票的穩定正期望值」，而非單獨挑勝率最高的 MA。

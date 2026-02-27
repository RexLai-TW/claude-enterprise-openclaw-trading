# Claude Enterprise + OpenClaw 交易策略工具（中文說明）

簡介
----
這個專案可以把你的交易想法（用自然語言描述）自動轉成可執行的「策略樹」。流程很直覺：用平常語言描述想法 → 轉成確定性的交易策略 → 對策略做回測 → 最後產生可用的交易訊號。設計上以 Agent（代理）為核心，並特別強調 AI 不會直接接觸或操作你的資金。

主要功能
-------
- 將自然語言交易想法轉成結構化、確定性的策略樹（deterministic strategy trees）
- 對策略進行歷史回測並產生績效報告
- 從已驗證的策略產生交易訊號（signals）
- Agent-first 架構，便於串接不同模型或外部服務
- 安全設計：AI 不直接下單或控制資金，使用者保留最後決策權

核心概念（白話）
------------
1. 自然語言 → 策略：你用中文或英文描述想做的交易想法，系統會把這些描述轉成可以執行的策略節點（策略樹）。
2. 確定性：轉出的策略是 deterministic（確定性的），同一輸入會得到可重現的策略行為，方便測試與稽核。
3. 回測：在歷史資料上跑回測，檢查策略表現與風險特徵。
4. 訊號：回測與驗證通過後，系統會輸出可用的買賣訊號，供你接入交易或手動審核。
5. Agent-first：整個系統以「代理（Agent）」為中心來協調模型、規則與外部服務，方便擴充或替換模型。

快速開始（範例）
-------------
> 以下為通用範例，實際命令與檔名請依專案內的 README 或設定檔為準。

1. 建立虛擬環境並安裝相依套件
```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. 設定環境變數（範例）
- Claude Enterprise API Key、資料路徑、回測參數等。詳細參數請查看 config 範本或 docs。

3. 執行範例：把自然語言轉成策略、回測並輸出訊號
```bash
# 範例指令（請依專案實際腳本調整）
python scripts/convert_to_strategy.py --input "在 50 日均線之上，當 RSI < 30 時買進" --output strategy.json
python scripts/backtest.py --strategy strategy.json --history data/prices.csv --out results/
python scripts/generate_signals.py --strategy strategy.json --out signals.csv
```

使用流程（一步一步）
----------------
1. 用自然語言描述交易思路（範例：在 50 日均線之上，當 RSI < 30 時買進）
2. 系統把描述轉成策略樹（你可以檢視或手動微調）
3. 對策略做回測並檢查績效與風險指標
4. 確認無誤後啟用訊號產出（或接入你的交易執行系統）
5. 所有執行需經由使用者或系統的安全層審核，AI 不直接操作資金

架構概覽
-------
- Claude Enterprise：負責語意理解與策略生成的模型層
- OpenClaw：用於策略轉換與執行邏輯（專案整合部分）
- Agent 層：負責協調模型、規則、外部 API 與執行流程
- Backtester：回測引擎與績效分析
- Signal Exporter / UI（TypeScript）：前端或介面層，用於檢視與手動操作

安全與責任
--------
- AI 僅用於策略生成與建議；實際資金控制應由使用者或受信任的執行層負責
- 建議在真實交易前大量回測與模擬，並採用逐步上線（逐小額、逐步驗證）
- 保留完整日誌以供稽核與回溯

貢獻
---
歡迎提出 issue 或 PR。請先閱讀 CONTRIBUTING 文件（若有），在提交 code 前執行測試並遵守風格規範。

授權
---
請參考專案根目錄的 LICENSE 檔案。

聯絡
----
如需協助、翻譯其他檔案或將此中���版直接加到 repo（例如新增 README.zh.md 或替換 README），請告訴我你想要的動作：我可以直接幫你產生檔案內容、或協助建立 commit（你需授權或提供推送權限)。
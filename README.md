
# Toyota Live Quant (4-Process)

## 4プロセス
1) `scripts/ingest_loop.py`  … 1分足 / FX / 市場の取得→SQLite
2) `scripts/strategy_loop.py` … 100特徴→10式→Sを算出→signals_1m
3) `scripts/trader_loop.py`   … Sとポジションから発注（モック）・約定・PnL計上
4) `scripts/risk_loop.py`     … 当日PnL監視→HALT制御

## セットアップ
```bash
pip install -r requirements.txt
python scripts/ingest_loop.py --config config.sample.yaml
python scripts/strategy_loop.py --config config.sample.yaml
python scripts/trader_loop.py --config config.sample.yaml
python scripts/risk_loop.py --config config.sample.yaml --daily_loss_limit 50000
```

- DB: SQLite `runtime.db`（WAL）。UTCで保存、処理はJST。
- 実運用では `tq/io.py` を **kabuステAPI** 実装に差し替えてください。

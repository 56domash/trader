# fetch_real_data.py
import yfinance as yf
import pandas as pd

print("="*60)
print("Toyota (7203.T) 実データ取得")
print("="*60)

# 最近のデータを取得
symbol = "7203.T"
data = yf.download(symbol, period="5d", interval="1m",
                   progress=False, auto_adjust=False)

if data.empty:
    print("❌ データ取得失敗")
else:
    print(f"\n✓ 取得成功: {len(data)} bars")
    print(f"  期間: {data.index[0]} ～ {data.index[-1]}")

    # MultiIndex対応: 最初のレベルだけ取る
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 列名を小文字に統一
    data.columns = [c.lower() for c in data.columns]

    print(f"\n列名: {list(data.columns)}")

    # 最新100本を保存
    recent = data.tail(100).copy()

    # 必要な列だけ抽出
    cols_needed = ['open', 'high', 'low', 'close', 'volume']
    recent = recent[cols_needed]

    recent.to_parquet("toyota_real_data.parquet")

    print(f"\n✓ 保存: toyota_real_data.parquet ({len(recent)} bars)")
    print(f"  価格範囲: {recent['close'].min():.2f} - {recent['close'].max():.2f}")
    print(f"\n最新の5本:")
    print(recent.tail())

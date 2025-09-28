import pandas as pd
import numpy as np

# # ===== ダミーデータ生成 =====
# n = 1000  # 行数（例: 1000分足）
# # tz-aware（UTCつき）にする
# idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")

# # OHLCVデータをランダム生成（価格はランダムウォーク風）
# price = 100 + np.cumsum(np.random.randn(n)) * 0.1
# high = price + np.random.rand(n) * 0.5
# low = price - np.random.rand(n) * 0.5
# open_ = price + np.random.randn(n) * 0.05
# close = price
# volume = np.random.randint(100, 1000, n)

# df = pd.DataFrame({
#     "open": open_,
#     "high": high,
#     "low": low,
#     "close": close,
#     "volume": volume,
# }, index=idx)

# # ===== parquet 保存 =====
# df.to_parquet("sample.parquet")
# print("✅ sample.parquet (UTC index) を生成しました")

# make_sample_parquet.py

n = 500
idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")

# ランダムウォークでCloseを生成
price = np.cumprod(1 + 0.001 * np.random.randn(n)) * 100

df = pd.DataFrame({
    "Open": price * (1 + 0.0005 * np.random.randn(n)),
    "high": price * (1 + np.abs(0.001 * np.random.randn(n))),
    "low":  price * (1 - np.abs(0.001 * np.random.randn(n))),
    "close": price,
    "volume": np.random.randint(100, 1000, size=n),
    "Open": price * (1 + 0.0005 * np.random.randn(n)),
    "High": price * (1 + np.abs(0.001 * np.random.randn(n))),
    "Low":  price * (1 - np.abs(0.001 * np.random.randn(n))),
    "Close": price,
    "Volume": np.random.randint(100, 1000, size=n)

}, index=idx)


df.to_parquet("sample.parquet")
print("✅ sample.parquet (OHLCV付き) を生成しました")

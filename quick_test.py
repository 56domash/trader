# quick_test.py
from core.features.implementations.momentum import RSI14Feature
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


print("="*50)
print("RSI Feature テスト")
print("="*50)


# テストデータ
np.random.seed(42)
dates = pd.date_range("2025-09-29", periods=50, freq="1min", tz="UTC")
close = 3000 + np.cumsum(np.random.randn(50) * 5)

data = pd.DataFrame({
    "open": close,
    "high": close * 1.01,
    "low": close * 0.99,
    "close": close,
    "volume": 100000
}, index=dates)

print(f"データ: {len(data)} bars")

rsi = RSI14Feature()
result = rsi.compute(data)

print(f"RSI範囲: {result.min():.2f} - {result.max():.2f}")
print(f"RSI平均: {result.mean():.2f}")
print("✅ 成功！")

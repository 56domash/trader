# debug_rsi.py
from core.features.implementations.momentum import RSI14Feature
from tq.features_packs import compute_pack1
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# テストデータ
np.random.seed(42)
dates = pd.date_range("2025-09-29", periods=100, freq="1min", tz="UTC")
close = 3000 + np.cumsum(np.random.randn(100) * 5)

data = pd.DataFrame({
    "open": close,
    "high": close * 1.01,
    "low": close * 0.99,
    "close": close,
    "volume": np.random.randint(100000, 500000, 100)
}, index=dates)

print("="*60)
print("RSI計算の詳細デバッグ")
print("="*60)

# 既存システム
old_features = compute_pack1(data)
print("\n既存システムの列:")
print([c for c in old_features.columns if 'rsi' in c])

# p1_rsi14_01 の生値
old_rsi_raw = old_features["p1_rsi14_01"]
print(f"\np1_rsi14_01 (生値):")
print(f"  範囲: [{old_rsi_raw.min():.4f}, {old_rsi_raw.max():.4f}]")
print(f"  平均: {old_rsi_raw.mean():.4f}")
print(f"\n最後の5値:")
print(old_rsi_raw.tail())

# 新システム
new_rsi_feature = RSI14Feature()
new_rsi = new_rsi_feature.compute(data)

print(f"\n新システムのRSI (0-100):")
print(f"  範囲: [{new_rsi.min():.2f}, {new_rsi.max():.2f}]")
print(f"  平均: {new_rsi.mean():.2f}")
print(f"\n最後の5値:")
print(new_rsi.tail())

# 変換を試す
print("\n--- 変換テスト ---")
print("既存システムは (RSI-30)/40 という変換をしている可能性:")
converted_new = (new_rsi - 30) / 40
print(f"  新RSIを変換: [{converted_new.min():.4f}, {converted_new.max():.4f}]")

diff = (old_rsi_raw - converted_new).abs()
print(f"  差分: max={diff.max():.6f}, mean={diff.mean():.6f}")

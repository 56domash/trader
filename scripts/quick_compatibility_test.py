# quick_compatibility_test.py
from core.features.implementations.momentum import RSI14Feature
from tq.features_packs import compute_pack1
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# 既存システム

# 新システム

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
print("互換性クイックチェック")
print("="*60)

# 既存システムでRSI計算
old_features = compute_pack1(data)
old_rsi = old_features["p1_rsi14_01"] * 100  # 0-1 → 0-100に戻す

# 新システムでRSI計算
new_rsi_feature = RSI14Feature()
new_rsi = new_rsi_feature.compute(data)

# 比較
diff = (old_rsi - new_rsi).abs()
max_diff = diff.max()

print(f"\nRSI比較:")
print(f"  既存システム範囲: [{old_rsi.min():.2f}, {old_rsi.max():.2f}]")
print(f"  新システム範囲:   [{new_rsi.min():.2f}, {new_rsi.max():.2f}]")
print(f"  最大差分: {max_diff:.6f}")

if max_diff < 0.01:
    print("\n✅ 互換性OK！実データ統合に進めます")
else:
    print(f"\n⚠️ 差分が大きい（{max_diff:.4f}）")
    print("修正してから統合した方が良いです")

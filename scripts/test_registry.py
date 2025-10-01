# test_registry.py
from core.features.implementations.microstructure import VWAPDeviationFeature
from core.features.implementations.momentum import RSI14Feature, MACDHistogramFeature
from core.features.registry import FeatureRegistry
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


print("="*60)
print("Feature Registry 統合テスト")
print("="*60)


# テストデータ生成
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

print(f"\n✓ テストデータ: {len(data)} bars")
print(f"  価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")

# Registry作成と登録
print("\n--- Registry作成 ---")
registry = FeatureRegistry()
registry.register(RSI14Feature())
registry.register(MACDHistogramFeature())
registry.register(VWAPDeviationFeature())

print(f"✓ 登録特徴量: {registry.list_features()}")

# 全特徴量計算
print("\n--- 特徴量計算 ---")
results = registry.compute_all(data)

print(f"\n✓ 計算結果: {len(results)} features")
for name, series in results.items():
    print(f"  {name}:")
    print(f"    範囲: [{series.min():.4f}, {series.max():.4f}]")
    print(f"    平均: {series.mean():.4f}")
    print(f"    NaN数: {series.isna().sum()}")

# DataFrame化
df_features = pd.DataFrame(results)
print(f"\n✓ DataFrame化: shape={df_features.shape}")
print("\n最後の5行:")
print(df_features.tail())

print("\n✅ 全テスト成功！")

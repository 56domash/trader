# test_signals.py
from core.signals.aggregator import SignalAggregator
from core.signals.scoring import FeatureScorer, ScoringConfig
from core.features.implementations.microstructure import VWAPDeviationFeature
from core.features.implementations.momentum import RSI14Feature, MACDHistogramFeature
from core.features.registry import FeatureRegistry
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


print("="*60)
print("Signal Pipeline テスト")
print("="*60)

# データ生成
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

print(f"\n✓ データ: {len(data)} bars")

# 1. 特徴量計算
registry = FeatureRegistry()
registry.register(RSI14Feature())
registry.register(MACDHistogramFeature())
registry.register(VWAPDeviationFeature())

features = registry.compute_all(data)
print(f"✓ 特徴量: {len(features)} features")

# 2. スコアリング
scoring_config = {
    "rsi_14": ScoringConfig(method="direct_scale", direction="bullish"),
    "macd_histogram": ScoringConfig(method="tanh_normalize", direction="bullish", params={"window": 60}),
    "vwap_deviation": ScoringConfig(method="tanh_normalize", direction="bullish", params={"window": 20})
}

scorer = FeatureScorer(scoring_config)
scores = scorer.transform(features)
print(f"✓ スコア: {len(scores)} scores")

for name, score in scores.items():
    print(f"  {name}: [{score.min():.3f}, {score.max():.3f}]")

# 3. シグナル統合
weights = {
    "rsi_14": 1.2,
    "macd_histogram": 1.0,
    "vwap_deviation": 1.1
}

aggregator = SignalAggregator(weights)
signals = aggregator.aggregate(scores)

print(f"\n✓ シグナル生成: shape={signals.shape}")
print(f"  S範囲: [{signals['S'].min():.3f}, {signals['S'].max():.3f}]")

print("\n最後の10行:")
print(signals[['S', 'S_buy', 'S_sell']].tail(10))

print("\n✅ Phase 1プロトタイプ完成！")

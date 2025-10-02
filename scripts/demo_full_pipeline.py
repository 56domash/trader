# demo_full_pipeline.py
"""
エンドツーエンドパイプラインのデモ
Toyota Trading System V3
"""
from core.signals.decision import DecisionEngine, ThresholdConfig
from core.signals.aggregator import SignalAggregator
from core.signals.scoring import FeatureScorer, ScoringConfig
from core.features.implementations.microstructure import VWAPDeviationFeature
from core.features.implementations.momentum import RSI14Feature, MACDHistogramFeature
from core.features.registry import FeatureRegistry
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# 新システムのインポート

print("="*60)
print("V3 Full Pipeline Demo")
print("="*60)

# 実データ読み込み
data = pd.read_parquet("toyota_real_data.parquet")
print(f"\n✓ データ: {len(data)} bars")
print(f"  期間: {data.index[0]} ～ {data.index[-1]}")

# ========================================
# Layer 2: Feature Engine
# ========================================
print("\n--- Layer 2: Feature計算 ---")
registry = FeatureRegistry()
registry.register(RSI14Feature())
registry.register(MACDHistogramFeature())
registry.register(VWAPDeviationFeature())

features = registry.compute_all(data)
print(f"✓ 特徴量: {len(features)} features")

# ========================================
# Layer 3: Signal Pipeline
# ========================================
print("\n--- Layer 3: シグナル生成 ---")

# 3-1. スコアリング
scoring_config = {
    "rsi_14": ScoringConfig(method="direct_scale", direction="bullish"),
    "macd_histogram": ScoringConfig(method="tanh_normalize", direction="bullish", params={"window": 60}),
    "vwap_deviation": ScoringConfig(method="tanh_normalize", direction="bullish", params={"window": 20})
}
scorer = FeatureScorer(scoring_config)
scores = scorer.transform(features)
print(f"✓ スコアリング完了")

# 3-2. 統合
weights = {
    "rsi_14": 1.2,
    "macd_histogram": 1.0,
    "vwap_deviation": 1.1
}
aggregator = SignalAggregator(weights)
signals = aggregator.aggregate(scores)
print(f"✓ シグナル統合: S範囲=[{signals['S'].min():.3f}, {signals['S'].max():.3f}]")

# 3-3. 判定
threshold_config = ThresholdConfig(
    thr_long=0.15,
    thr_short=-0.15,
    exit_long=0.05,
    exit_short=-0.05,
    confirm_bars=2,
    ema_span=3
)
engine = DecisionEngine(threshold_config)
decisions = engine.decide(signals, current_position=0)
print(f"✓ 判定完了")

# ========================================
# 結果サマリー
# ========================================
print("\n" + "="*60)
print("結果サマリー")
print("="*60)

entry_long = (decisions['action'] == 'ENTRY_LONG').sum()
entry_short = (decisions['action'] == 'ENTRY_SHORT').sum()
hold = (decisions['action'] == 'HOLD').sum()

print(f"\nアクション統計:")
print(f"  ENTRY_LONG:  {entry_long:3d} 回")
print(f"  ENTRY_SHORT: {entry_short:3d} 回")
print(f"  HOLD:        {hold:3d} 回")

print(f"\n最新の10本:")
output = decisions[['S', 'S_ema', 'can_long', 'can_short', 'action']].tail(10)
print(output)

# 最新の判定
latest = engine.get_latest_action(signals)
print(f"\n最新状態:")
print(f"  時刻: {latest['timestamp']}")
print(f"  S_ema: {latest['S_ema']:.3f}")
print(f"  Action: {latest['action']}")

print("\n✅ Full Pipeline 動作確認完了！")

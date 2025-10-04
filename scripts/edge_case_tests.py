# tests/integration/test_edge_cases.py
"""
エッジケーステスト
異常系の挙動を確認
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.features.registry import FeatureRegistry
from core.features.implementations.momentum import RSI14Feature
from core.signals.decision import DecisionEngine, ThresholdConfig


def test_market_data_missing():
    """データ欠損時の挙動"""
    registry = FeatureRegistry()
    registry.register(RSI14Feature())
    
    # 空のDataFrame
    empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    features = registry.compute_all(empty_data)
    
    # 空データでもエラーにならないこと
    assert len(features) == 1
    assert 'rsi_14' in features


def test_all_nan_data():
    """全てNaNのデータ"""
    registry = FeatureRegistry()
    registry.register(RSI14Feature())
    
    # NaNデータ
    dates = pd.date_range('2025-01-01', periods=100, freq='1min', tz='UTC')
    nan_data = pd.DataFrame({
        'open': np.nan,
        'high': np.nan,
        'low': np.nan,
        'close': np.nan,
        'volume': np.nan
    }, index=dates)
    
    features = registry.compute_all(nan_data)
    
    # NaNデータでもエラーにならないこと
    assert 'rsi_14' in features
    # RSIはデフォルト値50.0を返す設計
    assert features['rsi_14'].notna().any()


def test_extreme_volatility():
    """極端なボラティリティ"""
    from core.features.implementations.volatility import ATRNormalizedFeature
    
    registry = FeatureRegistry()
    registry.register(ATRNormalizedFeature())
    
    # 価格が激しく変動
    dates = pd.date_range('2025-01-01', periods=100, freq='1min', tz='UTC')
    extreme_data = pd.DataFrame({
        'open': np.random.uniform(1000, 5000, 100),
        'high': np.random.uniform(5000, 10000, 100),
        'low': np.random.uniform(100, 1000, 100),
        'close': np.random.uniform(1000, 5000, 100),
        'volume': 1000
    }, index=dates)
    
    features = registry.compute_all(extreme_data)
    
    # ATRが計算されること
    assert 'atr_normalized' in features
    
    # 値域チェック（0-1の範囲内）
    atr = features['atr_normalized']
    assert atr.min() >= 0
    assert atr.max() <= 1


def test_zero_volume():
    """出来高ゼロ"""
    from core.features.implementations.microstructure import VolumeSpikeFeature
    
    registry = FeatureRegistry()
    registry.register(VolumeSpikeFeature())
    
    dates = pd.date_range('2025-01-01', periods=100, freq='1min', tz='UTC')
    zero_volume_data = pd.DataFrame({
        'open': 2500,
        'high': 2510,
        'low': 2490,
        'close': 2500,
        'volume': 0  # ゼロ
    }, index=dates)
    
    features = registry.compute_all(zero_volume_data)
    
    # ゼロ除算エラーにならないこと
    assert 'volume_spike' in features


def test_single_price():
    """価格が一定（変動なし）"""
    registry = FeatureRegistry()
    registry.register(RSI14Feature())
    
    dates = pd.date_range('2025-01-01', periods=100, freq='1min', tz='UTC')
    flat_data = pd.DataFrame({
        'open': 2500.0,
        'high': 2500.0,
        'low': 2500.0,
        'close': 2500.0,
        'volume': 1000
    }, index=dates)
    
    features = registry.compute_all(flat_data)
    
    # RSIは50.0（中立）になるはず
    rsi = features['rsi_14']
    assert rsi.iloc[-1] == 50.0


def test_missing_features_in_scoring():
    """一部の特徴量が欠損した場合"""
    from core.signals.scoring import FeatureScorer, ScoringConfig
    from core.signals.aggregator import SignalAggregator
    
    dates = pd.date_range('2025-01-01', periods=10, freq='1min', tz='UTC')
    
    # 一部の特徴量だけ存在
    features = {
        'rsi_14': pd.Series(np.random.uniform(0, 100, 10), index=dates),
        # macd_histogramは存在しない
    }
    
    # スコアリング
    config = {
        'rsi_14': ScoringConfig('direct_scale', 'bullish'),
        'macd_histogram': ScoringConfig('tanh_normalize', 'bullish', {'window': 20})
    }
    scorer = FeatureScorer(config)
    scores = scorer.transform(features)
    
    # rsi_14だけスコアリングされる
    assert 'rsi_14' in scores
    assert 'macd_histogram' not in scores
    
    # シグナル統合
    weights = {'rsi_14': 1.0, 'macd_histogram': 1.0}
    aggregator = SignalAggregator(weights)
    signals = aggregator.aggregate(scores)
    
    # エラーにならず、rsi_14だけで統合される
    assert 'S' in signals.columns


def test_decision_with_neutral_signal():
    """中立シグナルでの判定"""
    engine = DecisionEngine(ThresholdConfig(
        thr_long=0.15,
        thr_short=-0.15,
        exit_long=0.05,
        exit_short=-0.05
    ))
    
    dates = pd.date_range('2025-01-01', periods=10, freq='1min', tz='UTC')
    
    # S=0（中立）
    neutral_signals = pd.DataFrame({
        'S': 0.0,
        'S_buy': 0.5,
        'S_sell': 0.5
    }, index=dates)
    
    decisions = engine.decide(neutral_signals, current_position=0)
    
    # 全てHOLD
    assert (decisions['action'] == 'HOLD').all()


def test_decision_with_extreme_signal():
    """極端なシグナル（±1超え）での判定"""
    engine = DecisionEngine(ThresholdConfig(
        thr_long=0.15,
        thr_short=-0.15
    ))
    
    dates = pd.date_range('2025-01-01', periods=5, freq='1min', tz='UTC')
    
    # S > 1（異常値）
    extreme_signals = pd.DataFrame({
        'S': 2.0,  # ありえない値
        'S_buy': 1.0,
        'S_sell': 0.0
    }, index=dates)
    
    decisions = engine.decide(extreme_signals, current_position=0)
    
    # エラーにならず、ENTRY_LONGになる
    assert (decisions['action'] == 'ENTRY_LONG').any()


def test_insufficient_warmup_data():
    """ウォームアップ期間が不足"""
    registry = FeatureRegistry()
    registry.register(RSI14Feature())
    
    # RSIは14期間必要だが、10期間しかない
    dates = pd.date_range('2025-01-01', periods=10, freq='1min', tz='UTC')
    short_data = pd.DataFrame({
        'open': 2500,
        'high': 2510,
        'low': 2490,
        'close': 2500,
        'volume': 1000
    }, index=dates)
    
    features = registry.compute_all(short_data)
    
    # エラーにならず、計算可能な範囲で値を返す
    assert 'rsi_14' in features
    # 前半はNaN、後半は値があるはず
    assert features['rsi_14'].notna().any()


def test_concurrent_feature_computation():
    """複数特徴量の同時計算"""
    from core.features.implementations.momentum import MACDHistogramFeature
    from core.features.implementations.volatility import BollingerPositionFeature
    
    registry = FeatureRegistry()
    registry.register(RSI14Feature())
    registry.register(MACDHistogramFeature())
    registry.register(BollingerPositionFeature())
    
    dates = pd.date_range('2025-01-01', periods=100, freq='1min', tz='UTC')
    data = pd.DataFrame({
        'open': np.random.uniform(2400, 2600, 100),
        'high': np.random.uniform(2500, 2700, 100),
        'low': np.random.uniform(2300, 2500, 100),
        'close': np.random.uniform(2400, 2600, 100),
        'volume': 1000
    }, index=dates)
    
    features = registry.compute_all(data)
    
    # 全て計算される
    assert len(features) == 3
    assert 'rsi_14' in features
    assert 'macd_histogram' in features
    assert 'bollinger_position' in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# tests/integration/test_all_features.py
"""全特徴量の統合テスト"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from core.features.registry import FeatureRegistry
from core.features.implementations.momentum import (
    RSI14Feature, MACDHistogramFeature,
    WilliamsRFeature, StochasticKFeature
)
from core.features.implementations.volatility import (
    ATRPercentileFeature, BollingerPositionFeature,
    VolatilityRegimeFeature
)
from core.features.implementations.microstructure import (
    VWAPDeviationFeature, VolumeSpikeFeature,
    VolumeImbalanceFeature, VolumeRatioFeature,
    OpeningRangePositionFeature, OpeningRangeBreakoutFeature
)
from core.features.implementations.cross_market import (
    USDJPYCorrelationFeature, Nikkei225CorrelationFeature
)
from core.features.implementations.seasonality import (
    TimeOfDayFeature, EarlySessionBiasFeature,
    DayOfWeekFeature, MonthEndEffectFeature
)


@pytest.fixture
def sample_ohlcv():
    """サンプルOHLCVデータ"""
    # 2024-01-15 08:55-10:05 JST のデータ
    dates = pd.date_range(
        start="2024-01-14 23:55:00",  # UTC
        end="2024-01-15 01:05:00",
        freq="1min",
        tz="UTC"
    )

    n = len(dates)
    np.random.seed(42)

    close = 1000 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 3)
    low = close - np.abs(np.random.randn(n) * 3)
    open_ = close.shift(1).fillna(close[0])
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df


def test_all_features_computation(sample_ohlcv):
    """全特徴量の計算テスト"""
    registry = FeatureRegistry()

    # 全特徴量を登録
    registry.register(RSI14Feature())
    registry.register(MACDHistogramFeature())
    registry.register(WilliamsRFeature())
    registry.register(StochasticKFeature())

    registry.register(ATRPercentileFeature())
    registry.register(BollingerPositionFeature())
    registry.register(VolatilityRegimeFeature())

    registry.register(VWAPDeviationFeature())
    registry.register(VolumeSpikeFeature())
    registry.register(VolumeImbalanceFeature())
    registry.register(VolumeRatioFeature())
    registry.register(OpeningRangePositionFeature())
    registry.register(OpeningRangeBreakoutFeature())

    # 計算実行
    results = registry.compute_all(sample_ohlcv)

    # 全特徴量が計算されたか
    assert len(results) == 13

    # 各特徴量の出力をチェック
    for name, values in results.items():
        assert isinstance(values, pd.Series)
        assert len(values) == len(sample_ohlcv)
        assert values.notna().sum() > 0  # NaNでない値が存在


def test_feature_output_ranges(sample_ohlcv):
    """特徴量の出力範囲テスト"""
    registry = FeatureRegistry()
    registry.register(RSI14Feature())
    registry.register(BollingerPositionFeature())
    registry.register(OpeningRangePositionFeature())

    results = registry.compute_all(sample_ohlcv)

    # RSI: 0-100
    rsi = results["rsi_14"].dropna()
    assert rsi.min() >= 0 and rsi.max() <= 100

    # BB Position: 0-1
    bb_pos = results["bollinger_position"].dropna()
    assert bb_pos.min() >= 0 and bb_pos.max() <= 1

    # OR Position: 0-1
    or_pos = results["opening_range_position"].dropna()
    assert or_pos.min() >= 0 and or_pos.max() <= 1


def test_feature_registry_enable_disable(sample_ohlcv):
    """特徴量の有効/無効化テスト"""
    registry = FeatureRegistry()
    registry.register(RSI14Feature())
    registry.register(MACDHistogramFeature())

    # 全て有効の状態
    results_all = registry.compute_all(sample_ohlcv)
    assert len(results_all) == 2

    # MACD無効化
    registry.disable("macd_histogram")
    results_filtered = registry.compute_all(sample_ohlcv)
    assert len(results_filtered) == 1
    assert "rsi_14" in results_filtered
    assert "macd_histogram" not in results_filtered

    # 再有効化
    registry.enable("macd_histogram")
    results_reactivated = registry.compute_all(sample_ohlcv)
    assert len(results_reactivated) == 2


def test_cross_market_features_without_external_data(sample_ohlcv):
    """外部データなしでのCross-Market特徴量テスト"""
    registry = FeatureRegistry()

    # 外部データなしで登録
    registry.register(USDJPYCorrelationFeature())
    registry.register(Nikkei225CorrelationFeature())

    results = registry.compute_all(sample_ohlcv)

    # デフォルト値（0.5）が返されるべき
    assert results["usdjpy_correlation"].unique()[0] == 0.5
    assert results["nikkei225_correlation"].unique()[0] == 0.5


def test_cross_market_features_with_external_data():
    """外部データありでのCross-Market特徴量テスト"""
    # 外部データ付きサンプル作成
    dates = pd.date_range("2024-01-15 00:00", periods=60,
                          freq="1min", tz="UTC")

    data = pd.DataFrame({
        'open': 1000,
        'high': 1010,
        'low': 990,
        'close': 1000 + np.cumsum(np.random.randn(60)),
        'volume': 5000,
        'usdjpy_close': 150 + np.cumsum(np.random.randn(60) * 0.1),
        'nikkei225_close': 30000 + np.cumsum(np.random.randn(60) * 50)
    }, index=dates)

    registry = FeatureRegistry()
    registry.register(USDJPYCorrelationFeature())
    registry.register(Nikkei225CorrelationFeature())

    results = registry.compute_all(data)

    # 相関値が計算されているべき
    assert results["usdjpy_correlation"].notna().sum() > 20
    assert results["nikkei225_correlation"].notna().sum() > 20


def test_seasonality_features(sample_ohlcv):
    """Seasonality特徴量テスト"""
    registry = FeatureRegistry()

    registry.register(TimeOfDayFeature())
    registry.register(EarlySessionBiasFeature())
    registry.register(DayOfWeekFeature())
    registry.register(MonthEndEffectFeature())

    results = registry.compute_all(sample_ohlcv)

    # 全て計算されるべき
    assert len(results) == 4

    # Time of Day: 0-1 の範囲
    tod = results["time_of_day"]
    assert tod.min() >= 0 and tod.max() <= 1

    # Early Session: 0 or 1
    early = results["early_session_bias"]
    assert set(early.unique()).issubset({0.0, 1.0})

    # Day of Week: 0-1
    dow = results["day_of_week"]
    assert dow.min() >= 0 and dow.max() <= 1


def test_full_feature_set():
    """全特徴量の統合テスト"""
    from scripts.register_features import register_all_features

    dates = pd.date_range("2024-01-15 00:00", periods=70,
                          freq="1min", tz="UTC")
    data = pd.DataFrame({
        'open': 1000,
        'high': 1010,
        'low': 990,
        'close': 1000 + np.cumsum(np.random.randn(70)),
        'volume': 5000
    }, index=dates)

    registry = FeatureRegistry()
    register_all_features(registry)

    # Cross-Market特徴量を無効化（外部データなし）
    registry.disable("usdjpy_correlation")
    registry.disable("nikkei225_correlation")
    registry.disable("market_beta")
    registry.disable("relative_strength_vs_nikkei")

    results = registry.compute_all(data)

    # 有効な特徴量が計算されるべき
    # Momentum(4) + Volatility(3) + Micro(6) + Season(4)
    assert len(results) >= 17

    print(f"\n✅ Computed {len(results)} features:")
    for name in sorted(results.keys()):
        print(f"  - {name}")

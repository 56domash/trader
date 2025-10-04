# tests/unit/test_new_features_batch2.py

import pytest
import pandas as pd
import numpy as np

from core.features.implementations.volatility import (
    KeltnerChannelPositionFeature,
    KeltnerChannelWidthFeature
)
from core.features.implementations.momentum import (
    AroonUpFeature,
    AroonDownFeature
)
from core.features.implementations.microstructure import (
    GapOpenFeature,
    StreakUpFeature,
    StreakDownFeature
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100, freq="1min", tz="UTC")

    close = 3000 + np.cumsum(np.random.randn(100) * 5)
    high = close + np.abs(np.random.randn(100) * 3)
    low = close - np.abs(np.random.randn(100) * 3)
    open_price = close + np.random.randn(100) * 2

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(100000, 500000, 100)
    }, index=dates)


def test_keltner_features(sample_data):
    """Keltner Channel テスト"""
    kc_pos = KeltnerChannelPositionFeature()
    kc_width = KeltnerChannelWidthFeature()

    pos = kc_pos.compute(sample_data)
    width = kc_width.compute(sample_data)

    # 範囲チェック
    assert pos.min() >= 0 and pos.max() <= 1
    assert width.min() >= 0 and width.max() <= 1

    # NaNチェック
    assert pos.notna().sum() > 50
    assert width.notna().sum() > 50


def test_aroon_features(sample_data):
    """Aroon Indicator テスト"""
    aroon_up = AroonUpFeature()
    aroon_down = AroonDownFeature()

    up = aroon_up.compute(sample_data)
    down = aroon_down.compute(sample_data)

    # 範囲チェック
    assert up.min() >= 0 and up.max() <= 100
    assert down.min() >= 0 and down.max() <= 100


def test_gap_streak_features(sample_data):
    """Gap & Streak テスト"""
    gap = GapOpenFeature()
    streak_up = StreakUpFeature()
    streak_down = StreakDownFeature()

    gap_result = gap.compute(sample_data)
    up_result = streak_up.compute(sample_data)
    down_result = streak_down.compute(sample_data)

    # 範囲チェック
    assert gap_result.min() >= 0 and gap_result.max() <= 1
    assert up_result.min() >= 0 and up_result.max() <= 1
    assert down_result.min() >= 0 and down_result.max() <= 1

    # Streak は排他的（同時に両方高くならない）
    combined = up_result + down_result
    assert combined.max() <= 1.1  # 切り替わり時の誤差を許容

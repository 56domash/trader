# tests/unit/test_new_features.py（新規作成）
"""新規特徴量のテスト"""

import pytest
import pandas as pd
import numpy as np

from core.features.implementations.momentum import ADXFeature, IchimokuConversionFeature
from core.features.implementations.microstructure import MoneyFlowIndexFeature


@pytest.fixture
def sample_data():
    """テストデータ"""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100, freq="1min", tz="UTC")

    close = 3000 + np.cumsum(np.random.randn(100) * 5)
    high = close + np.abs(np.random.randn(100) * 3)
    low = close - np.abs(np.random.randn(100) * 3)

    return pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(100000, 500000, 100)
    }, index=dates)


def test_adx_feature(sample_data):
    """ADX計算テスト"""
    feature = ADXFeature()
    result = feature.compute(sample_data)

    # 範囲チェック
    assert result.min() >= 0
    assert result.max() <= 100

    # NaNチェック
    assert result.notna().sum() > 50


def test_mfi_feature(sample_data):
    """MFI計算テスト"""
    feature = MoneyFlowIndexFeature()
    result = feature.compute(sample_data)

    assert result.min() >= 0
    assert result.max() <= 100


def test_ichimoku_feature(sample_data):
    """一目均衡表テスト"""
    feature = IchimokuConversionFeature()
    result = feature.compute(sample_data)

    # 0-1範囲
    assert result.min() >= 0
    assert result.max() <= 1

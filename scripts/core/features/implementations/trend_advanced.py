# core/features/implementations/trend_advanced.py
"""
Trend Advanced Features
トレンド系の高度な特徴量
"""

import pandas as pd
import numpy as np
from ..base import Feature, FeatureMetadata


class EMACrossFeature(Feature):
    """EMA Cross (5期間 vs 20期間)"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="ema_cross",
            category="trend_advanced",
            version="1.0",
            lookback_bars=20,
            expected_range=(0, 1),
            description="EMA5 > EMA20なら1、それ以外0"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """EMAクロス検出"""
        close = data["close"]

        ema5 = close.ewm(span=5, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()

        # クロス（1=ゴールデンクロス、0=デッドクロス）
        cross = (ema5 > ema20).astype(float)

        return cross


class EMADistanceFeature(Feature):
    """EMA間の距離（正規化）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="ema_distance",
            category="trend_advanced",
            version="1.0",
            lookback_bars=20,
            expected_range=(0, 1),
            description="(EMA5 - EMA20) / EMA20の正規化"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """EMA距離 → tanh正規化"""
        close = data["close"]

        ema5 = close.ewm(span=5, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()

        # 乖離率
        distance = (ema5 - ema20) / ema20

        # tanh正規化で -0.05~+0.05 → 0~1
        normalized = (np.tanh(distance * 20) + 1) / 2

        return normalized.fillna(0.5).clip(0, 1)


class RangeBreakUpFeature(Feature):
    """20期間高値ブレイクアウト"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="range_break_up",
            category="trend_advanced",
            version="1.0",
            lookback_bars=20,
            expected_range=(0, 1),
            description="20期間高値を上抜けたら1"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """上方ブレイクアウト検出"""
        high = data["high"]

        # 20期間高値（1本前まで）
        rolling_high = high.rolling(20).max().shift(1)

        # ブレイクアウト
        breakout = (high > rolling_high).astype(float)

        return breakout


class RangeBreakDownFeature(Feature):
    """20期間安値ブレイクダウン"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="range_break_down",
            category="trend_advanced",
            version="1.0",
            lookback_bars=20,
            expected_range=(0, 1),
            description="20期間安値を下抜けたら1"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """下方ブレイクダウン検出"""
        low = data["low"]

        # 20期間安値（1本前まで）
        rolling_low = low.rolling(20).min().shift(1)

        # ブレイクダウン
        breakdown = (low < rolling_low).astype(float)

        return breakdown

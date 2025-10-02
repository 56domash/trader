# core/features/implementations/volatility.py

import ta
from ..base import Feature, FeatureMetadata
import pandas as pd


class ATRNormalizedFeature(Feature):
    """
    正規化ATR (Average True Range)
    価格に対する相対的なボラティリティ
    """

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="atr_normalized",
            category="volatility",
            version="1.0",
            lookback_bars=14,
            expected_range=(0, 0.05),
            description="ATR / Close (正規化ATR)"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """ATR計算"""
        atr = ta.volatility.AverageTrueRange(
            data['high'],
            data['low'],
            data['close'],
            window=14
        ).average_true_range()

        # 価格で正規化
        atr_norm = atr / data['close']

        return atr_norm.fillna(0.01).clip(0, 0.05)

# core/features/implementations/volatility.py に追加


class BollingerPositionFeature(Feature):
    """
    Bollinger Bands Position
    BB内の相対位置 (0=下限, 1=上限)
    """

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="bollinger_position",
            category="volatility",
            version="1.0",
            lookback_bars=20,
            expected_range=(0, 1),
            description="Bollinger Bands内の位置"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """BB Position計算"""
        bb = ta.volatility.BollingerBands(
            data['close'],
            window=20,
            window_dev=2
        )

        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()

        # BB内の位置
        position = (data['close'] - lower) / (upper - lower + 1e-10)

        return position.fillna(0.5).clip(0, 1)

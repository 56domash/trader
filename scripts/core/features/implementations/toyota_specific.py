# core/features/implementations/toyota_specific.py

import pandas as pd
import numpy as np
from ..base import Feature, FeatureMetadata


class OpeningRangeFeature(Feature):
    """
    Toyota Opening Range (09:00-09:05)
    最初5分の高値/安値からの相対位置
    """

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="opening_range",
            category="toyota_specific",
            version="1.0",
            lookback_bars=5,
            expected_range=(0, 1),
            description="Opening Range Position (09:00-09:05)"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Opening Range計算"""
        # 最初5本の高値/安値
        or_high = data['high'].rolling(5, min_periods=1).max()
        or_low = data['low'].rolling(5, min_periods=1).min()

        # 現在価格の位置 (0=下限, 1=上限)
        position = (data['close'] - or_low) / (or_high - or_low + 1e-10)

        return position.fillna(0.5).clip(0, 1)

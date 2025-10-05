# core/features/implementations/volume_advanced.py
"""
Volume Advanced Features
出来高系の高度な特徴量
"""

import pandas as pd
import numpy as np
from ..base import Feature, FeatureMetadata


class OBVFeature(Feature):
    """OBV - On-Balance Volume"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="obv",
            category="volume_advanced",
            version="1.0",
            lookback_bars=1,
            expected_range=(-1e6, 1e6),  # 累積値なので範囲広い
            description="On-Balance Volume（累積出来高）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        OBV計算
        - 終値が前日より高い → +volume
        - 終値が前日より低い → -volume
        """
        close = data["close"]
        volume = data["volume"]

        # 価格変化の方向
        direction = np.sign(close.diff())

        # 方向 × 出来高の累積
        obv = (direction * volume).cumsum()

        return obv.fillna(0.0)


class OBVNormalizedFeature(Feature):
    """OBV正規化版（0-1スケール）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="obv_normalized",
            category="volume_advanced",
            version="1.0",
            lookback_bars=60,
            expected_range=(0, 1),
            description="OBVの60期間パーセンタイル"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """OBV → 60期間パーセンタイルで正規化"""
        close = data["close"]
        volume = data["volume"]

        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()

        # 60期間でのパーセンタイル
        percentile = obv.rolling(60, min_periods=10).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x),
            raw=False
        )

        return percentile.fillna(0.5).clip(0, 1)

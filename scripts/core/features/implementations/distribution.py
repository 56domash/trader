# core/features/implementations/distribution.py
"""
Distribution Features
リターン分布の形状特徴量
"""

import pandas as pd
import numpy as np
from ..base import Feature, FeatureMetadata


class SkewnessFeature(Feature):
    """歪度（リターン分布の非対称性）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="skewness_20",
            category="distribution",
            version="1.0",
            lookback_bars=20,
            expected_range=(-3, 3),
            description="20期間リターンの歪度"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Skewness計算
        - 正: 右に長い尾（上昇トレンド）
        - 負: 左に長い尾（下落トレンド）
        """
        close = data["close"]
        returns = close.pct_change()

        # 20期間の歪度
        skew = returns.rolling(20, min_periods=10).skew()

        return skew.fillna(0.0).clip(-3, 3)


class KurtosisFeature(Feature):
    """尖度（リターン分布の裾の厚さ）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="kurtosis_20",
            category="distribution",
            version="1.0",
            lookback_bars=20,
            expected_range=(-3, 10),
            description="20期間リターンの尖度（3を引いた値）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Kurtosis計算
        - 正: ファットテール（急激な変動）
        - 負: 緩やかな分布
        """
        close = data["close"]
        returns = close.pct_change()

        # 20期間の尖度（excess kurtosis = kurtosis - 3）
        kurt = returns.rolling(20, min_periods=10).kurt()

        return kurt.fillna(0.0).clip(-3, 10)

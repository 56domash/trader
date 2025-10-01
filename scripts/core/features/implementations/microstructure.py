# core/features/implementations/microstructure.py
"""
マイクロストラクチャ系特徴量: VWAP
Toyota Trading System V3 - Phase 1
"""

import pandas as pd
from ..base import Feature, FeatureMetadata


class VWAPDeviationFeature(Feature):
    """VWAP からの乖離"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="vwap_deviation",
            category="microstructure",
            version="1.0",
            lookback_bars=1,
            expected_range=(-0.05, 0.05),
            description="VWAP からの乖離率"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """VWAP乖離計算"""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Typical Price
        tp = (high + low + close) / 3

        # VWAP
        cumulative_tpv = (tp * volume).cumsum()
        cumulative_vol = volume.cumsum()
        vwap = cumulative_tpv / (cumulative_vol + 1e-10)

        # 乖離率
        deviation = (close - vwap) / (vwap + 1e-10)

        return deviation.fillna(0.0)

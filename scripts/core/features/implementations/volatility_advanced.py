# core/features/implementations/volatility_advanced.py
"""
Volatility Advanced Features
ボラティリティ系の高度な特徴量
"""

import pandas as pd
import numpy as np
from ..base import Feature, FeatureMetadata


class ParkinsonVolatilityFeature(Feature):
    """Parkinson Volatility（高値-安値ベース）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="parkinson_volatility",
            category="volatility_advanced",
            version="1.0",
            lookback_bars=20,
            expected_range=(0, 0.1),
            description="Parkinson推定ボラティリティ（20期間）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Parkinson Volatility = sqrt(sum(log(H/L)^2) / (4*log(2)*n))
        """
        high = data["high"]
        low = data["low"]

        # log(H/L)
        hl_ratio = np.log(high / low).replace([np.inf, -np.inf], np.nan)

        # Parkinson推定
        pk_vol = (hl_ratio.pow(2) / (4 * np.log(2))).rolling(20).mean()
        pk_vol = pk_vol.clip(lower=0).pow(0.5)

        return pk_vol.fillna(0.01)


class GarmanKlassVolatilityFeature(Feature):
    """Garman-Klass Volatility（OHLC全体ベース）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="garman_klass_volatility",
            category="volatility_advanced",
            version="1.0",
            lookback_bars=20,
            expected_range=(0, 0.1),
            description="Garman-Klass推定ボラティリティ（20期間）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        GK = sqrt(0.5*log(H/L)^2 - (2*log(2)-1)*log(C/O)^2)
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_price = data["open"]

        # log(H/L)
        hl = np.log(high / low).replace([np.inf, -np.inf], np.nan)

        # log(C/O)
        co = np.log(close / open_price).replace([np.inf, -np.inf], np.nan)

        # GK推定
        gk_vol = (0.5 * hl.pow(2) - (2*np.log(2)-1)
                  * co.pow(2)).rolling(20).mean()
        gk_vol = gk_vol.clip(lower=0).pow(0.5)

        return gk_vol.fillna(0.01)

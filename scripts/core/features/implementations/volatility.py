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

# core/features/implementations/volatility.py に追加


class ATRPercentileFeature(Feature):
    """ATR Percentile (ボラティリティレジーム検出)"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="atr_percentile",
            category="volatility",
            version="1.0",
            lookback_bars=60,
            expected_range=(0, 1),
            description="ATRの60期間パーセンタイル（ボラ高=1, 低=0）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """ATRパーセンタイル計算"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR (14期間)
        atr = tr.rolling(14).mean()

        # 60期間でのパーセンタイル
        percentile = atr.rolling(60).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x),
            raw=False
        )

        return percentile.fillna(0.5)


class BollingerPositionFeature(Feature):
    """Bollinger Band Position"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="bollinger_position",
            category="volatility",
            version="1.0",
            lookback_bars=20,
            expected_range=(0, 1),
            description="BB内での価格位置（下限=0, 上限=1）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """BB Position計算"""
        close = data["close"]

        # BB計算
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()

        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20

        # Position [0,1]
        position = (close - lower) / (upper - lower + 1e-10)

        return position.clip(0, 1).fillna(0.5)


class VolatilityRegimeFeature(Feature):
    """ボラティリティレジーム（高/中/低）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="volatility_regime",
            category="volatility",
            version="1.0",
            lookback_bars=60,
            expected_range=(0, 1),
            description="ボラティリティレジーム（低=0, 高=1）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """ボラレジーム検出"""
        returns = data["close"].pct_change()
        vol = returns.rolling(20).std()

        # 60期間での分位点
        vol_percentile = vol.rolling(60).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x),
            raw=False
        )

        return vol_percentile.fillna(0.5)

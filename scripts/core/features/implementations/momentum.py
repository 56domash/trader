# core/features/implementations/momentum.py

"""
モメンタム系特徴量: RSI, MACD (ta-lib準拠)
Toyota Trading System V3 - Phase 1 (修正版)
"""

import pandas as pd
import ta
from ..base import Feature, FeatureMetadata


class RSI14Feature(Feature):
    """RSI 14期間（ta-lib標準実装）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="rsi_14",
            category="momentum",
            version="1.1",  # バージョンアップ
            lookback_bars=14,
            expected_range=(0, 100),
            description="14期間RSI（0-100、ta-lib準拠）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """RSI計算（ta-lib使用）"""
        close = data["close"]

        # ta-lib標準実装を使用
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()

        return rsi.fillna(50.0)


class MACDHistogramFeature(Feature):
    """MACD ヒストグラム（ta-lib準拠）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="macd_histogram",
            category="momentum",
            version="1.1",
            lookback_bars=26,
            expected_range=(-10, 10),
            description="MACD Histogram (ta-lib準拠)"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """MACD Histogram計算（ta-lib使用）"""
        close = data["close"]

        # ta-lib標準実装
        macd_indicator = ta.trend.MACD(
            close, window_slow=26, window_fast=12, window_sign=9)
        histogram = macd_indicator.macd_diff()

        return histogram.fillna(0.0)


class VWAPDeviationFeature(Feature):
    """VWAP からの乖離"""
    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="vwap_deviation",
            category="microstructure",
            version="1.0",
            lookback_bars=1,  # リアルタイム計算
            expected_range=(-0.05, 0.05),
            description="VWAP からの乖離率"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """VWAP乖離計算"""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Typical Price = (H + L + C) / 3
        tp = (high + low + close) / 3

        # VWAP = cumsum(TP * Volume) / cumsum(Volume)
        cumulative_tpv = (tp * volume).cumsum()
        cumulative_vol = volume.cumsum()

        vwap = cumulative_tpv / (cumulative_vol + 1e-10)

        # 乖離率 = (Close - VWAP) / VWAP
        deviation = (close - vwap) / (vwap + 1e-10)

        return deviation.fillna(0.0)

# core/features/implementations/momentum.py (完全修正版)
"""
モメンタム系特徴量: RSI, MACD (修正版)
Toyota Trading System V3 - Phase 3A完全修正
"""

import pandas as pd
import numpy as np
import ta
from ..base import Feature, FeatureMetadata


class RSI14Feature(Feature):
    """RSI 14期間（ta-lib標準実装）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="rsi_14",
            category="momentum",
            version="1.1",
            lookback_bars=14,
            expected_range=(0, 100),
            description="14期間RSI（0-100、ta-lib準拠）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """RSI計算（ta-lib使用）"""
        close = data["close"]
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        return rsi.fillna(50.0)


class MACDHistogramFeature(Feature):
    """MACD ヒストグラム（修正版）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="macd_histogram",
            category="momentum",
            version="1.2",  # バージョンアップ
            lookback_bars=26,
            expected_range=(-10, 10),
            description="MACD Histogram（修正版）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """MACD Histogram計算（修正版）"""
        close = data["close"]

        # 🔧 修正: EMAを直接計算
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()

        # MACD Line
        macd_line = ema_fast - ema_slow

        # Signal Line
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        # 🔧 修正: 初期値の問題を回避
        # 最初の26本は計算不可なので0埋め
        histogram = histogram.fillna(0.0)

        # 🔧 修正: 異常値クリップ
        histogram = histogram.clip(-10, 10)

        return histogram


class VWAPDeviationFeature(Feature):
    """VWAP からの乖離（完全修正版）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="vwap_deviation",
            category="microstructure",
            version="1.2",  # バージョンアップ
            lookback_bars=1,
            expected_range=(-0.05, 0.05),
            description="VWAP からの乖離率（完全修正版）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """VWAP乖離計算（完全修正版）"""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Typical Price
        tp = (high + low + close) / 3

        # 🔧 修正: 累積計算の安全性向上
        tpv = tp * volume
        cumulative_tpv = tpv.cumsum()
        cumulative_vol = volume.cumsum()

        # 🔧 修正: ゼロ除算とNaNを完全回避
        # volumeが0の場合は前の値を使う
        cumulative_vol = cumulative_vol.replace(
            0, np.nan).fillna(method='ffill').fillna(1.0)

        # VWAP計算
        vwap = cumulative_tpv / cumulative_vol

        # 初期値の処理（VWAPが計算できない場合はclose使用）
        vwap = vwap.fillna(close)

        # 乖離率計算
        deviation = (close - vwap) / vwap

        # 🔧 修正: 異常値除去
        deviation = deviation.replace([np.inf, -np.inf], np.nan)
        deviation = deviation.clip(-0.5, 0.5)  # ±50%でクリップ
        deviation = deviation.fillna(0.0)

        return deviation


class WilliamsRFeature(Feature):
    """Williams %R - 10期間"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="williams_r",
            category="momentum",
            version="1.0",
            lookback_bars=10,
            expected_range=(-100, 0),
            description="Williams %R 10期間"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Williams %R = -100 * (highest_high - close) / (highest_high - lowest_low)
        範囲: -100 (売られ過ぎ) ~ 0 (買われ過ぎ)
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # 10期間の高値/安値
        highest_high = high.rolling(10, min_periods=1).max()
        lowest_low = low.rolling(10, min_periods=1).min()

        # Williams %R計算
        wr = -100 * (highest_high - close) / \
            (highest_high - lowest_low + 1e-10)

        # 範囲制限とNaN処理
        wr = wr.clip(-100, 0).fillna(-50.0)

        return wr


class StochasticFeature(Feature):
    """Stochastic %K - 14期間"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="stochastic_k",
            category="momentum",
            version="1.0",
            lookback_bars=14,
            expected_range=(0, 100),
            description="Stochastic %K 14期間"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Stochastic %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
        範囲: 0 (売られ過ぎ) ~ 100 (買われ過ぎ)
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # ta-libのStochasticを使用
        stoch = ta.momentum.StochasticOscillator(
            high=high,
            low=low,
            close=close,
            window=14,
            smooth_window=3
        )

        # %K値を取得
        k_value = stoch.stoch()

        # 範囲制限とNaN処理
        k_value = k_value.clip(0, 100).fillna(50.0)

        return k_value

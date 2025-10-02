# core/features/implementations/cross_market.py
"""
Cross-Market Features
Toyota株と他市場の相関を捉える特徴量
"""

from ..base import Feature, FeatureMetadata
import pandas as pd
import numpy as np


class USDJPYCorrelationFeature(Feature):
    """USD/JPY相関（Toyota輸出企業特性）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="usdjpy_correlation",
            category="cross_market",
            version="1.0",
            lookback_bars=20,
            expected_range=(-1, 1),
            description="20期間のUSD/JPY相関係数"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        USD/JPY相関計算

        Note: 外部データが必要。利用不可の場合はデフォルト値（0.5）を返す
        """
        # 実装時は外部データソースから取得
        # ここでは簡易実装として、存在チェックのみ
        if 'usdjpy_close' in data.columns:
            toyota_returns = data['close'].pct_change()
            usdjpy_returns = data['usdjpy_close'].pct_change()

            correlation = toyota_returns.rolling(20).corr(usdjpy_returns)

            # [-1,1] → [0,1] に変換
            normalized = (correlation + 1) / 2
            return normalized.fillna(0.5)
        else:
            # 外部データなし → 中立値
            return pd.Series(0.5, index=data.index)


class Nikkei225CorrelationFeature(Feature):
    """日経225相関"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="nikkei225_correlation",
            category="cross_market",
            version="1.0",
            lookback_bars=20,
            expected_range=(-1, 1),
            description="20期間の日経225相関係数"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """日経225相関計算"""
        if 'nikkei225_close' in data.columns:
            toyota_returns = data['close'].pct_change()
            nikkei_returns = data['nikkei225_close'].pct_change()

            correlation = toyota_returns.rolling(20).corr(nikkei_returns)

            normalized = (correlation + 1) / 2
            return normalized.fillna(0.5)
        else:
            return pd.Series(0.5, index=data.index)


class MarketBetaFeature(Feature):
    """市場ベータ（日経225に対する感応度）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="market_beta",
            category="cross_market",
            version="1.0",
            lookback_bars=60,
            expected_range=(0, 2),
            description="60期間の市場ベータ（日経225対比）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """市場ベータ計算"""
        if 'nikkei225_close' in data.columns:
            toyota_returns = data['close'].pct_change()
            market_returns = data['nikkei225_close'].pct_change()

            # ローリング回帰でベータ計算
            def rolling_beta(window):
                cov = toyota_returns.rolling(window).cov(market_returns)
                var = market_returns.rolling(window).var()
                return cov / (var + 1e-10)

            beta = rolling_beta(60)

            # ベータ 0-2 を 0-1 に正規化
            normalized = (beta / 2.0).clip(0, 1)
            return normalized.fillna(0.5)
        else:
            return pd.Series(0.5, index=data.index)


class RelativeStrengthIndexFeature(Feature):
    """相対強度（日経225対比のパフォーマンス）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="relative_strength_vs_nikkei",
            category="cross_market",
            version="1.0",
            lookback_bars=20,
            expected_range=(0, 1),
            description="日経225に対する相対パフォーマンス"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """相対強度計算"""
        if 'nikkei225_close' in data.columns:
            toyota_ret_20 = data['close'].pct_change(20)
            nikkei_ret_20 = data['nikkei225_close'].pct_change(20)

            # アウトパフォーマンス度合い
            relative_perf = toyota_ret_20 - nikkei_ret_20

            # tanh正規化で [-0.1, 0.1] くらいの範囲を [0,1]に
            normalized = (np.tanh(relative_perf * 10) + 1) / 2
            return normalized.fillna(0.5)
        else:
            return pd.Series(0.5, index=data.index)

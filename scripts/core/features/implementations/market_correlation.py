# core/features/implementations/market_correlation.py
"""
Market Correlation Features
マーケット相関系特徴量
"""

import pandas as pd
import numpy as np
from ..base import Feature, FeatureMetadata


class MarketBetaNK225Feature(Feature):
    """日経225に対する市場ベータ"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="market_beta_nk225",
            category="market_correlation",
            version="1.0",
            lookback_bars=60,
            expected_range=(0, 2),
            description="60期間の日経225ベータ"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Beta = Cov(Toyota, NK225) / Var(NK225)

        Note: 外部データ必要。なければ中立値1.0
        """
        if 'nikkei225_close' not in data.columns:
            # 外部データなし → Toyotaは日経225構成銘柄なので1.0想定
            return pd.Series(1.0, index=data.index)

        toyota_returns = data['close'].pct_change()
        nk225_returns = data['nikkei225_close'].pct_change()

        # ローリング共分散 / ローリング分散
        cov = toyota_returns.rolling(60).cov(nk225_returns)
        var = nk225_returns.rolling(60).var()

        beta = cov / (var + 1e-10)

        # 0-2に正規化（ベータ>2は異常）
        beta_normalized = (beta / 2.0).clip(0, 1)

        return beta_normalized.fillna(0.5)

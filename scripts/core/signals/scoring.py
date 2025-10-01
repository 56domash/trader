# core/signals/scoring.py

from typing import Dict
import pandas as pd
import numpy as np


class ScoringConfig:
    """スコアリング設定"""

    def __init__(self, method: str, direction: str, **kwargs):
        self.method = method      # direct_scale, tanh_normalize, minmax
        self.direction = direction  # bullish, bearish, neutral
        self.params = kwargs      # window, clip_range, etc.


class FeatureScorer:
    """特徴量→[0,1]スコア変換"""

    def __init__(self, config: Dict[str, ScoringConfig]):
        """
        Args:
            config: {feature_name: ScoringConfig}

        Example:
            config = {
                "rsi_14": ScoringConfig(
                    method="direct_scale",
                    direction="bullish"
                ),
                "macd_histogram": ScoringConfig(
                    method="tanh_normalize",
                    direction="bullish",
                    window=60
                )
            }
        """
        self.config = config

    def transform(self, features: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        全特徴量を[0,1]スコアに変換

        Returns:
            Dict[feature_name, score_series]
        """
        scores = {}

        for name, values in features.items():
            if name not in self.config:
                continue

            cfg = self.config[name]

            # メソッド別変換
            if cfg.method == "direct_scale":
                # 例: RSI 0-100 → 0-1
                scored = values / 100.0

            elif cfg.method == "tanh_normalize":
                # z-score → tanh → [0,1]
                window = cfg.params.get("window", 60)
                z = self._rolling_zscore(values, window)
                scored = (np.tanh(z) + 1) / 2

            elif cfg.method == "minmax":
                # min-max正規化
                window = cfg.params.get("window", 60)
                scored = self._rolling_minmax(values, window)

            else:
                raise ValueError(f"Unknown method: {cfg.method}")

            # direction処理
            if cfg.direction == "bearish":
                scored = 1.0 - scored

            scores[name] = scored.clip(0, 1).fillna(0.5)

        return scores

    def _rolling_zscore(self, s: pd.Series, window: int) -> pd.Series:
        """ローリングZ-score"""
        mean = s.rolling(window).mean()
        std = s.rolling(window).std()
        return (s - mean) / (std + 1e-10)

    def _rolling_minmax(self, s: pd.Series, window: int) -> pd.Series:
        """ローリングMin-Max正規化"""
        min_val = s.rolling(window).min()
        max_val = s.rolling(window).max()
        return (s - min_val) / (max_val - min_val + 1e-10)

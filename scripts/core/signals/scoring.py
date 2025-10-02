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

    # def _rolling_zscore(self, s: pd.Series, window: int) -> pd.Series:
    #     """ローリングZ-score"""
    #     mean = s.rolling(window).mean()
    #     std = s.rolling(window).std()
    #     return (s - mean) / (std + 1e-10)

    # def _rolling_minmax(self, s: pd.Series, window: int) -> pd.Series:
    #     """ローリングMin-Max正規化"""
    #     min_val = s.rolling(window).min()
    #     max_val = s.rolling(window).max()
    #     return (s - min_val) / (max_val - min_val + 1e-10)

    def _rolling_zscore(self, s: pd.Series, window: int) -> pd.Series:
        """ローリングZ-score（改善版）"""
        # 🔧 修正: min_periodsを追加して初期値でもz-scoreを計算可能に
        mean = s.rolling(window, min_periods=max(1, window//2)).mean()
        std = s.rolling(window, min_periods=max(1, window//2)).std()

        # 🔧 修正: stdが0の場合の処理
        std = std.replace(0, np.nan).fillna(s.std() if s.std() > 0 else 1.0)

        z = (s - mean) / std

        # 🔧 修正: 異常値をクリップ
        z = z.clip(-3, 3)

        return z.fillna(0)

    def _rolling_minmax(self, s: pd.Series, window: int) -> pd.Series:
        """ローリングMin-Max正規化（改善版）"""
        # 🔧 修正: min_periodsを追加
        min_val = s.rolling(window, min_periods=max(1, window//2)).min()
        max_val = s.rolling(window, min_periods=max(1, window//2)).max()

        # 🔧 修正: min==maxの場合の処理
        range_val = max_val - min_val
        range_val = range_val.replace(0, np.nan).fillna(
            s.max() - s.min() if s.max() > s.min() else 1.0)

        normalized = (s - min_val) / range_val

        return normalized.fillna(0.5).clip(0, 1)

# core/signals/aggregator.py

from typing import Dict
import pandas as pd


class SignalAggregator:
    """スコアの重み付き統合"""

    def __init__(self, weights: Dict[str, float]):
        """
        Args:
            weights: {feature_name: weight}

        Example:
            weights = {
                "rsi_14": 1.2,
                "macd_histogram": 1.0,
                "vwap_deviation": 1.1
            }
        """
        self.weights = weights

    def aggregate(self, scores: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        重み付き統合

        Returns:
            DataFrame with columns:
            - S: 統合シグナル [-1, 1]
            - S_buy: 買いスコア [0, 1]
            - S_sell: 売りスコア [0, 1]
            - contrib_{feature}: 各特徴量の寄与度
        """
        index = next(iter(scores.values())).index

        weighted_sum = pd.Series(0.0, index=index)
        total_weight = 0.0
        contributions = {}

        for name, score in scores.items():
            w = self.weights.get(name, 0.0)

            if w > 0:
                # [-0.5, +0.5]にシフト
                contribution = w * (score - 0.5)
                weighted_sum += contribution
                total_weight += w

                # 寄与度保存
                contributions[f"contrib_{name}"] = contribution

        # 正規化
        S_raw = weighted_sum / (total_weight + 1e-10)

        result = pd.DataFrame({
            "S": S_raw.clip(-1, 1),
            "S_buy": (S_raw + 1) / 2,
            "S_sell": (-S_raw + 1) / 2,
            **contributions
        }, index=index)

        return result

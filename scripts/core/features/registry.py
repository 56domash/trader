# core/features/registry.py

import logging
from typing import Dict, Set
import pandas as pd
from .base import Feature


class FeatureRegistry:
    """特徴量登録・管理"""

    def __init__(self):
        self._features: Dict[str, Feature] = {}
        self._enabled: Set[str] = set()
        self.logger = logging.getLogger(__name__)

    def register(self, feature: Feature):
        """
        特徴量を登録

        Example:
            registry.register(RSI14Feature())
        """
        name = feature.metadata.name

        if name in self._features:
            self.logger.warning(f"Overwriting feature: {name}")

        self._features[name] = feature
        self._enabled.add(name)  # デフォルトで有効

        self.logger.info(f"Registered: {name} ({feature.metadata.category})")

    def enable(self, name: str):
        """特徴量を有効化"""
        if name not in self._features:
            raise ValueError(f"Unknown feature: {name}")
        self._enabled.add(name)

    def disable(self, name: str):
        """特徴量を無効化（計算スキップ）"""
        self._enabled.discard(name)

    def compute_all(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        有効な特徴量を全て計算

        Args:
            data: OHLCV DataFrame (warmup含む)

        Returns:
            Dict[feature_name, computed_values]
        """
        results = {}

        for name in sorted(self._enabled):
            feature = self._features[name]

            try:
                # 計算実行
                result = feature.compute(data)

                # 妥当性チェック
                is_valid, msg = feature.validate_output(result)

                if is_valid:
                    results[name] = result
                else:
                    self.logger.warning(f"{name}: {msg}")

            except Exception as e:
                self.logger.error(f"{name} failed: {e}")

        return results

    def list_features(self) -> list[str]:
        """登録済み特徴量一覧"""
        return list(self._features.keys())

    def list_enabled(self) -> list[str]:
        """有効な特徴量一覧"""
        return list(self._enabled)

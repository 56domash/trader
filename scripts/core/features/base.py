# core/features/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class FeatureMetadata:
    """特徴量メタデータ"""
    name: str                    # 例: "rsi_14"
    category: str                # momentum, volatility, microstructure
    version: str                 # "1.0"
    lookback_bars: int           # 計算に必要な過去足数
    expected_range: tuple        # (min, max) 期待値範囲
    description: str             # 説明文


class Feature(ABC):
    """特徴量基底クラス"""

    @property
    @abstractmethod
    def metadata(self) -> FeatureMetadata:
        """メタデータを返す"""
        pass

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        特徴量計算

        Args:
            data: OHLCV DataFrame (index: DatetimeIndex, UTC)
                  columns: open, high, low, close, volume
                  warmup期間を含む（08:55-10:05 JST分）

        Returns:
            pd.Series: 計算結果（生値、正規化前）
                      index: data.indexと同一
        """
        pass

    def validate_output(self, result: pd.Series) -> tuple[bool, str]:
        """
        出力の妥当性チェック

        Returns:
            (is_valid, message)
        """
        # NaN率チェック
        nan_ratio = result.isna().sum() / len(result)
        if nan_ratio > 0.5:
            return False, f"High NaN ratio: {nan_ratio:.1%}"

        # 範囲チェック
        min_val, max_val = self.metadata.expected_range
        if result.min() < min_val or result.max() > max_val:
            return False, f"Out of range: [{result.min():.2f}, {result.max():.2f}]"

        return True, "OK"

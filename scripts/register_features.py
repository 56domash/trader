# scripts/register_features.py

from core.features.registry import FeatureRegistry
from core.features.implementations.momentum import (
    RSI14Feature,
    MACDHistogramFeature
)
from core.features.implementations.volatility import (
    ATRPercentileFeature,
    BollingerPositionFeature
)


def register_all_features(registry: FeatureRegistry):
    """全特徴量を登録"""

    # Momentum
    registry.register(RSI14Feature())
    registry.register(MACDHistogramFeature())

    # Volatility
    registry.register(ATRPercentileFeature())
    registry.register(BollingerPositionFeature())

    # ... 他の特徴量

# scripts/register_features.py

from core.features.registry import FeatureRegistry
from core.features.implementations.momentum import (
    RSI14Feature,
    MACDHistogramFeature,
    WilliamsRFeature,
    StochasticKFeature
)

from core.features.implementations.microstructure import (
    VWAPDeviationFeature,
    VolumeSpikeFeature,
    VolumeImbalanceFeature,
    VolumeRatioFeature,
    OpeningRangePositionFeature,
    OpeningRangeBreakoutFeature
)


from core.features.implementations.volatility import (
    ATRPercentileFeature,
    BollingerPositionFeature,
    VolatilityRegimeFeature,

)

from core.features.implementations.cross_market import (
    USDJPYCorrelationFeature,
    Nikkei225CorrelationFeature,
    MarketBetaFeature,
    RelativeStrengthIndexFeature
)
from core.features.implementations.seasonality import (
    TimeOfDayFeature,
    EarlySessionBiasFeature,
    DayOfWeekFeature,
    MonthEndEffectFeature
)


def register_all_features(registry: FeatureRegistry):
    """全特徴量を登録"""

    # === Momentum ===
    registry.register(RSI14Feature())
    registry.register(MACDHistogramFeature())
    registry.register(WilliamsRFeature())
    registry.register(StochasticKFeature())

    # === Volatility ===
    registry.register(ATRPercentileFeature())
    registry.register(BollingerPositionFeature())
    registry.register(VolatilityRegimeFeature())

    # === Microstructure ===
    registry.register(VWAPDeviationFeature())
    registry.register(VolumeSpikeFeature())
    registry.register(VolumeImbalanceFeature())
    registry.register(VolumeRatioFeature())
    registry.register(OpeningRangePositionFeature())
    registry.register(OpeningRangeBreakoutFeature())

    # === Cross-Market ===
    registry.register(USDJPYCorrelationFeature())
    registry.register(Nikkei225CorrelationFeature())
    registry.register(MarketBetaFeature())
    registry.register(RelativeStrengthIndexFeature())

    # === Seasonality ===
    registry.register(TimeOfDayFeature())
    registry.register(EarlySessionBiasFeature())
    registry.register(DayOfWeekFeature())
    registry.register(MonthEndEffectFeature())

    print(f"✅ Registered {len(registry.list_features())} features")

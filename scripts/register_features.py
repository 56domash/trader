# scripts/register_features.py

from core.features.registry import FeatureRegistry
from core.features.implementations.momentum import (
    RSI14Feature,
    MACDHistogramFeature,
    WilliamsRFeature,
    StochasticKFeature,
    CCIFeature
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
from core.features.implementations.momentum import (
    # ... 既存
    ADXFeature,  # 追加
    IchimokuConversionFeature,  # 追加
)

from core.features.implementations.microstructure import (
    # ... 既存
    MoneyFlowIndexFeature,  # 追加
)

from core.features.implementations.volatility import (
    # ... 既存
    KeltnerChannelPositionFeature,
    KeltnerChannelWidthFeature,
)

from core.features.implementations.momentum import (
    # ... 既存
    AroonUpFeature,
    AroonDownFeature,
)

from core.features.implementations.microstructure import (
    # ... 既存
    GapOpenFeature,
    StreakUpFeature,
    StreakDownFeature,
)


def register_all_features(registry: FeatureRegistry):
    """全特徴量を登録"""

    # === Momentum ===
    registry.register(RSI14Feature())
    registry.register(MACDHistogramFeature())
    registry.register(WilliamsRFeature())
    registry.register(StochasticKFeature())
    registry.register(CCIFeature())

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
    registry.register(VWAPDeviationFeature())
    registry.register(MoneyFlowIndexFeature())  # 追加
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
    registry.register(ADXFeature())  # 追加
    registry.register(IchimokuConversionFeature())  # 追加

    # Volatility
    registry.register(KeltnerChannelPositionFeature())
    registry.register(KeltnerChannelWidthFeature())

    # Momentum
    registry.register(AroonUpFeature())
    registry.register(AroonDownFeature())

    # Microstructure
    registry.register(GapOpenFeature())
    registry.register(StreakUpFeature())
    registry.register(StreakDownFeature())

    print(f"✅ Registered {len(registry.list_features())} features")

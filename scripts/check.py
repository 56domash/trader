"""
重複登録されている特徴量を検出して修正
"""

from core.features.implementations.toyota_specific import *
from core.features.implementations.seasonality import *
from core.features.implementations.cross_market import *
from core.features.implementations.microstructure import *
from core.features.implementations.volatility import *
from core.features.implementations.momentum import *
from core.features.registry import FeatureRegistry
import sys
sys.path.append('scripts')


def detect_duplicates():
    """重複を検出"""

    registry = FeatureRegistry()
    registered_names = set()
    duplicates = []

    # 各カテゴリから特徴量を登録
    feature_classes = [
        # Momentum
        RSI14Feature,
        MACDHistogramFeature,
        WilliamsRFeature,
        StochasticKFeature,
        ADXFeature,
        # MFIFeature,
        IchimokuConversionFeature,
        CCIFeature,

        # Volatility
        ATRPercentileFeature,
        BollingerPositionFeature,
        VolatilityRegimeFeature,
        KeltnerChannelPositionFeature,
        KeltnerChannelWidthFeature,

        # Microstructure
        VWAPDeviationFeature,
        VolumeSpikeFeature,
        VolumeImbalanceFeature,
        VolumeRatioFeature,

        # Cross-market
        USDJPYCorrelationFeature,
        Nikkei225CorrelationFeature,
        # RelativeStrengthVsNikkeiFeature,
        MarketBetaFeature,

        # Seasonality
        TimeOfDayFeature,
        DayOfWeekFeature,
        MonthEndEffectFeature,
        EarlySessionBiasFeature,

        # Toyota-specific
        OpeningRangePositionFeature,
        OpeningRangeBreakoutFeature,
        GapOpenFeature,
        StreakUpFeature,
        StreakDownFeature,
    ]

    print("\n" + "="*70)
    print("Checking for duplicate feature registrations")
    print("="*70 + "\n")

    for feature_class in feature_classes:
        try:
            feature = feature_class()
            name = feature.metadata.name

            if name in registered_names:
                duplicates.append(name)
                print(f"⚠️  DUPLICATE: {name} ({feature_class.__name__})")
            else:
                registered_names.add(name)
                print(f"✅ {name:30s} ({feature_class.__name__})")

            registry.register(feature)

        except Exception as e:
            print(f"❌ Error registering {feature_class.__name__}: {e}")

    print("\n" + "="*70 + "\n")

    if duplicates:
        print(f"Found {len(duplicates)} duplicate(s):")
        for dup in duplicates:
            print(f"  - {dup}")
        print("\nAction needed:")
        print("  Check scripts/register_features.py")
        print("  Remove duplicate registrations")
    else:
        print("✅ No duplicates found!")

    print(f"\nTotal unique features: {len(registered_names)}")
    print()


if __name__ == "__main__":
    detect_duplicates()

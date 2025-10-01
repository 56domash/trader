# scripts/run_strategy.py

from datetime import datetime
from core.data.pipeline import DataPipeline
from core.features.registry import FeatureRegistry
from core.signals.scoring import FeatureScorer
from core.signals.aggregator import SignalAggregator
from scripts.register_features import register_all_features
from core.utils.config import load_config


def run_strategy(date: datetime, symbol: str = "7203.T"):
    """戦略実行"""

    # 1. 設定読み込み
    config = load_config("config/features.yaml")

    # 2. データ取得
    pipeline = DataPipeline()
    data = pipeline.fetch_trading_window(date, symbol)

    # 3. 特徴量計算
    registry = FeatureRegistry()
    register_all_features(registry)
    features = registry.compute_all(data.ohlcv)

    # 4. スコアリング
    scorer = FeatureScorer(config.scoring_configs)
    scores = scorer.transform(features)

    # 5. 統合
    aggregator = SignalAggregator(config.weights)
    signals = aggregator.aggregate(scores)

    return signals

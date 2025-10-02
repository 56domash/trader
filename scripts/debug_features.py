# scripts/debug_features.py
"""
V3特徴量の詳細デバッグ
各ステップでの値を確認
"""

from strategy_loop_v3 import jst_window_utc, load_bars_from_db
from core.signals.aggregator import SignalAggregator
from core.signals.scoring import FeatureScorer, ScoringConfig
from core.features.implementations.microstructure import VWAPDeviationFeature
from core.features.implementations.momentum import RSI14Feature, MACDHistogramFeature
from core.features.registry import FeatureRegistry
import argparse
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


JST = ZoneInfo("Asia/Tokyo")


def debug_features(db_path: str, symbol: str, target_date: datetime.date):
    """特徴量を段階的にデバッグ"""

    conn = sqlite3.connect(db_path)
    start_utc, end_utc = jst_window_utc(target_date, "09:00", "10:00")

    # データ読み込み
    bars = load_bars_from_db(conn, symbol, start_utc, end_utc)
    conn.close()

    if bars.empty:
        print(f"⚠️ データなし: {target_date}")
        return

    print(f"\n{'='*70}")
    print(f"Feature Debug: {target_date}")
    print(f"{'='*70}\n")

    print(f"📊 Data Info:")
    print(f"  Bars: {len(bars)}")
    print(
        f"  Close range: [{bars['close'].min():.1f}, {bars['close'].max():.1f}]")
    print(
        f"  Volume range: [{bars['volume'].min():.0f}, {bars['volume'].max():.0f}]")
    print()

    # === Step 1: 特徴量計算（生値） ===
    print("🔧 Step 1: Feature Computation (Raw Values)")
    print("-" * 70)

    registry = FeatureRegistry()
    registry.register(RSI14Feature())
    registry.register(MACDHistogramFeature())
    registry.register(VWAPDeviationFeature())

    features = registry.compute_all(bars)

    for name, values in features.items():
        print(f"\n{name}:")
        print(f"  Range: [{values.min():.6f}, {values.max():.6f}]")
        print(f"  Mean: {values.mean():.6f}")
        print(f"  Std: {values.std():.6f}")
        print(f"  NaN count: {values.isna().sum()} / {len(values)}")
        print(f"  Sample (last 5): {values.tail().values}")

    print()

    # === Step 2: スコアリング ===
    print("🔧 Step 2: Feature Scoring (0-1 Normalization)")
    print("-" * 70)

    scoring_config = {
        "rsi_14": ScoringConfig(method="direct_scale", direction="bullish"),
        "macd_histogram": ScoringConfig(method="tanh_normalize", direction="bullish", params={"window": 60}),
        "vwap_deviation": ScoringConfig(method="tanh_normalize", direction="bullish", params={"window": 20})
    }

    scorer = FeatureScorer(scoring_config)
    scores = scorer.transform(features)

    for name, values in scores.items():
        print(f"\n{name} (scored):")
        print(f"  Range: [{values.min():.6f}, {values.max():.6f}]")
        print(f"  Mean: {values.mean():.6f}")
        print(f"  Std: {values.std():.6f}")
        print(f"  Sample (last 5): {values.tail().values}")

    print()

    # === Step 3: 統合 ===
    print("🔧 Step 3: Signal Aggregation")
    print("-" * 70)

    weights = {
        "rsi_14": 1.2,
        "macd_histogram": 1.0,
        "vwap_deviation": 1.1
    }

    aggregator = SignalAggregator(weights)
    signals = aggregator.aggregate(scores)

    print(f"\nS (統合シグナル):")
    print(f"  Range: [{signals['S'].min():.6f}, {signals['S'].max():.6f}]")
    print(f"  Mean: {signals['S'].mean():.6f}")
    print(f"  Std: {signals['S'].std():.6f}")

    # 寄与度確認
    contrib_cols = [c for c in signals.columns if c.startswith('contrib_')]
    print(f"\n寄与度:")
    for col in contrib_cols:
        feat_name = col.replace('contrib_', '')
        mean_contrib = signals[col].mean()
        max_contrib = signals[col].abs().max()
        print(f"  {feat_name}: mean={mean_contrib:+.6f}, max={max_contrib:.6f}")

    print()

    # === Step 4: 詳細タイムライン ===
    print("🔧 Step 4: Detailed Timeline (JST)")
    print("-" * 70)
    print(f"{'Time':<8} {'RSI_raw':<10} {'RSI_score':<10} {'MACD_raw':<10} {'MACD_score':<10} {'VWAP_raw':<10} {'VWAP_score':<10} {'S':<8}")
    print("-" * 100)

    for i in range(0, len(bars), 10):
        ts = bars.index[i]
        ts_jst = ts.tz_convert(JST)

        rsi_raw = features['rsi_14'].iloc[i] if i < len(
            features['rsi_14']) else 0
        rsi_score = scores['rsi_14'].iloc[i] if i < len(
            scores['rsi_14']) else 0

        macd_raw = features['macd_histogram'].iloc[i] if i < len(
            features['macd_histogram']) else 0
        macd_score = scores['macd_histogram'].iloc[i] if i < len(
            scores['macd_histogram']) else 0

        vwap_raw = features['vwap_deviation'].iloc[i] if i < len(
            features['vwap_deviation']) else 0
        vwap_score = scores['vwap_deviation'].iloc[i] if i < len(
            scores['vwap_deviation']) else 0

        s_val = signals['S'].iloc[i] if i < len(signals['S']) else 0

        print(f"{ts_jst.strftime('%H:%M'):<8} "
              f"{rsi_raw:>9.2f} {rsi_score:>9.4f} "
              f"{macd_raw:>9.4f} {macd_score:>9.4f} "
              f"{vwap_raw:>9.6f} {vwap_score:>9.4f} "
              f"{s_val:>7.4f}")


def main():
    parser = argparse.ArgumentParser(description="V3 Feature Debug")
    parser.add_argument("--db", default="runtime.db", help="Database path")
    parser.add_argument("--symbol", default="7203.T", help="Symbol")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD in JST)")
    parser.add_argument("--use-last-session", action="store_true")

    args = parser.parse_args()

    # 日付決定
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    elif args.use_last_session:
        conn = sqlite3.connect(args.db)
        row = conn.execute(
            "SELECT ts FROM bars_1m WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (args.symbol,)
        ).fetchone()
        conn.close()

        if row:
            target_date = pd.to_datetime(
                row[0], utc=True).tz_convert(JST).date()
        else:
            target_date = datetime.now(JST).date()
    else:
        target_date = datetime.now(JST).date()

    debug_features(args.db, args.symbol, target_date)


if __name__ == "__main__":
    main()

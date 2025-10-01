# scripts/strategy_loop_v3.py
"""
V3 Strategy Loop
既存DBのbars_1mから読み込み、V3パイプラインで計算、signals_1mに書き込み
"""
from core.signals.decision import DecisionEngine, ThresholdConfig
from core.signals.aggregator import SignalAggregator
from core.signals.scoring import FeatureScorer, ScoringConfig
from core.features.implementations.microstructure import VWAPDeviationFeature
from core.features.implementations.momentum import RSI14Feature, MACDHistogramFeature
from core.features.registry import FeatureRegistry
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime, time, timedelta
import argparse
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


JST = ZoneInfo("Asia/Tokyo")


def jst_window_utc(date_jst: datetime.date, start_hm="09:00", end_hm="10:00"):
    """JST日付から UTC窓を計算"""
    s_h, s_m = map(int, start_hm.split(":"))
    e_h, e_m = map(int, end_hm.split(":"))

    s_jst = datetime.combine(date_jst, time(s_h, s_m), tzinfo=JST)
    e_jst = datetime.combine(date_jst, time(e_h, e_m), tzinfo=JST)

    return s_jst.astimezone(ZoneInfo("UTC")), e_jst.astimezone(ZoneInfo("UTC"))


def load_bars_from_db(conn, symbol, start_utc, end_utc):
    """既存DBから bars_1m を読み込み（warmup込み）"""
    # 特徴量計算用に10分前から取得
    warmup_start = start_utc - timedelta(minutes=10)

    query = """
    SELECT ts, open, high, low, close, volume
    FROM bars_1m
    WHERE symbol = ? AND ts >= ? AND ts < ?
    ORDER BY ts
    """
    df = pd.read_sql(
        query, conn,
        params=(symbol, warmup_start.isoformat(), end_utc.isoformat()),
        parse_dates=['ts']
    )

    if df.empty:
        return df

    df.set_index('ts', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)

    return df


def upsert_v3_signals(conn, symbol, decisions, contributions):
    """V3シグナルを signals_1m に保存"""
    rows_updated = 0

    for ts, row in decisions.iterrows():
        # 寄与度を取得
        contrib_rsi = contributions.get(
            'contrib_rsi_14', pd.Series()).get(ts, None)
        contrib_macd = contributions.get(
            'contrib_macd_histogram', pd.Series()).get(ts, None)
        contrib_vwap = contributions.get(
            'contrib_vwap_deviation', pd.Series()).get(ts, None)

        conn.execute("""
            INSERT INTO signals_1m (
                symbol, ts, 
                V3_S, V3_S_ema, V3_action, V3_can_long, V3_can_short,
                V3_contrib_rsi, V3_contrib_macd, V3_contrib_vwap
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, ts) DO UPDATE SET
                V3_S = excluded.V3_S,
                V3_S_ema = excluded.V3_S_ema,
                V3_action = excluded.V3_action,
                V3_can_long = excluded.V3_can_long,
                V3_can_short = excluded.V3_can_short,
                V3_contrib_rsi = excluded.V3_contrib_rsi,
                V3_contrib_macd = excluded.V3_contrib_macd,
                V3_contrib_vwap = excluded.V3_contrib_vwap
        """, (
            symbol,
            ts.isoformat(),
            float(row['S']),
            float(row['S_ema']),
            str(row['action']),
            int(row.get('can_long', 0)),
            int(row.get('can_short', 0)),
            float(contrib_rsi) if contrib_rsi is not None else None,
            float(contrib_macd) if contrib_macd is not None else None,
            float(contrib_vwap) if contrib_vwap is not None else None,
        ))
        rows_updated += 1

    conn.commit()
    return rows_updated


def run_v3_strategy(db_path, symbol, target_date, verbose=False):
    """V3戦略を実行"""
    conn = sqlite3.connect(db_path, timeout=10_000)

    try:
        # 1. 窓計算（JST 09:00-10:00）
        start_utc, end_utc = jst_window_utc(target_date, "09:00", "10:00")

        if verbose:
            print(f"\n{'='*60}")
            print(f"V3 Strategy Loop")
            print(f"{'='*60}")
            print(f"Date: {target_date} (JST)")
            print(f"Window: {start_utc} - {end_utc} (UTC)")

        # 2. データ読み込み
        bars = load_bars_from_db(conn, symbol, start_utc, end_utc)

        if bars.empty:
            print(f"⚠️ データなし: {target_date}")
            return

        if verbose:
            print(f"\n✓ データ読み込み: {len(bars)} bars")

        # 3. Feature計算
        registry = FeatureRegistry()
        registry.register(RSI14Feature())
        registry.register(MACDHistogramFeature())
        registry.register(VWAPDeviationFeature())

        features = registry.compute_all(bars)

        if verbose:
            print(f"✓ 特徴量計算: {len(features)} features")

        # 4. スコアリング
        scoring_config = {
            "rsi_14": ScoringConfig(method="direct_scale", direction="bullish"),
            "macd_histogram": ScoringConfig(method="tanh_normalize", direction="bullish", params={"window": 60}),
            "vwap_deviation": ScoringConfig(method="tanh_normalize", direction="bullish", params={"window": 20})
        }
        scorer = FeatureScorer(scoring_config)
        scores = scorer.transform(features)

        # 5. 統合
        weights = {
            "rsi_14": 1.2,
            "macd_histogram": 1.0,
            "vwap_deviation": 1.1
        }
        aggregator = SignalAggregator(weights)
        signals = aggregator.aggregate(scores)

        if verbose:
            print(
                f"✓ シグナル統合: S範囲=[{signals['S'].min():.3f}, {signals['S'].max():.3f}]")

        # 6. 判定
        threshold_config = ThresholdConfig(
            thr_long=0.15,
            thr_short=-0.15,
            exit_long=0.05,
            exit_short=-0.05,
            confirm_bars=2,
            ema_span=3
        )
        engine = DecisionEngine(threshold_config)
        decisions = engine.decide(signals, current_position=0)

        # 出力窓でフィルタ
        decisions = decisions[(decisions.index >= start_utc)
                              & (decisions.index < end_utc)]

        if verbose:
            entry_long = (decisions['action'] == 'ENTRY_LONG').sum()
            entry_short = (decisions['action'] == 'ENTRY_SHORT').sum()
            print(f"✓ 判定完了: LONG={entry_long}, SHORT={entry_short}")

        # 7. DB書き込み
        contributions = {
            col: signals[col] for col in signals.columns if col.startswith('contrib_')
        }

        n = upsert_v3_signals(conn, symbol, decisions, contributions)

        if verbose:
            print(f"\n✅ DB書き込み完了: {n} rows")

        return n

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="V3 Strategy Loop")
    parser.add_argument("--db", default="runtime.db", help="Database path")
    parser.add_argument("--symbol", default="7203.T", help="Symbol")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD in JST)")
    parser.add_argument("--use-last-session",
                        action="store_true", help="Use last session date")
    parser.add_argument("--verbose", "-v",
                        action="store_true", help="Verbose output")

    args = parser.parse_args()

    # 日付決定
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    elif args.use_last_session:
        # 最新データの日付を使用
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

    # 実行
    run_v3_strategy(args.db, args.symbol, target_date, verbose=args.verbose)


if __name__ == "__main__":
    main()

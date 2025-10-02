# scripts/analyze_v3_signals.py
"""
V3シグナルの詳細分析
なぜトレードが発生しないのかを調査
"""

import argparse
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from strategy_loop_v3 import jst_window_utc

JST = ZoneInfo("Asia/Tokyo")


def analyze_signals(db_path: str, symbol: str, target_date: datetime.date):
    """指定日のV3シグナルを詳細分析"""

    conn = sqlite3.connect(db_path)
    start_utc, end_utc = jst_window_utc(target_date, "09:00", "10:00")

    # V3シグナル取得
    query = """
    SELECT 
        ts, 
        V3_S, V3_S_ema, V3_action, V3_can_long, V3_can_short,
        V3_contrib_rsi, V3_contrib_macd, V3_contrib_vwap
    FROM signals_1m
    WHERE symbol = ? AND ts >= ? AND ts < ?
    ORDER BY ts
    """

    df = pd.read_sql(
        query, conn,
        params=(symbol, start_utc.isoformat(), end_utc.isoformat()),
        parse_dates=['ts']
    )

    conn.close()

    if df.empty:
        print(f"⚠️ シグナルが見つかりません: {target_date}")
        return

    df.set_index('ts', inplace=True)

    print(f"\n{'='*70}")
    print(f"V3 Signal Analysis: {target_date} (JST 09:00-10:00)")
    print(f"{'='*70}\n")

    # 基本統計
    print("📊 Signal Statistics:")
    print(f"  Total bars: {len(df)}")
    print(f"  V3_S range: [{df['V3_S'].min():.3f}, {df['V3_S'].max():.3f}]")
    print(f"  V3_S mean: {df['V3_S'].mean():.3f}")
    print(f"  V3_S std: {df['V3_S'].std():.3f}")
    print(
        f"  V3_S_ema range: [{df['V3_S_ema'].min():.3f}, {df['V3_S_ema'].max():.3f}]")
    print()

    # アクション統計
    print("🎯 Action Statistics:")
    action_counts = df['V3_action'].value_counts()
    for action, count in action_counts.items():
        print(f"  {action}: {count} times")
    print()

    # Threshold通過回数
    print("📈 Threshold Analysis:")
    thr_long = 0.15
    thr_short = -0.15
    exit_long = 0.05
    exit_short = -0.05

    above_long = (df['V3_S_ema'] >= thr_long).sum()
    below_short = (df['V3_S_ema'] <= thr_short).sum()
    near_neutral = ((df['V3_S_ema'] > exit_short) &
                    (df['V3_S_ema'] < exit_long)).sum()

    print(f"  V3_S_ema >= {thr_long} (LONG threshold): {above_long} bars")
    print(f"  V3_S_ema <= {thr_short} (SHORT threshold): {below_short} bars")
    print(
        f"  In neutral zone [{exit_short}, {exit_long}]: {near_neutral} bars")
    print()

    # 寄与度分析
    print("🔍 Feature Contributions:")
    if df['V3_contrib_rsi'].notna().any():
        print(f"  RSI avg: {df['V3_contrib_rsi'].mean():.4f}")
        print(f"  MACD avg: {df['V3_contrib_macd'].mean():.4f}")
        print(f"  VWAP avg: {df['V3_contrib_vwap'].mean():.4f}")
    else:
        print("  ⚠️ Contribution data not available")
    print()

    # 最高/最低シグナル
    print("🏆 Extreme Signals:")
    max_idx = df['V3_S_ema'].idxmax()
    min_idx = df['V3_S_ema'].idxmin()

    print(
        f"  Highest V3_S_ema: {df.loc[max_idx, 'V3_S_ema']:.3f} at {max_idx.tz_convert(JST).strftime('%H:%M')}")
    print(f"    Action: {df.loc[max_idx, 'V3_action']}")

    print(
        f"  Lowest V3_S_ema: {df.loc[min_idx, 'V3_S_ema']:.3f} at {min_idx.tz_convert(JST).strftime('%H:%M')}")
    print(f"    Action: {df.loc[min_idx, 'V3_action']}")
    print()

    # タイムシリーズ表示（サンプル）
    print("📅 Sample Timeline (every 10 minutes):")
    print(f"{'Time (JST)':<12} {'V3_S':<8} {'V3_S_ema':<10} {'Action':<12} {'RSI':<8} {'MACD':<8} {'VWAP':<8}")
    print("-" * 80)

    for i in range(0, len(df), 10):
        row = df.iloc[i]
        ts_jst = row.name.tz_convert(JST)

        rsi = row['V3_contrib_rsi'] if pd.notna(row['V3_contrib_rsi']) else 0.0
        macd = row['V3_contrib_macd'] if pd.notna(
            row['V3_contrib_macd']) else 0.0
        vwap = row['V3_contrib_vwap'] if pd.notna(
            row['V3_contrib_vwap']) else 0.0

        print(f"{ts_jst.strftime('%H:%M'):<12} "
              f"{row['V3_S']:>7.3f} "
              f"{row['V3_S_ema']:>9.3f} "
              f"{row['V3_action']:<12} "
              f"{rsi:>7.4f} "
              f"{macd:>7.4f} "
              f"{vwap:>7.4f}")
    print()

    # 推奨事項
    print("💡 Recommendations:")

    if above_long == 0 and below_short == 0:
        print("  ⚠️ シグナルがthresholdに到達していません")
        print("  → threshold を緩和することを検討 (例: 0.15 → 0.10)")
        print("  → または特徴量の追加を検討")

    if df['V3_S'].std() < 0.05:
        print("  ⚠️ シグナルの変動が小さすぎます")
        print("  → 特徴量の重みを調整")
        print("  → またはスコアリング方法を見直し")

    if (df['V3_action'] == 'HOLD').sum() == len(df):
        print("  ⚠️ 全てHOLDアクションです")
        print("  → confirm_bars を減らす (2 → 1)")
        print("  → EMA span を減らす (3 → 2)")


def compare_with_legacy(db_path: str, symbol: str, target_date: datetime.date):
    """V3とLegacyシグナルを比較"""

    conn = sqlite3.connect(db_path)
    start_utc, end_utc = jst_window_utc(target_date, "09:00", "10:00")

    query = """
    SELECT ts, V3_S_ema, S
    FROM signals_1m
    WHERE symbol = ? AND ts >= ? AND ts < ?
    ORDER BY ts
    """

    df = pd.read_sql(
        query, conn,
        params=(symbol, start_utc.isoformat(), end_utc.isoformat()),
        parse_dates=['ts']
    )

    conn.close()

    if df.empty or df['V3_S_ema'].isna().all() or df['S'].isna().all():
        print("⚠️ 比較データが不足しています")
        return

    print(f"\n{'='*70}")
    print(f"V3 vs Legacy Comparison")
    print(f"{'='*70}\n")

    # 相関
    corr = df[['V3_S_ema', 'S']].corr().iloc[0, 1]
    print(f"📊 Correlation: {corr:.3f}")

    # 統計比較
    print(f"\n統計比較:")
    print(f"{'Metric':<20} {'V3':<15} {'Legacy':<15}")
    print("-" * 50)
    print(
        f"{'Mean':<20} {df['V3_S_ema'].mean():>14.3f} {df['S'].mean():>14.3f}")
    print(f"{'Std':<20} {df['V3_S_ema'].std():>14.3f} {df['S'].std():>14.3f}")
    print(f"{'Min':<20} {df['V3_S_ema'].min():>14.3f} {df['S'].min():>14.3f}")
    print(f"{'Max':<20} {df['V3_S_ema'].max():>14.3f} {df['S'].max():>14.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="V3 Signal Analysis")
    parser.add_argument("--db", default="runtime.db", help="Database path")
    parser.add_argument("--symbol", default="7203.T", help="Symbol")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD in JST)")
    parser.add_argument("--use-last-session", action="store_true",
                        help="Use last session date")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with legacy signals")

    args = parser.parse_args()

    # 日付決定
    conn = sqlite3.connect(args.db)

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    elif args.use_last_session:
        row = conn.execute(
            "SELECT ts FROM signals_1m WHERE symbol=? AND V3_S IS NOT NULL ORDER BY ts DESC LIMIT 1",
            (args.symbol,)
        ).fetchone()
        conn.close()

        if row:
            target_date = pd.to_datetime(
                row[0], utc=True).tz_convert(JST).date()
        else:
            print("⚠️ V3シグナルが見つかりません")
            return
    else:
        target_date = datetime.now(JST).date()

    conn.close()

    # 分析実行
    analyze_signals(args.db, args.symbol, target_date)

    if args.compare:
        compare_with_legacy(args.db, args.symbol, target_date)


if __name__ == "__main__":
    main()

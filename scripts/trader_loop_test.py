# trader_loop_test.py（テスト用）
"""
V3シグナルを読み込んでトレード判定をテストする
"""
import sqlite3
import pandas as pd


def test_v3_trading_signals(db_path="runtime.db", symbol="7203.T", date="2025-09-22"):
    conn = sqlite3.connect(db_path)

    print("="*60)
    print("V3シグナルでのトレーディングシミュレーション")
    print("="*60)

    # V3シグナル取得
    query = """
    SELECT 
        ts,
        V3_S_ema,
        V3_action,
        V3_can_long,
        V3_can_short
    FROM signals_1m
    WHERE symbol = ?
      AND date(ts) = ?
      AND V3_S IS NOT NULL
    ORDER BY ts
    """
    signals = pd.read_sql(query, conn, params=(
        symbol, date), parse_dates=['ts'])

    # 価格データ取得
    query = """
    SELECT ts, close
    FROM bars_1m
    WHERE symbol = ?
      AND date(ts) = ?
    ORDER BY ts
    """
    prices = pd.read_sql(query, conn, params=(
        symbol, date), parse_dates=['ts'])

    # マージ
    df = signals.merge(prices, on='ts', how='inner')

    print(f"\nデータ: {len(df)} bars")
    print(
        f"V3_S_ema範囲: [{df['V3_S_ema'].min():.3f}, {df['V3_S_ema'].max():.3f}]")

    # エントリーポイント
    entries = df[df['V3_action'].isin(['ENTRY_LONG', 'ENTRY_SHORT'])]

    print(f"\nエントリー機会: {len(entries)} 回")
    if not entries.empty:
        print("\nエントリーポイント:")
        print(entries[['ts', 'V3_action', 'V3_S_ema', 'close']
                      ].to_string(index=False))

    conn.close()

    print("\n✅ シミュレーション完了")


if __name__ == "__main__":
    test_v3_trading_signals()

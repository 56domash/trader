# scripts/demo_trading.py
"""
トレーディングシステム デモ実行
モックモードで完全動作確認
"""

import time
from tq.io_kabu import create_kabu_client
import sys
import os
from pathlib import Path

# scripts/ フォルダをパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


def demo_basic_trading():
    """基本的な発注デモ"""
    print("\n" + "="*60)
    print("Trading System Demo - Basic Operations")
    print("="*60 + "\n")

    # モッククライアント作成
    client = create_kabu_client(use_mock=True, db_path="runtime.db")

    # 1. 現在価格取得
    print("1. 現在価格取得")
    price = client.get_current_price("7203.T")
    print(f"   Toyota (7203.T): {price:.1f} 円\n")

    time.sleep(1)

    # 2. 買い注文
    print("2. 買い注文")
    order_id1 = client.place_order("7203.T", "BUY", 100, price, "LIMIT")
    print(f"   注文ID: {order_id1}\n")

    time.sleep(1)

    # 3. ポジション確認
    print("3. ポジション確認")
    positions = client.get_positions()
    for pos in positions:
        print(f"   {pos['side']} {pos['size']}株 @ {pos['avg_price']:.1f}")
    print()

    time.sleep(1)

    # 4. 価格変動
    print("4. 価格変動シミュレーション")
    new_price = client.get_current_price("7203.T")
    print(f"   新価格: {new_price:.1f} 円")
    print(f"   変動: {new_price - price:+.1f} 円\n")

    time.sleep(1)

    # 5. 決済注文
    print("5. 決済注文")
    order_id2 = client.place_order("7203.T", "SELL", 100, new_price, "LIMIT")
    print(f"   注文ID: {order_id2}\n")

    time.sleep(1)

    # 6. 約定一覧
    print("6. 約定一覧")
    if hasattr(client, 'get_fills'):
        fills = client.get_fills()
        for i, fill in enumerate(fills, 1):
            profit = ""
            if i == 2:  # 決済時
                pnl = (new_price - price) * 100
                profit = f" → P&L: {pnl:+,.0f} 円"
            print(
                f"   [{i}] {fill['side']} {fill['size']}株 @ {fill['fill_price']:.1f}{profit}")
    print()

    print("="*60)
    print("Demo completed successfully!")
    print("="*60 + "\n")


def demo_strategy_trading():
    """戦略ベースのデモ"""
    print("\n" + "="*60)
    print("Trading System Demo - Strategy Simulation")
    print("="*60 + "\n")

    import sqlite3
    import pandas as pd

    db_path = "runtime.db"
    client = create_kabu_client(use_mock=True, db_path=db_path)

    # シグナル確認
    print("1. V3シグナル確認")
    conn = sqlite3.connect(db_path)

    query = """
    SELECT ts, V3_S, V3_action 
    FROM signals_1m 
    WHERE symbol='7203.T' 
    ORDER BY ts DESC 
    LIMIT 5
    """

    df = pd.read_sql(query, conn)

    if not df.empty:
        print(f"   最新5件:")
        for _, row in df.iterrows():
            ts = pd.to_datetime(row['ts']).strftime('%Y-%m-%d %H:%M')
            print(f"   {ts} | S={row['V3_S']:.3f} | Action={row['V3_action']}")
    else:
        print("   シグナルなし")

    conn.close()
    print()

    # 仮想トレード
    print("2. シグナルに基づく仮想トレード")

    if not df.empty and df.iloc[0]['V3_action'] in ['ENTRY_LONG', 'ENTRY_SHORT']:
        action = df.iloc[0]['V3_action']
        side = "BUY" if action == "ENTRY_LONG" else "SELL"

        price = client.get_current_price("7203.T")

        print(f"   シグナル: {action}")
        print(f"   価格: {price:.1f}")

        order_id = client.place_order("7203.T", side, 100, price, "LIMIT")
        print(f"   注文ID: {order_id}")

        print("\n   ポジション:")
        positions = client.get_positions()
        for pos in positions:
            print(f"   {pos['side']} {pos['size']}株 @ {pos['avg_price']:.1f}")

    print("\n" + "="*60)
    print("Strategy demo completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading Demo")
    parser.add_argument("--mode", choices=["basic", "strategy"], default="basic",
                        help="Demo mode")

    args = parser.parse_args()

    if args.mode == "basic":
        demo_basic_trading()
    elif args.mode == "strategy":
        demo_strategy_trading()

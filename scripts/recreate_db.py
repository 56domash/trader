"""
データベースを完全にリセットして、Legacy + V3のテーブルを作成
"""
import sqlite3
import argparse
from datetime import datetime


def reset_database(db_path: str, confirm: bool = False):
    """データベースを完全リセット"""

    if not confirm:
        print("\n⚠️  WARNING: This will DELETE ALL DATA in the database!")
        response = input("Are you sure? Type 'yes' to continue: ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return

    print(f"\n{'='*70}")
    print(f"Resetting Database: {db_path}")
    print(f"{'='*70}\n")

    conn = sqlite3.connect(db_path, timeout=10_000)
    cursor = conn.cursor()

    # 既存テーブルを全て取得
    print("1. Dropping all existing tables...")
    tables = cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """).fetchall()

    for (table,) in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        print(f"   🗑️  Dropped: {table}")

    print()

    # ========================================
    # Legacy テーブル作成
    # ========================================
    print("2. Creating Legacy tables...")

    # bars_1m（市場データ）
    print("   Creating bars_1m...")
    cursor.execute("""
    CREATE TABLE bars_1m (
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume INTEGER NOT NULL,
        PRIMARY KEY (symbol, ts)
    )
    """)

    # signals_1m（Legacy + V3 シグナル）
    print("   Creating signals_1m...")
    cursor.execute("""
    CREATE TABLE signals_1m (
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        -- Legacy columns
        S REAL,
        S_buy REAL,
        S_sell REAL,
        -- V3 columns
        V3_S REAL,
        V3_S_ema REAL,
        V3_action TEXT,
        PRIMARY KEY (symbol, ts)
    )
    """)

    # fills（約定記録）
    print("   Creating fills...")
    cursor.execute("""
    CREATE TABLE fills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        side TEXT NOT NULL,
        size INTEGER NOT NULL,
        price REAL NOT NULL,
        pnl REAL DEFAULT 0.0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # orders（発注記録）
    print("   Creating orders...")
    cursor.execute("""
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        size INTEGER NOT NULL,
        price REAL,
        status TEXT DEFAULT 'PENDING',
        created_at TEXT NOT NULL,
        updated_at TEXT
    )
    """)

    print()

    # ========================================
    # V3 テーブル作成
    # ========================================
    print("3. Creating V3 tables...")

    # feature_values（特徴量の値）
    print("   Creating feature_values...")
    cursor.execute("""
    CREATE TABLE feature_values (
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        feature_name TEXT NOT NULL,
        value REAL,
        score REAL,
        quality_score REAL DEFAULT 1.0,
        computation_time_ms REAL,
        PRIMARY KEY (symbol, ts, feature_name)
    )
    """)

    # feature_contributions（特徴量の寄与度）
    print("   Creating feature_contributions...")
    cursor.execute("""
    CREATE TABLE feature_contributions (
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        feature_name TEXT NOT NULL,
        contribution REAL,
        weight REAL,
        PRIMARY KEY (symbol, ts, feature_name)
    )
    """)

    # v3_positions（V3のポジション管理）
    print("   Creating v3_positions...")
    cursor.execute("""
    CREATE TABLE v3_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        size INTEGER NOT NULL,
        entry_time TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_time TEXT,
        exit_price REAL,
        pnl REAL DEFAULT 0.0,
        status TEXT DEFAULT 'OPEN',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # trading_status（リスク管理用）
    print("   Creating trading_status...")
    cursor.execute("""
    CREATE TABLE trading_status (
        symbol TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    print()

    # ========================================
    # インデックス作成
    # ========================================
    print("4. Creating indexes...")

    indexes = [
        # bars_1m indexes
        ("idx_bars_ts", "bars_1m", "ts"),
        ("idx_bars_symbol_ts", "bars_1m", "symbol, ts"),

        # signals_1m indexes
        ("idx_signals_ts", "signals_1m", "ts"),
        ("idx_signals_symbol_ts", "signals_1m", "symbol, ts"),

        # fills indexes
        ("idx_fills_ts", "fills", "ts"),
        ("idx_fills_symbol", "fills", "symbol"),

        # orders indexes
        ("idx_orders_status", "orders", "status"),
        ("idx_orders_created", "orders", "created_at"),

        # feature_values indexes
        ("idx_features_ts", "feature_values", "ts"),
        ("idx_features_name", "feature_values", "feature_name"),
        ("idx_features_symbol_ts", "feature_values", "symbol, ts"),

        # feature_contributions indexes
        ("idx_contrib_ts", "feature_contributions", "ts"),
        ("idx_contrib_name", "feature_contributions", "feature_name"),

        # v3_positions indexes
        ("idx_v3pos_status", "v3_positions", "status"),
        ("idx_v3pos_entry", "v3_positions", "entry_time"),
        ("idx_v3pos_symbol", "v3_positions", "symbol"),
    ]

    for idx_name, table, columns in indexes:
        cursor.execute(f"CREATE INDEX {idx_name} ON {table}({columns})")
        print(f"   ✅ Created: {idx_name}")

    print()

    # WALモード有効化
    print("5. Enabling WAL mode...")
    cursor.execute("PRAGMA journal_mode=WAL")
    print("   ✅ WAL mode enabled")

    conn.commit()
    conn.close()

    print(f"\n{'='*70}")
    print("✅ Database reset completed!")
    print(f"{'='*70}\n")

    # 確認
    print("Verifying tables...")
    conn = sqlite3.connect(db_path)
    tables = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table'
        ORDER BY name
    """).fetchall()

    print("\nCreated tables:")
    for (table,) in tables:
        print(f"  ✅ {table}")

    conn.close()

    print(f"\n{'='*70}\n")
    print("Next steps:")
    print("  1. Ingest market data:")
    print("     python scripts/ingest_loop.py --config config.toyota.yaml")
    print()
    print("  2. Generate Legacy signals:")
    print("     python scripts/strategy_loop.py --config config.toyota.yaml")
    print()
    print("  3. Generate V3 signals:")
    print("     python scripts/strategy_loop_v3.py --db runtime.db --date 2024-10-01")
    print()
    print("  4. Check status:")
    print("     python scripts/quick_db_check.py")
    print()


def main():
    parser = argparse.ArgumentParser(description="Reset database")
    parser.add_argument("--db", default="runtime.db", help="Database path")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    reset_database(args.db, confirm=args.yes)


if __name__ == "__main__":
    main()

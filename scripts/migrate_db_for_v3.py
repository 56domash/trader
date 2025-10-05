#!/usr/bin/env python3
"""
Database Migration: Add V3 columns to existing tables
既存のruntime.dbにV3用のカラムを追加
"""

import argparse
import sqlite3
from datetime import datetime


def migrate_to_v3(db_path: str):
    """V3用のカラムを追加"""

    print(f"\n{'='*60}")
    print(f"Database Migration to V3")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Time: {datetime.now()}")
    print(f"{'='*60}\n")

    conn = sqlite3.connect(db_path, timeout=10_000)
    cursor = conn.cursor()

    try:
        # バックアップ推奨メッセージ
        print("⚠️  Recommendation: Backup your database before migration")
        print(
            f"   cp {db_path} {db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}\n")

        response = input("Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Migration cancelled.")
            return

        # ===== 1. signals_1m テーブルにV3カラム追加 =====
        print("\n[1/4] Adding V3 columns to signals_1m...")

        v3_columns = [
            ("V3_S", "REAL"),
            ("V3_S_ema", "REAL"),
            ("V3_can_long", "INTEGER DEFAULT 0"),
            ("V3_can_short", "INTEGER DEFAULT 0"),
            ("V3_action", "TEXT DEFAULT 'HOLD'"),
        ]

        # 寄与度カラム（現在の11特徴量分）
        contrib_columns = [
            "V3_contrib_rsi",
            "V3_contrib_macd",
            "V3_contrib_vwap",
            "V3_contrib_opening_range",
            "V3_contrib_atr",
            "V3_contrib_bollinger",
            "V3_contrib_williams",
            "V3_contrib_stochastic",
            "V3_contrib_volume_spike",
            "V3_contrib_volume_imbalance",
            "V3_contrib_volume_ratio",
        ]

        # 寄与度カラムを追加
        for col_name in contrib_columns:
            v3_columns.append((col_name, "REAL DEFAULT 0.0"))

        for col_name, col_type in v3_columns:
            try:
                cursor.execute(
                    f"ALTER TABLE signals_1m ADD COLUMN {col_name} {col_type}")
                print(f"  ✓ Added: {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"  ⊙ Already exists: {col_name}")
                else:
                    raise

        # ===== 2. feature_contributions テーブル作成 =====
        print("\n[2/4] Creating feature_contributions table...")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_contributions (
            symbol TEXT NOT NULL,
            ts TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            contribution REAL,
            weight REAL,
            PRIMARY KEY (symbol, ts, feature_name)
        )
        """)
        print("  ✓ Table created/verified")

        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_contrib_ts 
        ON feature_contributions(ts)
        """)
        print("  ✓ Index created")

        # ===== 3. v3_positions テーブル作成 =====
        print("\n[3/4] Creating v3_positions table...")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS v3_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            size INTEGER NOT NULL,
            pnl REAL,
            status TEXT DEFAULT 'OPEN',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)
        print("  ✓ Table created/verified")

        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_v3_pos_symbol 
        ON v3_positions(symbol, entry_time)
        """)
        print("  ✓ Index created")

        # ===== 4. trading_status テーブル作成 =====
        print("\n[4/4] Creating trading_status table...")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trading_status (
            symbol TEXT PRIMARY KEY,
            status TEXT,
            updated_at TEXT
        )
        """)
        print("  ✓ Table created/verified")

        # コミット
        conn.commit()

        # 確認
        print("\n" + "="*60)
        print("Migration completed successfully!")
        print("="*60)

        # テーブル情報確認
        print("\nVerifying signals_1m schema...")
        cursor.execute("PRAGMA table_info(signals_1m)")
        columns = cursor.fetchall()
        v3_cols = [col for col in columns if col[1].startswith('V3_')]

        print(f"  V3 columns found: {len(v3_cols)}")
        for col in v3_cols:
            print(f"    - {col[1]} ({col[2]})")

        print("\n✅ Database is ready for V3 system!")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()


def verify_schema(db_path: str):
    """スキーマ確認"""

    conn = sqlite3.connect(db_path, timeout=10_000)
    cursor = conn.cursor()

    try:
        print(f"\n{'='*60}")
        print(f"Database Schema Verification")
        print(f"{'='*60}\n")

        # signals_1m のカラム確認
        print("signals_1m columns:")
        cursor.execute("PRAGMA table_info(signals_1m)")
        for col in cursor.fetchall():
            marker = "✓" if col[1].startswith('V3_') else " "
            print(f"  {marker} {col[1]} ({col[2]})")

        # 新規テーブル確認
        print("\nV3 tables:")
        tables = ['feature_contributions', 'v3_positions', 'trading_status']

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            exists = table in existing_tables
            marker = "✓" if exists else "✗"
            print(f"  {marker} {table}")

    finally:
        conn.close()


def rollback_migration(db_path: str):
    """マイグレーションをロールバック（V3カラムを削除）"""

    print(f"\n{'='*60}")
    print(f"Rollback Migration")
    print(f"{'='*60}")
    print(f"⚠️  This will remove V3 columns and tables!")
    print(f"    Database: {db_path}")
    print(f"{'='*60}\n")

    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Rollback cancelled.")
        return

    conn = sqlite3.connect(db_path, timeout=10_000)
    cursor = conn.cursor()

    try:
        # V3カラムの削除はSQLiteでは直接できないため、
        # テーブルを再作成する必要がある
        print("\n⚠️  SQLite does not support DROP COLUMN directly.")
        print("    To rollback completely, restore from backup:")
        print(f"    cp {db_path}.backup_* {db_path}")

        # V3テーブルの削除のみ実行
        print("\nDropping V3-specific tables...")

        tables_to_drop = ['feature_contributions',
                          'v3_positions', 'trading_status']

        for table in tables_to_drop:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"  ✓ Dropped: {table}")

        conn.commit()
        print("\n✅ V3 tables removed. V3 columns in signals_1m remain.")

    except Exception as e:
        print(f"\n❌ Rollback failed: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="V3 Database Migration")
    parser.add_argument("--db", default="runtime.db", help="Database path")
    parser.add_argument("--verify", action="store_true",
                        help="Verify schema only (no changes)")
    parser.add_argument("--rollback", action="store_true",
                        help="Rollback migration (remove V3 tables)")

    args = parser.parse_args()

    if args.verify:
        verify_schema(args.db)
    elif args.rollback:
        rollback_migration(args.db)
    else:
        migrate_to_v3(args.db)


if __name__ == "__main__":
    main()

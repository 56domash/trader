# scripts/migrate_to_v3.py
"""
Toyota Trading System V3 - Database Migration
既存DBにV3用のテーブル・カラムを追加
"""

import argparse
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path


class V3Migrator:
    """V3マイグレーション"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.backup_path = None

    def run(self, dry_run: bool = False, skip_backup: bool = False):
        """マイグレーション実行"""
        print(f"\n{'='*70}")
        print(f"Toyota Trading System V3 - Database Migration")
        print(f"{'='*70}")
        print(f"Database: {self.db_path}")
        print(f"Dry Run: {dry_run}")
        print(f"{'='*70}\n")

        # バックアップ
        if not skip_backup and not dry_run:
            self._backup_database()

        # 接続
        conn = sqlite3.connect(self.db_path)

        try:
            # マイグレーション実行
            self._check_existing_schema(conn)
            self._migrate_signals_table(conn, dry_run)
            self._create_v3_positions_table(conn, dry_run)
            self._create_v3_fills_table(conn, dry_run)
            self._create_trading_status_table(conn, dry_run)

            if not dry_run:
                conn.commit()
                print("\n✅ マイグレーション完了")
                if self.backup_path:
                    print(f"📦 バックアップ: {self.backup_path}")
            else:
                print("\n✅ Dry Run完了（実際の変更なし）")

        except Exception as e:
            print(f"\n❌ エラー発生: {e}")
            if not dry_run:
                conn.rollback()
                print("ロールバックしました")
                if self.backup_path:
                    print(f"必要に応じてバックアップから復元: {self.backup_path}")
            raise

        finally:
            conn.close()

    def _backup_database(self):
        """DBバックアップ"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_path = f"{self.db_path}.backup_{timestamp}"

        print(f"📦 バックアップ作成中... {self.backup_path}")
        shutil.copy2(self.db_path, self.backup_path)
        print("✓ バックアップ完了\n")

    def _check_existing_schema(self, conn: sqlite3.Connection):
        """既存スキーマの確認"""
        print("🔍 既存スキーマの確認...")

        # テーブル一覧
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table'
            ORDER BY name
        """)

        tables = [row[0] for row in cursor.fetchall()]
        print(f"   既存テーブル: {', '.join(tables)}")

        # signals_1m のカラム確認
        if 'signals_1m' in tables:
            cursor = conn.execute("PRAGMA table_info(signals_1m)")
            columns = [row[1] for row in cursor.fetchall()]

            v3_columns = [c for c in columns if c.startswith('V3_')]
            if v3_columns:
                print(f"   ⚠️  signals_1m に既存のV3カラム: {', '.join(v3_columns)}")
            else:
                print(f"   signals_1m カラム数: {len(columns)}")

        print()

    def _migrate_signals_table(self, conn: sqlite3.Connection, dry_run: bool):
        """signals_1m にV3カラムを追加"""
        print("🔧 signals_1m テーブルのマイグレーション...")

        # 既存カラムチェック
        cursor = conn.execute("PRAGMA table_info(signals_1m)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        # 追加するカラム
        v3_columns = {
            'V3_S': 'REAL',
            'V3_S_ema': 'REAL',
            'V3_action': 'TEXT',
            'V3_can_long': 'INTEGER',
            'V3_can_short': 'INTEGER',
            'V3_contrib_rsi': 'REAL',
            'V3_contrib_macd': 'REAL',
            'V3_contrib_vwap': 'REAL',
            'V3_contrib_opening_range': 'REAL',
            'V3_contrib_atr': 'REAL',
            'V3_contrib_bollinger': 'REAL',
        }

        added = []
        skipped = []

        for col_name, col_type in v3_columns.items():
            if col_name in existing_columns:
                skipped.append(col_name)
                continue

            sql = f"ALTER TABLE signals_1m ADD COLUMN {col_name} {col_type}"

            if dry_run:
                print(f"   [DRY RUN] {sql}")
            else:
                conn.execute(sql)
                added.append(col_name)

        if added:
            print(f"   ✓ 追加したカラム: {', '.join(added)}")
        if skipped:
            print(f"   ⊘ スキップ（既存）: {', '.join(skipped)}")

        print()

    def _create_v3_positions_table(self, conn: sqlite3.Connection, dry_run: bool):
        """v3_positions テーブル作成"""
        print("🔧 v3_positions テーブルの作成...")

        # 既存チェック
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='v3_positions'
        """)

        if cursor.fetchone():
            print("   ⊘ スキップ（既存）")
            print()
            return

        sql = """
        CREATE TABLE v3_positions (
            symbol TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            size INTEGER NOT NULL,
            pnl REAL,
            status TEXT NOT NULL,
            PRIMARY KEY (symbol, entry_time)
        )
        """

        if dry_run:
            print(f"   [DRY RUN] CREATE TABLE v3_positions ...")
        else:
            conn.execute(sql)
            # インデックス作成
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_v3_positions_status 
                ON v3_positions(symbol, status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_v3_positions_entry_time 
                ON v3_positions(entry_time)
            """)
            print("   ✓ テーブル作成完了")

        print()

    def _create_v3_fills_table(self, conn: sqlite3.Connection, dry_run: bool):
        """v3_fills テーブル作成"""
        print("🔧 v3_fills テーブルの作成...")

        # 既存チェック
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='v3_fills'
        """)

        if cursor.fetchone():
            print("   ⊘ スキップ（既存）")
            print()
            return

        sql = """
        CREATE TABLE v3_fills (
            symbol TEXT NOT NULL,
            ts TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            size INTEGER NOT NULL,
            commission REAL,
            v3_signal REAL,
            PRIMARY KEY (symbol, ts)
        )
        """

        if dry_run:
            print(f"   [DRY RUN] CREATE TABLE v3_fills ...")
        else:
            conn.execute(sql)
            # インデックス作成
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_v3_fills_ts 
                ON v3_fills(ts)
            """)
            print("   ✓ テーブル作成完了")

        print()

    def _create_trading_status_table(self, conn: sqlite3.Connection, dry_run: bool):
        """trading_status テーブル作成"""
        print("🔧 trading_status テーブルの作成...")

        # 既存チェック
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='trading_status'
        """)

        if cursor.fetchone():
            print("   ⊘ スキップ（既存）")
            print()
            return

        sql = """
        CREATE TABLE trading_status (
            symbol TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """

        if dry_run:
            print(f"   [DRY RUN] CREATE TABLE trading_status ...")
        else:
            conn.execute(sql)
            print("   ✓ テーブル作成完了")

        print()


def rollback_migration(db_path: str, backup_path: str):
    """マイグレーションのロールバック"""
    print(f"\n🔄 ロールバック実行中...")
    print(f"   バックアップ: {backup_path}")
    print(f"   復元先: {db_path}")

    if not Path(backup_path).exists():
        print(f"❌ バックアップファイルが見つかりません: {backup_path}")
        return

    # 現在のDBを一時保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = f"{db_path}.failed_{timestamp}"
    shutil.move(db_path, temp_path)
    print(f"   失敗したDBを保存: {temp_path}")

    # バックアップから復元
    shutil.copy2(backup_path, db_path)
    print(f"✅ ロールバック完了")


def verify_migration(db_path: str):
    """マイグレーション結果の検証"""
    print(f"\n🔍 マイグレーション結果の検証...")

    conn = sqlite3.connect(db_path)

    try:
        # テーブル確認
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('v3_positions', 'v3_fills', 'trading_status')
            ORDER BY name
        """)

        v3_tables = [row[0] for row in cursor.fetchall()]
        print(f"   V3テーブル: {', '.join(v3_tables) if v3_tables else 'なし'}")

        # signals_1m のV3カラム確認
        cursor = conn.execute("PRAGMA table_info(signals_1m)")
        columns = [row[1] for row in cursor.fetchall()]
        v3_columns = [c for c in columns if c.startswith('V3_')]

        print(f"   signals_1m V3カラム: {len(v3_columns)}個")

        # 成功判定
        expected_tables = {'v3_positions', 'v3_fills', 'trading_status'}
        if set(v3_tables) == expected_tables and len(v3_columns) >= 8:
            print("\n✅ 検証成功: すべてのV3構造が正しく作成されています")
            return True
        else:
            print("\n⚠️  検証警告: 一部の構造が不足している可能性があります")
            return False

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="V3 Database Migration")
    parser.add_argument("--db", default="runtime.db", help="Database path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run (no actual changes)")
    parser.add_argument("--skip-backup", action="store_true",
                        help="Skip backup creation")
    parser.add_argument("--rollback", help="Rollback from backup file")
    parser.add_argument("--verify", action="store_true",
                        help="Verify migration results")

    args = parser.parse_args()

    # ロールバック
    if args.rollback:
        rollback_migration(args.db, args.rollback)
        return

    # 検証のみ
    if args.verify:
        verify_migration(args.db)
        return

    # マイグレーション実行
    migrator = V3Migrator(args.db)
    migrator.run(dry_run=args.dry_run, skip_backup=args.skip_backup)

    # 検証
    if not args.dry_run:
        verify_migration(args.db)


if __name__ == "__main__":
    main()

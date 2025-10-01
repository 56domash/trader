# migrate_db_for_v3.py
"""
signals_1m テーブルに V3用の列を追加
"""
import sqlite3


def migrate_signals_table(db_path="runtime.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("="*60)
    print("DB Migration for V3")
    print("="*60)

    # 追加する列のリスト
    v3_columns = [
        ("V3_S", "REAL"),
        ("V3_S_ema", "REAL"),
        ("V3_action", "TEXT"),
        ("V3_can_long", "INTEGER"),
        ("V3_can_short", "INTEGER"),
        # 寄与度（オプション）
        ("V3_contrib_rsi", "REAL"),
        ("V3_contrib_macd", "REAL"),
        ("V3_contrib_vwap", "REAL"),
    ]

    # 既存列を確認
    cursor.execute("PRAGMA table_info(signals_1m)")
    existing = {row[1] for row in cursor.fetchall()}

    # 列追加
    added = []
    for col_name, col_type in v3_columns:
        if col_name not in existing:
            try:
                cursor.execute(
                    f"ALTER TABLE signals_1m ADD COLUMN {col_name} {col_type}")
                added.append(col_name)
                print(f"✓ 追加: {col_name} ({col_type})")
            except sqlite3.Error as e:
                print(f"✗ エラー: {col_name} - {e}")
        else:
            print(f"- スキップ: {col_name} (既存)")

    conn.commit()

    print(f"\n✅ Migration完了: {len(added)} 列追加")

    # 確認
    cursor.execute("PRAGMA table_info(signals_1m)")
    all_cols = cursor.fetchall()
    v3_cols = [c for c in all_cols if c[1].startswith('V3_')]

    print(f"\nV3関連列: {len(v3_cols)} 個")
    for col in v3_cols:
        print(f"  - {col[1]} ({col[2]})")

    conn.close()


if __name__ == "__main__":
    migrate_signals_table()

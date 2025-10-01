# check_db_schema.py
import sqlite3

conn = sqlite3.connect("runtime.db")
cursor = conn.cursor()

print("="*60)
print("signals_1m テーブル構造")
print("="*60)

# テーブル構造取得
cursor.execute("PRAGMA table_info(signals_1m)")
columns = cursor.fetchall()

print("\n現在の列:")
for col in columns:
    print(f"  {col[1]:<20} {col[2]:<10} (PK={col[5]})")

# サンプルデータ
print("\n最新5行のサンプル:")
cursor.execute("SELECT * FROM signals_1m ORDER BY ts DESC LIMIT 5")
rows = cursor.fetchall()

if rows:
    # 列名
    col_names = [desc[0] for desc in cursor.description]
    print(f"\n列名: {col_names[:8]}...")  # 最初の8列だけ表示

    # データ
    for row in rows[:2]:
        print(f"サンプル: {row[:8]}...")
else:
    print("データなし")

conn.close()

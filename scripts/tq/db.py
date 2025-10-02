import sqlite3
import pandas as pd

DB_PATH = "./runtime.db"

def main():
    conn = sqlite3.connect(DB_PATH)

    # signals_1m の最新 10 行を確認
    df = pd.read_sql("""
        SELECT *
        FROM signals_1m
        ORDER BY ts DESC
        LIMIT 10
    """, conn)

    print("\n[signals_1m] last 10 rows:")
    print(df.head(10).to_string())

    # 特に b_pack1〜20 と s_pack1〜20 だけ抜き出し
    cols = ["ts"] + [f"b_pack{i}" for i in range(1, 21)] + [f"s_pack{i}" for i in range(1, 21)]
    df_packs = df[cols]

    print("\n[signals_1m] pack columns (last 10 rows):")
    print(df_packs.to_string())

    conn.close()

if __name__ == "__main__":
    main()

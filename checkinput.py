import sqlite3
import pandas as pd

def check_table(conn, table):
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT 5;", conn)
        print(f"\n[{table}] last 5 rows:")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"[{table}] ERROR: {e}")

if __name__ == "__main__":
    conn = sqlite3.connect("runtime.db")
    for table in ["bars_1m", "features_1m", "signals_1m"]:
        check_table(conn, table)
    conn.close()

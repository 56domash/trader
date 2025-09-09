import sqlite3

conn = sqlite3.connect("runtime.db")
cur = conn.cursor()

# テーブル一覧
print(cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall())

# 件数確認
print("bars_1m rows:", cur.execute("SELECT COUNT(*) FROM bars_1m;").fetchone()[0])
print("signals_1m rows:", cur.execute("SELECT COUNT(*) FROM signals_1m;").fetchone()[0])
print("features_1m rows:", cur.execute("SELECT COUNT(*) FROM features_1m;").fetchone()[0])

#print("features_1m rows:", cur.execute("DROP TABLE IF EXISTS features_1m;"))

import sqlite3, pandas as pd
conn = sqlite3.connect("runtime.db")
df = pd.read_sql("SELECT * FROM features_1m LIMIT 5", conn)
print(df.head())

df = pd.read_sql("SELECT * FROM fills_1m LIMIT 10;", conn)
print(df.head())

import sqlite3, pandas as pd

conn = sqlite3.connect("runtime.db")
df = pd.read_sql("SELECT * FROM fills_1m", conn, parse_dates=["ts"])

# 日別に集計
df["date"] = df["ts"].dt.date
daily = df.groupby("date").agg(
    trades=("pnl", "count"),
    pnl_total=("pnl", "sum"),
    pnl_avg=("pnl", "mean"),
    wins=("pnl", lambda x: (x > 0).sum()),
    losses=("pnl", lambda x: (x <= 0).sum())
).reset_index()

print(daily)

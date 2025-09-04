# create_features_table.py
import sqlite3

con = sqlite3.connect("runtime.db")
con.executescript("""
CREATE TABLE IF NOT EXISTS features_1m (
  symbol   TEXT NOT NULL,
  ts       TEXT NOT NULL,   -- UTC ISO8601
  pack     INTEGER NOT NULL, -- 何パック目か（例: 3）
  feat_json TEXT NOT NULL,   -- {"p3_macd_hist":..., "p3_wr10":..., ...}
  PRIMARY KEY(symbol, ts, pack)
);
""")
con.commit()
con.close()
print("features_1m ready.")

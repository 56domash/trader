# query_features.py
import argparse, sqlite3, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="runtime.db")
ap.add_argument("--symbol", required=True)
ap.add_argument("--date-utc", required=True, help="UTC日付 YYYY-MM-DD（注: 09:00-10:00 JST は 00:00-01:00 UTC）")
ap.add_argument("--pack", type=int, default=3)
args = ap.parse_args()

s = f"{args.date_utc}T00:00:00Z"
e = f"{args.date_utc}T01:00:00Z"

con = sqlite3.connect(args.db)
q = """
SELECT ts,
       json_extract(feat_json, '$.p3_macd_hist')   AS p3_macd_hist,
       json_extract(feat_json, '$.p3_wr10')        AS p3_wr10,
       json_extract(feat_json, '$.p3_vol_spike20') AS p3_vol_spike20
FROM features_1m
WHERE symbol=? AND pack=? AND ts>=? AND ts<? ORDER BY ts
"""
df = pd.read_sql_query(q, con, params=[args.symbol, args.pack, s, e], parse_dates=["ts"]).set_index("ts")
print(df.head(20))

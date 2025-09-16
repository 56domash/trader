import sqlite3
import pandas as pd

conn = sqlite3.connect("runtime.db")
df = pd.read_sql_query("""
SELECT 
    symbol, 
    ts,
    S_buy, S_sell, S,
    w_pack1, w_pack2, w_pack3, w_pack4, w_pack5,
    w_pack6, w_pack7, w_pack8, w_pack9, w_pack10,
    w_pack11, w_pack12, w_pack13, w_pack14, w_pack15,
    w_pack16, w_pack17, w_pack18, w_pack19, w_pack20
FROM signals_1m
ORDER BY ts DESC
LIMIT 10
""", conn)
print(df)

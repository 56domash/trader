# verify_v3_data.py
"""
V3シグナルが正しくDBに保存されているか検証
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect("runtime.db")

print("="*60)
print("V3データ検証")
print("="*60)

# 1. V3データの件数
query = """
SELECT COUNT(*) as total,
       COUNT(V3_S) as v3_count,
       COUNT(S) as old_count
FROM signals_1m
WHERE symbol = '7203.T'
  AND date(ts) = '2025-09-22'
"""
result = pd.read_sql(query, conn)
print("\n件数:")
print(result)

# 2. V3と既存システムの比較
query = """
SELECT 
    ts,
    S as old_S,
    V3_S,
    V3_S_ema,
    V3_action,
    V3_can_long,
    V3_can_short
FROM signals_1m
WHERE symbol = '7203.T'
  AND date(ts) = '2025-09-22'
  AND V3_S IS NOT NULL
ORDER BY ts
LIMIT 10
"""
df = pd.read_sql(query, conn, parse_dates=['ts'])

print("\n最初の10行（既存 vs V3）:")
print(df.to_string(index=False))

# 3. V3アクション統計
query = """
SELECT 
    V3_action,
    COUNT(*) as count
FROM signals_1m
WHERE symbol = '7203.T'
  AND date(ts) = '2025-09-22'
  AND V3_action IS NOT NULL
GROUP BY V3_action
"""
actions = pd.read_sql(query, conn)

print("\n\nV3アクション統計:")
print(actions.to_string(index=False))

# 4. 寄与度の確認
query = """
SELECT 
    ts,
    V3_S,
    V3_contrib_rsi,
    V3_contrib_macd,
    V3_contrib_vwap
FROM signals_1m
WHERE symbol = '7203.T'
  AND date(ts) = '2025-09-22'
  AND V3_S IS NOT NULL
ORDER BY ts DESC
LIMIT 5
"""
contrib = pd.read_sql(query, conn, parse_dates=['ts'])

print("\n\n寄与度（最新5行）:")
print(contrib.to_string(index=False))

conn.close()

print("\n✅ 検証完了")

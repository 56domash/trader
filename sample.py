# # # # # # # # import yfinance as yf, pandas as pd
# # # # # # # #
# # # # # # # # cands = ["1306.T","1308.T","1475.T","1305.T","2529.T","2559.T"]  # TOPIX系の候補
# # # # # # # #
# # # # # # # # def nrows(ticker, interval, period):
# # # # # # # #     try:
# # # # # # # #         df = yf.download(ticker, interval=interval, period=period,
# # # # # # # #                          progress=False, auto_adjust=False)
# # # # # # # #     except Exception as e:
# # # # # # # #         print(f"{ticker} {interval}/{period} ERR: {e}")
# # # # # # # #         return 0
# # # # # # # #     if df is None or getattr(df, "empty", True):
# # # # # # # #         return 0
# # # # # # # #     return len(df)
# # # # # # # #
# # # # # # # # rows = []
# # # # # # # # for t in cands:
# # # # # # # #     r1 = nrows(t, "1m", "7d")
# # # # # # # #     r5 = nrows(t, "5m", "5d")
# # # # # # # #     rows.append((t, r1, r5))
# # # # # # # #
# # # # # # # # out = pd.DataFrame(rows, columns=["ticker","rows_1m(7d)","rows_5m(5d)"])
# # # # # # # # print(out)
# # # # # # #
# # # # # # # import sqlite3
# # # # # # # c = sqlite3.connect('runtime.db')
# # # # # # #
# # # # # # # def has_table(name):
# # # # # # #     return c.execute(
# # # # # # #         "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
# # # # # # #     ).fetchone() is not None
# # # # # # #
# # # # # # # if not has_table('fx_1m'):
# # # # # # #     c.execute("""
# # # # # # #         CREATE TABLE fx_1m (
# # # # # # #           symbol TEXT NOT NULL,
# # # # # # #           ts     TEXT NOT NULL,
# # # # # # #           open REAL, high REAL, low REAL, close REAL, volume REAL,
# # # # # # #           PRIMARY KEY (symbol, ts)
# # # # # # #         )
# # # # # # #     """)
# # # # # # #     c.commit()
# # # # # # #     print("fx_1m created fresh.")
# # # # # # # else:
# # # # # # #     cols = [r[1] for r in c.execute("PRAGMA table_info(fx_1m)")]
# # # # # # #     print("fx_1m columns:", cols)
# # # # # # #     need = {"symbol","ts","open","high","low","close","volume"}
# # # # # # #     if set(cols) != need:
# # # # # # #         c.execute("ALTER TABLE fx_1m RENAME TO fx_1m_backup")
# # # # # # #         bcols = {r[1] for r in c.execute("PRAGMA table_info(fx_1m_backup)")}
# # # # # # #         c.execute("""
# # # # # # #             CREATE TABLE fx_1m (
# # # # # # #               symbol TEXT NOT NULL,
# # # # # # #               ts     TEXT NOT NULL,
# # # # # # #               open REAL, high REAL, low REAL, close REAL, volume REAL,
# # # # # # #               PRIMARY KEY (symbol, ts)
# # # # # # #             )
# # # # # # #         """)
# # # # # # #         symbol_expr = "pair" if "pair" in bcols else ("symbol" if "symbol" in bcols else "'USDJPY=X'")
# # # # # # #         ts_expr     = "ts" if "ts" in bcols else "NULL"
# # # # # # #         open_expr   = "open"  if "open"  in bcols else "NULL"
# # # # # # #         high_expr   = "high"  if "high"  in bcols else "NULL"
# # # # # # #         low_expr    = "low"   if "low"   in bcols else "NULL"
# # # # # # #         close_expr  = "close" if "close" in bcols else "NULL"
# # # # # # #         volume_expr = "volume"if "volume"in bcols else "NULL"
# # # # # # #         sql = f"""
# # # # # # #             INSERT INTO fx_1m (symbol, ts, open, high, low, close, volume)
# # # # # # #             SELECT {symbol_expr} AS symbol,
# # # # # # #                    {ts_expr}     AS ts,
# # # # # # #                    {open_expr}   AS open,
# # # # # # #                    {high_expr}   AS high,
# # # # # # #                    {low_expr}    AS low,
# # # # # # #                    {close_expr}  AS close,
# # # # # # #                    {volume_expr} AS volume
# # # # # # #             FROM fx_1m_backup
# # # # # # #         """
# # # # # # #         c.execute(sql)
# # # # # # #         c.execute("DROP TABLE fx_1m_backup")
# # # # # # #         c.commit()
# # # # # # #         print("fx_1m migrated.")
# # # # # # #     else:
# # # # # # #         print("fx_1m already OK.")
# # # # # # # c.close()
# # # # # #
# # # # # # import sqlite3, pandas as pd
# # # # # # c=sqlite3.connect('runtime.db')
# # # # # # print("1306.T bars_1m:")
# # # # # # print(pd.read_sql_query("SELECT MIN(ts) min_ts, MAX(ts) max_ts, COUNT(*) n FROM bars_1m WHERE symbol='1306.T'", c))
# # # # # # print("\nUSDJPY=X fx_1m:")
# # # # # # print(pd.read_sql_query("SELECT MIN(ts) min_ts, MAX(ts) max_ts, COUNT(*) n FROM fx_1m WHERE symbol='USDJPY=X'", c))
# # # # # import sqlite3, pandas as pd
# # # # # c=sqlite3.connect('runtime.db')
# # # # # print(pd.read_sql_query("""
# # # # #   SELECT ts, ROUND(buy1,3) b1, ROUND(buy2,3) b2, ROUND(sell1,3) s1,
# # # # #          ROUND(S,3) S
# # # # #   FROM signals_1m
# # # # #   WHERE symbol='1306.T'
# # # # #   ORDER BY ts
# # # # #   LIMIT 5
# # # # # """, c))
# # # # # print(pd.read_sql_query("""
# # # # #   SELECT ts, ROUND(S,3) S
# # # # #   FROM signals_1m
# # # # #   WHERE symbol='1306.T'
# # # # #   ORDER BY ts DESC
# # # # #   LIMIT 5
# # # # # """, c))
# # # # import sqlite3, pandas as pd
# # # # c=sqlite3.connect('runtime.db')
# # # # print(pd.read_sql_query("SELECT distinct * FROM fills_1m WHERE symbol='1306.T' ORDER BY ts", c))
# # # # print(pd.read_sql_query("""
# # # #   SELECT COUNT(*) n, ROUND(SUM(COALESCE(pnl,0)),2) total_pnl
# # # #   FROM fills_1m WHERE symbol='1306.T'
# # # # """, c))
# # # #
# # # # import sqlite3
# # # # c=sqlite3.connect('runtime.db')
# # # # # 完全一致の重複を1つ残して削除（OPENのpnlは0.0扱いで比較）
# # # # c.execute("""
# # # # DELETE FROM fills_1m
# # # # WHERE rowid NOT IN (
# # # #   SELECT MIN(rowid) FROM fills_1m
# # # #   GROUP BY symbol, ts, side, action, price, qty, COALESCE(pnl,0), position, note
# # # # )
# # # # """)
# # # # c.commit(); c.close()
# # # # print("OK: deduped fills_1m")
# # # import sqlite3
# # # c=sqlite3.connect('runtime.db')
# # # for t in ["bars_1m","signals_1m","fills_1m"]:
# # #     try: c.execute(f"DELETE FROM {t} WHERE symbol='7203.T'")
# # #     except: pass
# # # c.commit(); c.close()
# # # print("cleaned 7203.T")
# # #
# # import sqlite3, pandas as pd
# # c=sqlite3.connect('runtime.db')
# # df=pd.read_sql_query("""
# #   SELECT ts, S FROM signals_1m
# #   WHERE symbol='7203.T' AND ts>='2025-09-02T00:00:00Z' AND ts<'2025-09-02T01:00:00Z'
# #   ORDER BY ts
# # """, c, parse_dates=['ts'])
# # df['S_ema']=df['S'].ewm(span=5, adjust=False).mean()
# # print("max S:", round(df['S'].max(),3), " | max S_ema:", round(df['S_ema'].max(),3))
# # print("bars S_ema >= 0.65:", int((df['S_ema']>=0.65).sum()))
#
# import sqlite3, pandas as pd
# c=sqlite3.connect('runtime.db')
# df = pd.read_sql_query("""
#   SELECT ts, S FROM signals_1m
#   WHERE symbol='7203.T'
#     AND ts>='2025-09-02T00:00:00' AND ts<'2025-09-02T01:00:00'
#   ORDER BY ts
# """, c, parse_dates=['ts'])
# if df.empty:
#     print("no signals in window"); raise SystemExit
# ema_span = 5  # ← configのema_spanと合わせて
# ema = df['S'].ewm(span=ema_span, adjust=False).mean()
# print("max S     :", round(df['S'].max(),3))
# print("max S_ema :", round(ema.max(),3))
# thr_long, thr_short = 0.60, -0.60  # ← configのthr_*と合わせて
# print("bars S_ema >= thr_long :", int((ema >= thr_long).sum()))
# print("bars S_ema <= thr_short:", int((ema <= thr_short).sum()))







import sqlite3, pandas as pd
c=sqlite3.connect('runtime.db')
print(pd.read_sql_query("""
  SELECT COUNT(*) n
  FROM fills_1m
  WHERE symbol='7203.T'
    AND ts>='2025-08-28T00:00:00' AND ts<'2025-09-11T00:00:00'
""", c))
print(pd.read_sql_query("""
  SELECT * FROM fills_1m
  WHERE symbol='7203.T'
  ORDER BY ts
  LIMIT 5
""", c))






# report.py
import argparse, sqlite3, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="runtime.db")
    ap.add_argument("--symbol", default="7203.T")
    ap.add_argument("--start", required=True, help="UTC日付(YYYY-MM-DD) 例: 2025-09-01")
    ap.add_argument("--end",   required=True, help="UTC日付(YYYY-MM-DD) ※この日を含まない上限")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    q = """
    WITH per_day AS (
      SELECT
        date(ts) AS utc_date,
        COUNT(*) FILTER (WHERE action='CLOSE') AS trades,
        ROUND(SUM(CASE WHEN action='CLOSE' THEN pnl ELSE 0 END), 2) AS pnl
      FROM fills_1m
      WHERE symbol = ?
        AND ts >= ? || 'T00:00:00Z'
        AND ts <  ? || 'T00:00:00Z'
      GROUP BY 1
    )
    SELECT utc_date AS date_utc, trades, pnl,
           ROUND(SUM(pnl) OVER(ORDER BY utc_date), 2) AS cum_pnl
    FROM per_day
    ORDER BY utc_date;
    """
    df = pd.read_sql_query(q, conn, params=(args.symbol, args.start, args.end))
    conn.close()
    if df.empty:
        print("no trades in range.")
    else:
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()

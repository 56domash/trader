# inspect_signals.py
import argparse, sqlite3, pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")

def jst_window_utc(date_str: str, start="09:00", end="10:00"):
    y,m,d = map(int, date_str.split("-"))
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    s = datetime(y,m,d,sh,sm,tzinfo=JST).astimezone(timezone.utc)
    e = datetime(y,m,d,eh,em,tzinfo=JST).astimezone(timezone.utc)
    return s.isoformat().replace("+00:00","Z"), e.isoformat().replace("+00:00","Z")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="runtime.db")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (JST)")
    ap.add_argument("--ema-span", type=int, default=3)
    ap.add_argument("--thr-long", type=float, default=0.10)
    ap.add_argument("--thr-short", type=float, default=-0.50)
    ap.add_argument("--show", choices=["head","tail","all","none"], default="head",
                    help="print which rows")
    args = ap.parse_args()

    s_utc, e_utc = jst_window_utc(args.date, "09:00", "10:00")
    con = sqlite3.connect(args.db)
    q = """
      SELECT ts, S
      FROM signals_1m
      WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts
    """
    df = pd.read_sql_query(q, con, params=[args.symbol, s_utc, e_utc], parse_dates=["ts"]).set_index("ts")
    if df.empty:
        print("no signals in window.")
        return

    # 数値化（'None'文字列対策）
    df["S"] = pd.to_numeric(df["S"], errors="coerce")
    df["S_ema"] = df["S"].ewm(span=args.ema_span, adjust=False).mean()

    # 最初に数値が出た時刻
    first_valid = df["S"].first_valid_index()
    last_valid  = df["S"].last_valid_index()

    print(f"rows in window: {len(df)}")
    print(f"valid S rows  : {int(df['S'].notna().sum())}")
    if first_valid is not None:
        print(f"first valid @ JST: {first_valid.tz_convert(JST) if first_valid.tzinfo else first_valid}")
        print(f"last  valid @ JST: {last_valid.tz_convert(JST) if last_valid.tzinfo else last_valid}")
    else:
        print("all S are NaN in this window.")

    # 統計
    print()
    print("stats:",
          f"max S={df['S'].max() if df['S'].notna().any() else float('nan'):.3f}",
          f"min S={df['S'].min() if df['S'].notna().any() else float('nan'):.3f}",
          f"max S_ema={df['S_ema'].max() if df['S_ema'].notna().any() else float('nan'):.3f}",
          f"min S_ema={df['S_ema'].min() if df['S_ema'].notna().any() else float('nan'):.3f}")

    nL = int((df["S_ema"] >= args.thr_long).sum())
    nS = int((df["S_ema"] <= args.thr_short).sum())
    print(f"bars S_ema >= {args.thr_long}: {nL}")
    print(f"bars S_ema <= {args.thr_short}: {nS}")

    # 表示
    print()
    if args.show == "head":
        print(df.head(12))
    elif args.show == "tail":
        print(df.tail(12))
    elif args.show == "all":
        print(df)
    else:
        pass

if __name__ == "__main__":
    main()

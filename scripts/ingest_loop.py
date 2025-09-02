from __future__ import annotations
import argparse
import pandas as pd
from tq.config import load
from tq.db import connect, upsert_bar, upsert_fx, upsert_market
from tq.ingest import fetch_price_1m, fetch_fx_1m, fetch_market_1m
from tq.scheduler import sleep_to_next_minute

def _as_float(x):
    if isinstance(x, pd.Series):
        x = x.iloc[0]
    try:
        return float(x)
    except Exception:
        return float(pd.to_numeric(x, errors="coerce"))

def _to_iso_utc(x) -> str:
    if isinstance(x, pd.Series):
        x = x.iloc[0]
    t = pd.to_datetime(x, utc=False, errors="coerce")
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%S%z")

def upsert_bars_df(conn, symbol: str, df_price: pd.DataFrame):
    for _, r in df_price.iterrows():
        ts = _to_iso_utc(r["datetime"])
        upsert_bar(conn, symbol, ts,
                   _as_float(r["open"]), _as_float(r["high"]),
                   _as_float(r["low"]),  _as_float(r["close"]),
                   _as_float(r["volume"]))

def upsert_fx_df(conn, df_fx: pd.DataFrame):
    if df_fx is None or df_fx.empty: 
        return
    # 候補名から最初に見つかった列を使用（USDJPY優先）
    candidates_priority = ["USDJPY", "USDJPY=X", "JPY", "JPY=X", "EURJPY", "EURJPY=X", "GBPJPY", "GBPJPY=X"]
    fx_col = next((c for c in candidates_priority if c in df_fx.columns), None)
    if fx_col is None:
        print("[INGEST][INFO] FX column not found. columns=", list(df_fx.columns)); 
        return
    sub = df_fx[["datetime", fx_col]].dropna(subset=[fx_col])
    for _, r in sub.iterrows():
        ts = _to_iso_utc(r["datetime"])
        upsert_fx(conn, "USDJPY", ts, _as_float(r[fx_col]))  # DB上はUSDJPYとして格納

def upsert_market_df(conn, df_mkt: pd.DataFrame):
    if df_mkt is None or df_mkt.empty: 
        return
    cols = [c for c in ["NK225_FUT", "SECTOR_AUTO"] if c in df_mkt.columns]
    for c in cols:
        sub = df_mkt[["datetime", c]].dropna(subset=[c])
        for _, r in sub.iterrows():
            ts = _to_iso_utc(r["datetime"])
            upsert_market(conn, c, ts, _as_float(r[c]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()
    s = load(args.config)
    conn = connect(s.db_path)
    print("[INGEST] start ->", s.db_path, "period=", s.ingest_period)
    while True:
        try:
            price = fetch_price_1m(s.symbol, period=s.ingest_period)
            upsert_bars_df(conn, s.symbol, price)

            fx = fetch_fx_1m(["USDJPY=X"], period=s.ingest_period)
            if isinstance(fx, pd.DataFrame) and not fx.empty:
                upsert_fx_df(conn, fx)

            mkt = fetch_market_1m(period=s.ingest_period)
            if isinstance(mkt, pd.DataFrame) and not mkt.empty:
                upsert_market_df(conn, mkt)

            last_ts = _to_iso_utc(price["datetime"].iloc[-1])
            print("[INGEST] up to", last_ts)
        except Exception as e:
            print("[INGEST][WARN]", e)
        sleep_to_next_minute(offset_sec=2)

if __name__ == "__main__":
    main()
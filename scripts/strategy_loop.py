# === FILE: scripts/strategy_loop.py ===
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo
import pandas as pd, yaml

from tq.utils import jst_window_utc, ensure_utc_index, vwap_day
from tq.features import compute_all

JST = ZoneInfo("Asia/Tokyo")

@dataclass
class Config:
    db_path: str = "./runtime.db"
    symbol: str = "7203.T"
    jst_start: str = "09:00"
    jst_end: str = "10:00"
    usd_jpy: str = "USDJPY=X"

def load_config(path: str) -> Config:
    try:
        raw = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
    except FileNotFoundError:
        raw = {}
    c = Config()
    c.db_path = raw.get("db_path", c.db_path)
    c.symbol = raw.get("symbol", raw.get("trading", {}).get("symbol", c.symbol))
    c.jst_start = raw.get("jst_start", c.jst_start)
    c.jst_end   = raw.get("jst_end", c.jst_end)
    c.usd_jpy   = raw.get("usd_jpy", raw.get("market", {}).get("usd_jpy", c.usd_jpy))
    return c

CREATE_SIGNALS_SQL = """
CREATE TABLE IF NOT EXISTS signals_1m (
  symbol TEXT NOT NULL, ts TEXT NOT NULL,
  buy1 REAL, buy2 REAL, buy3 REAL, buy4 REAL, buy5 REAL,
  sell1 REAL, sell2 REAL, sell3 REAL, sell4 REAL, sell5 REAL,
  S REAL, meta_json TEXT,
  PRIMARY KEY(symbol, ts)
);
"""

def _load_bars(conn, symbol, s, e):
    df = pd.read_sql_query("""
        SELECT ts, open, high, low, close, volume
        FROM bars_1m WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts
    """, conn, params=(symbol, s.isoformat(), e.isoformat()), parse_dates=["ts"])
    if df.empty: return df
    df = df.set_index("ts"); df = ensure_utc_index(df)
    df["vwap"] = vwap_day(df); return df

def _load_fx(conn, symbol, s, e):
    if not symbol: return pd.DataFrame()
    df = pd.read_sql_query("""
        SELECT ts, close, open, high, low, volume
        FROM fx_1m WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts
    """, conn, params=(symbol, s.isoformat(), e.isoformat()), parse_dates=["ts"])
    if df.empty: return df
    df = df.set_index("ts"); df = ensure_utc_index(df); return df

def systems_10(bars: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    buy1 = feat[["ret1","ret3","mom_slope3"]].mean(axis=1)
    buy2 = feat[["vwap_gap","price_above_vwap","ma_fast_slow"]].mean(axis=1)
    buy3 = feat[["bb_pos","kelt_pos","stoch_k"]].mean(axis=1)
    buy4 = feat[["vol_z60","vol_ratio_5_20","obv_slope"]].mean(axis=1)
    buy5 = feat[["ret10","ret20","rsi14"]].mean(axis=1)

    sell1 = 1 - feat["ret1"]
    sell2 = 1 - feat[["vwap_gap","price_above_vwap"]].mean(axis=1)
    sell3 = 1 - feat[["bb_pos","stoch_k","stoch_d"]].mean(axis=1)
    sell4 = 1 - feat[["vol_z60","vol_ratio_5_20"]].mean(axis=1)
    sell5 = 1 - feat[["ret5","rsi14"]].mean(axis=1)

    df = pd.DataFrame({
        "buy1":buy1,"buy2":buy2,"buy3":buy3,"buy4":buy4,"buy5":buy5,
        "sell1":sell1,"sell2":sell2,"sell3":sell3,"sell4":sell4,"sell5":sell5,
    }, index=feat.index)
    df["S"] = df[[f"buy{i}" for i in range(1,6)]].mean(axis=1) - df[[f"sell{i}" for i in range(1,6)]].mean(axis=1)
    return df

def upsert_signals(conn, symbol, sys10, meta) -> int:
    if sys10 is None or sys10.empty: return 0
    conn.execute(CREATE_SIGNALS_SQL)
    rows = []
    for ts, r in sys10.iterrows():
        tsu = pd.Timestamp(ts)
        tsu = tsu.tz_localize("UTC") if tsu.tzinfo is None else tsu.tz_convert("UTC")
        rows.append((symbol, tsu.isoformat(),
                     float(r.get("buy1",0.5)), float(r.get("buy2",0.5)), float(r.get("buy3",0.5)), float(r.get("buy4",0.5)), float(r.get("buy5",0.5)),
                     float(r.get("sell1",0.5)), float(r.get("sell2",0.5)), float(r.get("sell3",0.5)), float(r.get("sell4",0.5)), float(r.get("sell5",0.5)),
                     float(r.get("S",0.0)), json.dumps(meta, ensure_ascii=False)))
    sql = """
    INSERT INTO signals_1m (symbol, ts, buy1,buy2,buy3,buy4,buy5, sell1,sell2,sell3,sell4,sell5, S, meta_json)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(symbol, ts) DO UPDATE SET
      buy1=excluded.buy1, buy2=excluded.buy2, buy3=excluded.buy3, buy4=excluded.buy4, buy5=excluded.buy5,
      sell1=excluded.sell1, sell2=excluded.sell2, sell3=excluded.sell3, sell4=excluded.sell4, sell5=excluded.sell5,
      S=excluded.S, meta_json=excluded.meta_json
    """
    conn.executemany(sql, rows); conn.commit(); return len(rows)

def decide_target_date(conn, symbol, use_last, target_date_str):
    if target_date_str: return datetime.strptime(target_date_str, "%Y-%m-%d").date()
    if use_last:
        row = conn.execute("SELECT ts FROM bars_1m WHERE symbol=? ORDER BY ts DESC LIMIT 1", (symbol,)).fetchone()
        if row: return pd.to_datetime(row[0], utc=True).tz_convert(JST).date()
    return pd.Timestamp.now(JST).date()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--use-last-session", action="store_true")
    ap.add_argument("--target-date")
    args = ap.parse_args()

    cfg = load_config(args.config)
    conn = sqlite3.connect(cfg.db_path, timeout=5_000)
    try:
        tgt = decide_target_date(conn, cfg.symbol, args.use_last_session, args.target_date)
        s_utc, e_utc = jst_window_utc(tgt, tuple(map(int, cfg.jst_start.split(":"))), tuple(map(int, cfg.jst_end.split(":"))))
        print(f"[STRATEGY] window JST {cfg.jst_start}-{cfg.jst_end} -> UTC {s_utc.isoformat()} ~ {e_utc.isoformat()} | symbol={cfg.symbol}")

        bars = _load_bars(conn, cfg.symbol, s_utc, e_utc)
        if bars.empty: raise RuntimeError("no bars in window")
        fx   = _load_fx(conn, cfg.usd_jpy, s_utc, e_utc)

        feat = compute_all(bars, fx, None)
        sys10 = systems_10(bars, feat)
        meta = {"symbol": cfg.symbol, "jst_window":[cfg.jst_start, cfg.jst_end], "n_features": feat.shape[1], "feature_names": list(feat.columns)}
        n = upsert_signals(conn, cfg.symbol, sys10, meta)
        print(f"[STRATEGY] done -> {n} rows to signals_1m")
    finally:
        conn.close()

if __name__ == "__main__":
    main()

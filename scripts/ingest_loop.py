# -*- coding: utf-8 -*-
"""
ingest_loop.py (windowed)
- 指定した日付（JST）の 8:55〜10:05 だけ 1分足を取得して SQLite(runtime.db) にUPSERT
- いつでもテストできるよう --target-date / --use-last-session / --once をサポート
- 日経先物は候補 ["NIY=F","JP225USD=X"] でフォールバック
"""

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
from typing import Iterable, Tuple, Optional

import pandas as pd
import yfinance as yf
import yaml

JST = ZoneInfo("Asia/Tokyo")

# =========================
# Config
# =========================
@dataclass
class Config:
    db_path: str = "./runtime.db"
    symbol: str = "7203.T"
    usd_jpy: str = "USDJPY=X"
    nikkei_fut_candidates: list = None
    auto_sector_proxy: Optional[str] = "1622.T"  # 任意

def load_config(path: str) -> Config:
    try:
        raw = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
    except FileNotFoundError:
        raw = {}
    c = Config()
    c.db_path = raw.get("db_path", c.db_path)
    c.symbol = raw.get("symbol", raw.get("trading", {}).get("symbol", c.symbol))
    market = raw.get("market", {})
    c.usd_jpy = market.get("usd_jpy", c.usd_jpy)
    c.nikkei_fut_candidates = market.get("nikkei_fut_candidates", ["NIY=F", "JP225USD=X"])
    c.auto_sector_proxy = market.get("auto_sector_proxy", c.auto_sector_proxy)
    return c

# =========================
# Time window helpers
# =========================
def compute_window_for_date(date_jst: datetime.date) -> Tuple[datetime, datetime]:
    start_jst = datetime.combine(date_jst, time(8, 55), JST)
    end_jst   = datetime.combine(date_jst, time(10, 5), JST)
    return (start_jst.astimezone(timezone.utc),
            end_jst.astimezone(timezone.utc))

def decide_target_date(conn: sqlite3.Connection, symbol: str,
                       use_last_session: bool,
                       target_date_str: Optional[str]) -> datetime.date:
    if target_date_str:
        return datetime.strptime(target_date_str, "%Y-%m-%d").date()
    if use_last_session:
        row = conn.execute(
            "SELECT ts FROM bars_1m WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (symbol,)
        ).fetchone()
        if not row:
            # 何も無ければ今日
            return datetime.now(JST).date()
        return pd.to_datetime(row[0], utc=True).tz_convert(JST).date()
    return datetime.now(JST).date()

# =========================
# yfinance helpers
# =========================
def yf_download_1m(symbol: str, period: str = "7d") -> pd.DataFrame:
    df = yf.download(symbol, interval="1m", period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # index をUTC Datetimeに揃える
    idx = pd.to_datetime(df.index, utc=True)
    df = df.copy()
    df.index = idx
    df.rename(columns=str.lower, inplace=True)
    # volume が無い銘柄もあるので欠損を0埋め
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["open","high","low","close","volume"]]

def fetch_window(symbol: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    base = yf_download_1m(symbol)
    if base.empty:
        return base
    df = base[(base.index >= start_utc) & (base.index < end_utc)].copy()
    df["symbol"] = symbol
    return df

def fetch_with_candidates(cands: Iterable[str], start_utc: datetime, end_utc: datetime) -> Tuple[str, pd.DataFrame]:
    for sym in cands:
        df = fetch_window(sym, start_utc, end_utc)
        if not df.empty:
            return sym, df
    return "", pd.DataFrame()

# =========================
# DB helpers
# =========================
CREATE_BARS_SQL = """
CREATE TABLE IF NOT EXISTS bars_1m (
  symbol TEXT NOT NULL,
  ts     TEXT NOT NULL,   -- UTC ISO8601
  open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY (symbol, ts)
);
"""
CREATE_FX_SQL = """
CREATE TABLE IF NOT EXISTS fx_1m (
  symbol TEXT NOT NULL,
  ts     TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY (symbol, ts)
);
"""
CREATE_MKT_SQL = """
CREATE TABLE IF NOT EXISTS market_1m (
  source TEXT NOT NULL,   -- 取得に使ったティッカー（例: NIY=F / JP225USD=X）
  ts     TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY (source, ts)
);
"""

def to_utc_iso(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.isoformat()

def upsert_bars(conn: sqlite3.Connection, table: str, keyname: str, df: pd.DataFrame):
    if df.empty:
        return 0
    if table == "bars_1m":
        conn.execute(CREATE_BARS_SQL)
    elif table == "fx_1m":
        conn.execute(CREATE_FX_SQL)
    elif table == "market_1m":
        conn.execute(CREATE_MKT_SQL)
    rows = []
    for ts, r in df.iterrows():
        rows.append((
            r[keyname],
            to_utc_iso(ts),
            float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])
        ))
    if table in ("bars_1m", "fx_1m"):
        sql = f"""
        INSERT INTO {table}
          (symbol, ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, ts) DO UPDATE SET
          open=excluded.open, high=excluded.high, low=excluded.low,
          close=excluded.close, volume=excluded.volume
        """
    else:
        sql = f"""
        INSERT INTO {table}
          (source, ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, ts) DO UPDATE SET
          open=excluded.open, high=excluded.high, low=excluded.low,
          close=excluded.close, volume=excluded.volume
        """
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--use-last-session", action="store_true",
                    help="bars_1m の最新営業日をターゲットにする")
    ap.add_argument("--target-date", help="YYYY-MM-DD（JST）を指定してその日の 08:55-10:05 を取得")
    ap.add_argument("--once", action="store_true",
                    help="1回だけ取得して終了（ループしない）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    conn = sqlite3.connect(cfg.db_path, timeout=5_000)

    try:
        target_date = decide_target_date(conn, cfg.symbol, args.use_last_session, args.target_date)
        start_utc, end_utc = compute_window_for_date(target_date)
        print(f"[INGEST] window JST {target_date} 08:55–10:05 -> "
              f"UTC {start_utc.isoformat()} ~ {end_utc.isoformat()} | db={cfg.db_path}")

        # 1) メイン銘柄（bars_1m）
        d_stock = fetch_window(cfg.symbol, start_utc, end_utc)
        n1 = upsert_bars(conn, "bars_1m", "symbol", d_stock)
        print(f"[INGEST][bars_1m] {cfg.symbol}: +{n1} rows")

        # 2) 為替（fx_1m）
        d_fx = fetch_window(cfg.usd_jpy, start_utc, end_utc)
        n2 = upsert_bars(conn, "fx_1m", "symbol", d_fx)
        print(f"[INGEST][fx_1m] {cfg.usd_jpy}: +{n2} rows")

        # 3) 先物/CFD（market_1m）フォールバック
        src, d_mkt = fetch_with_candidates(cfg.nikkei_fut_candidates, start_utc, end_utc)
        if d_mkt.empty:
            print(f"[INGEST][market_1m][WARN] fut/CFD not available: {cfg.nikkei_fut_candidates}")
        else:
            d_mkt = d_mkt.rename(columns={"symbol":"source"})
            n3 = upsert_bars(conn, "market_1m", "source", d_mkt)
            print(f"[INGEST][market_1m] {src}: +{n3} rows")

        # 4) セクタ代理（任意）
        if cfg.auto_sector_proxy:
            d_sec = fetch_window(cfg.auto_sector_proxy, start_utc, end_utc)
            if d_sec.empty:
                print(f"[INGEST][market_1m][INFO] sector proxy empty: {cfg.auto_sector_proxy}")
            else:
                d_sec = d_sec.rename(columns={"symbol":"source"})
                n4 = upsert_bars(conn, "market_1m", "source", d_sec)
                print(f"[INGEST][market_1m] {cfg.auto_sector_proxy}: +{n4} rows")

    finally:
        conn.close()

    if not args.once:
        # ループ版が必要ならここにバックオフ付きループを実装
        # 今回は “指定日時だけ取る” リクエストなので一回で終了
        pass

if __name__ == "__main__":
    main()

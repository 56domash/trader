# -*- coding: utf-8 -*-
"""
ingest_loop.py (simple & robust - 1m/7d only)
- 取得は常に yfinance の interval=1m, period=7d のみ
- その後に JST 08:55〜10:05 の窓を切り出し（JST基準→UTC保存）
- --target-date / --use-last-session / --once をサポート
- DB: runtime.db に UTC ISO8601 で UPSERT
"""

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

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
    usd_jpy: str = "USDJPY=X"  # 任意の外部系列（やめたい場合は空文字に）

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
    return c

# =========================
# Time Window
# =========================
def compute_window_for_date(date_jst) -> Tuple[datetime, datetime]:
    start_jst = datetime.combine(date_jst, time(8, 55), JST)
    end_jst   = datetime.combine(date_jst, time(10, 5), JST)
    return (start_jst, end_jst)  # JSTのまま返す（後でUTCに変換）

def decide_target_date(conn: sqlite3.Connection, symbol: str,
                       use_last_session: bool,
                       target_date_str: Optional[str]):
    if target_date_str:
        return datetime.strptime(target_date_str, "%Y-%m-%d").date()
    if use_last_session:
        row = conn.execute(
            "SELECT ts FROM bars_1m WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (symbol,)
        ).fetchone()
        if not row:
            return datetime.now(JST).date()
        return pd.to_datetime(row[0], utc=True).tz_convert(JST).date()
    return datetime.now(JST).date()

# =========================
# yfinance helpers
# =========================
def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """yfinanceの返却を OHLCV(+UTC index) に正規化（MultiIndexの両パターンに対応）"""
    if df is None or df.empty:
        return pd.DataFrame()

    # --- MultiIndex 列対応 ---
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = [str(x).lower() for x in df.columns.get_level_values(0)]
        lv1 = [str(x).lower() for x in df.columns.get_level_values(1)]
        ohlc_set = {"open","high","low","close","adj close","volume"}

        has_ohlc_lv0 = any(x in ohlc_set for x in lv0)
        has_ohlc_lv1 = any(x in ohlc_set for x in lv1)

        if has_ohlc_lv0 and not has_ohlc_lv1:
            # 例: ('Price','Adj Close') / ('Ticker','1306.T') → level0がフィールド
            df = df.droplevel(1, axis=1)
        elif has_ohlc_lv1 and not has_ohlc_lv0:
            # 逆パターン
            df = df.droplevel(0, axis=1)
        elif symbol in df.columns.get_level_values(0):
            df = df.xs(symbol, axis=1, level=0)
        elif symbol in df.columns.get_level_values(1):
            df = df.xs(symbol, axis=1, level=1)
        else:
            # フォールバック：片方のレベルが全て同一ティッカーならそれを落とす
            if len(set(df.columns.get_level_values(1))) == 1:
                df = df.droplevel(1, axis=1)
            else:
                df = df.droplevel(0, axis=1)

    # --- 列名正規化 ---
    cols = {str(c).lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            key = n.lower()
            if key in cols:
                return df[cols[key]]
        return None

    o = pick("open")
    h = pick("high")
    l = pick("low")
    c = pick("close", "adj close")
    v = pick("volume")

    out = pd.DataFrame()
    if o is not None: out["open"] = o
    if h is not None: out["high"] = h
    if l is not None: out["low"]  = l
    if c is not None: out["close"]= c
    out["volume"] = pd.to_numeric(v, errors="coerce") if v is not None else 0

    if out.empty or "close" not in out.columns:
        return pd.DataFrame()

    # --- index を UTC へ統一（.T はナイーブJSTのことがある）---
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is None:
        if symbol.endswith(".T"):
            idx = idx.tz_localize(ZoneInfo("Asia/Tokyo")).tz_convert(timezone.utc)
        else:
            idx = idx.tz_localize(timezone.utc)
    else:
        idx = idx.tz_convert(timezone.utc)
    out.index = idx

    # 数値化
    for col in ["open","high","low","close","volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out[["open","high","low","close","volume"]]

def yf_download_7d_1m(symbol: str) -> pd.DataFrame:
    """常に 1m / 7d で取得 → 正規化（UTC index）"""
    try:
        raw = yf.download(symbol, interval="1m", period="7d",
                          progress=False, auto_adjust=False, group_by="column")
        print(raw)
    except Exception as e:
        print(f"[INGEST][WARN] yfinance failed ({symbol}): {e}")
        return pd.DataFrame()
    return _normalize_ohlcv(raw, symbol)

def slice_window_jst(df_utc: pd.DataFrame, start_jst: datetime, end_jst: datetime) -> pd.DataFrame:
    """UTC indexのDFをJSTに変換してから窓切り→UTCへ戻す"""
    if df_utc.empty:
        return df_utc
    jst = df_utc.copy()
    jst.index = jst.index.tz_convert(JST)
    jwin = jst[(jst.index >= start_jst) & (jst.index < end_jst)]
    if jwin.empty:
        return jwin
    out = jwin.copy()
    out.index = out.index.tz_convert(timezone.utc)
    return out

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



def upsert_bars(conn: sqlite3.Connection, table: str, keyname: str, df: pd.DataFrame):
    if df.empty:
        print(f"[INGEST][INFO] skip empty df for {table}")
        return 0
    if table == "bars_1m":
        conn.execute(CREATE_BARS_SQL)
    elif table == "fx_1m":
        conn.execute(CREATE_FX_SQL)

    rows = []
    for ts, r in df.iterrows():
        rows.append((
            str(df.iloc[0].get(keyname, r.get(keyname, ""))) if keyname in df.columns else r.get(keyname, ""),
            pd.Timestamp(ts).tz_convert(timezone.utc).isoformat(),
            float(r["open"]) if pd.notna(r["open"]) else None,
            float(r["high"]) if pd.notna(r["high"]) else None,
            float(r["low"])  if pd.notna(r["low"])  else None,
            float(r["close"])if pd.notna(r["close"])else None,
            float(r["volume"]) if pd.notna(r["volume"]) else None,
        ))
    if table == "bars_1m":
        sql = """
        INSERT INTO bars_1m (symbol, ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, ts) DO UPDATE SET
          open=excluded.open, high=excluded.high, low=excluded.low,
          close=excluded.close, volume=excluded.volume
        """
    else:
        sql = """
        INSERT INTO fx_1m (symbol, ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, ts) DO UPDATE SET
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
    ap.add_argument("--use-last-session", action="store_true")
    ap.add_argument("--target-date", help="YYYY-MM-DD（JST）")
    ap.add_argument("--once", action="store_true", help="1回だけ取得して終了")
    args = ap.parse_args()

    cfg = load_config(args.config)
    conn = sqlite3.connect(cfg.db_path, timeout=5_000)

    try:
        target_date = decide_target_date(conn, cfg.symbol, args.use_last_session, args.target_date)
        start_jst, end_jst = compute_window_for_date(target_date)
        start_utc = start_jst.astimezone(timezone.utc)
        end_utc   = end_jst.astimezone(timezone.utc)
        print(f"[INGEST] window JST {target_date} 08:55–10:05 -> UTC {start_utc.isoformat()} ~ {end_utc.isoformat()} | db={cfg.db_path}")

        # ---- メイン銘柄（1m/7d → JST窓切り）----
        base = yf_download_7d_1m(cfg.symbol)
        print(base)
        if not base.empty:
            win = slice_window_jst(base, start_jst, end_jst)
            if not win.empty:
                win = win.copy(); win["symbol"] = cfg.symbol
        else:
            win = pd.DataFrame()
        n1 = upsert_bars(conn, "bars_1m", "symbol", win)
        print(f"[INGEST][bars_1m] {cfg.symbol}: +{n1} rows")

        # ---- 外部市場（SP500, GOLD, OIL, VIX, BOND10Y）----
        from tq.ingest import fetch_external_1m
        ext = fetch_external_1m(period="7d")
        if not ext.empty:
            win_ext = slice_window_jst(ext, start_jst, end_jst)
            if not win_ext.empty:
                rows = []
                for ts, r in win_ext.iterrows():
                    for col in ["SP500","GOLD","OIL","VIX","BOND10Y"]:
                        if col in r and pd.notna(r[col]):
                            rows.append((col, pd.Timestamp(ts).tz_convert(timezone.utc).isoformat(), float(r[col])))
                if rows:
                    conn.execute("""
                    CREATE TABLE IF NOT EXISTS ext_1m (
                      symbol TEXT NOT NULL,
                      ts     TEXT NOT NULL,
                      close  REAL,
                      PRIMARY KEY(symbol, ts)
                    )
                    """)
                    conn.executemany(
                        "INSERT INTO ext_1m(symbol, ts, close) VALUES(?,?,?) "
                        "ON CONFLICT(symbol, ts) DO UPDATE SET close=excluded.close",
                        rows
                    )
                    conn.commit()
                    print(f"[INGEST][ext_1m] +{len(rows)} rows")


        # ---- 為替（任意。使わないなら config で消す）----
        if cfg.usd_jpy:
            fx = yf_download_7d_1m(cfg.usd_jpy)
            print(fx)
            if not fx.empty:
                fxw = slice_window_jst(fx, start_jst, end_jst)
                if not fxw.empty:
                    fxw = fxw.copy(); fxw["symbol"] = cfg.usd_jpy
            else:
                fxw = pd.DataFrame()
            n2 = upsert_bars(conn, "fx_1m", "symbol", fxw)
            print(f"[INGEST][fx_1m] {cfg.usd_jpy}: +{n2} rows")

    finally:
        conn.close()

    if args.once:
        print("[INGEST] once done.")
        return

if __name__ == "__main__":
    main()

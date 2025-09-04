#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sqlite3
from datetime import datetime, time, timedelta, timezone
from typing import Tuple, Optional

import numpy as np
import pandas as pd


JST = timezone(timedelta(hours=9))


# ------------------------------
# 共通ユーティリティ
# ------------------------------
def jst_window_utc(date_str: str,
                   start_hm: Tuple[int, int] = (9, 0),
                   end_hm: Tuple[int, int] = (10, 0)) -> Tuple[datetime, datetime]:
    """JSTの日付と[HH:MM)で与えられた窓をUTCに変換して返す"""
    y, m, d = map(int, date_str.split("-"))
    s_jst = datetime(y, m, d, start_hm[0], start_hm[1], tzinfo=JST)
    e_jst = datetime(y, m, d, end_hm[0], end_hm[1], tzinfo=JST)
    return s_jst.astimezone(timezone.utc), e_jst.astimezone(timezone.utc)


def read_sql_df(conn, sql: str, params: tuple = ()) -> pd.DataFrame:
    df = pd.read_sql_query(sql, conn, params=params, parse_dates=["ts"])
    if "ts" in df.columns:
        df = df.set_index("ts").sort_index()
    return df


# ------------------------------
# 1) テーブル列の保証（signals_1m Pack2/3 / features_1m Pack3）
# ------------------------------
def ensure_signals_pack3(conn: sqlite3.Connection):
    # signals_1m に b1..b5, s1..s5, S を（無ければ）追加
    # features_1m に p3_*（Pack3列）を（無ければ）追加
    cur = conn.cursor()

    # signals_1m
    cur.execute("PRAGMA table_info(signals_1m)")
    cols = {r[1] for r in cur.fetchall()}
    add_cols = []
    for c in ["b1","b2","b3","b4","b5","s1","s2","s3","s4","s5","S"]:
        if c not in cols:
            add_cols.append(c)
    for c in add_cols:
        cur.execute(f"ALTER TABLE signals_1m ADD COLUMN {c} REAL")

    # features_1m
    cur.execute("PRAGMA table_info(features_1m)")
    fcols = {r[1] for r in cur.fetchall()}
    p3_cols = [
        "p3_macd_hist","p3_macd_hist01",
        "p3_wr10","p3_wr10_01",
        "p3_vol_spike20","p3_vol_spike20_01",
        "p3_vixj","p3_vixj_01",
    ]
    add_f = [c for c in p3_cols if c not in fcols]
    for c in add_f:
        cur.execute(f"ALTER TABLE features_1m ADD COLUMN {c} REAL")

    conn.commit()
    print(f"signals_1m: added {add_cols or 'nothing'}")
    print(f"features_1m: added {add_f or 'nothing'}")


# ------------------------------
# 2) Pack3 の存在チェック
# ------------------------------
def check_pack3(conn: sqlite3.Connection, symbol: str, date_jst: str):
    s_utc, e_utc = jst_window_utc(date_jst)
    # 列一覧
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(features_1m)")
    fcols = [r[1] for r in cur.fetchall()]
    p3_cols = [c for c in fcols if c.startswith("p3_")]
    print(f"features_1m の列数: {len(fcols)} / Pack3想定(p3_*)の列数: {len(p3_cols)}")

    # 非NULL件数
    q = """
      SELECT ts, *
      FROM features_1m
      WHERE symbol=? AND ts>=? AND ts<? 
      ORDER BY ts
    """
    df = read_sql_df(conn, q, (symbol, s_utc.isoformat(), e_utc.isoformat()))
    counts = []
    for c in p3_cols:
        if c in df.columns:
            cnt = int(df[c].notna().sum())
            counts.append((c, cnt))
    counts.sort(key=lambda x: x[1], reverse=True)
    total_non_null = sum(cnt for _, cnt in counts)
    print(f"JST {date_jst} 09:00-10:00 窓での p3_* 非NULL合計: {total_non_null}")
    print("p3_* 非NULL件数 上位5列:", counts[:5])


# ------------------------------
# 3) S/S_ema の表示
# ------------------------------
def show_signals(conn: sqlite3.Connection, symbol: str, date_jst: str, ema_span: int):
    s_utc, e_utc = jst_window_utc(date_jst)
    q = """
      SELECT ts, S
      FROM signals_1m
      WHERE symbol=? AND ts>=? AND ts<? 
      ORDER BY ts
    """
    df = read_sql_df(conn, q, (symbol, s_utc.isoformat(), e_utc.isoformat()))

    print(f"rows in window: {len(df)}")
    valid = df["S"].notna().sum() if "S" in df.columns else 0
    print(f"valid S rows  : {valid}")
    if valid == 0:
        print("\nall S are NaN in this window.\n")
        print(df.head(12))
        return

    df["S_ema"] = df["S"].ewm(span=ema_span, adjust=False).mean()
    print(f"\nstats: max S={df['S'].max()} min S={df['S'].min()} "
          f"max S_ema={df['S_ema'].max()} min S_ema={df['S_ema'].min()}\n")
    print(df.head(10))


# ------------------------------
# 4) しきい値の自動提案（NEW）
# ------------------------------
def suggest_thresholds(conn: sqlite3.Connection, symbol: str, date_jst: str,
                       ema_span: int = 3, q_long: float = 0.75, q_short: float = 0.25,
                       exit_mult: float = 0.6):
    """
    S_ema の分位点からしきい値を提案。
    - thr_long  : S_ema の q_long 分位
    - thr_short : S_ema の q_short 分位（負側ならそのまま、正側なら -thr_long の対称も表示）
    - exit_*    : エントリーしきい値 * exit_mult （目安）
    """
    s_utc, e_utc = jst_window_utc(date_jst)
    q = """
      SELECT ts, S
      FROM signals_1m
      WHERE symbol=? AND ts>=? AND ts<? 
      ORDER BY ts
    """
    df = read_sql_df(conn, q, (symbol, s_utc.isoformat(), e_utc.isoformat()))
    if df.empty or df["S"].notna().sum() == 0:
        print("no usable S rows in window.")
        return

    df["S_ema"] = df["S"].ewm(span=ema_span, adjust=False).mean()
    s = df["S_ema"].dropna()
    if s.empty:
        print("S_ema is empty after EMA.")
        return

    thr_long = float(np.quantile(s, q_long))
    thr_short_raw = float(np.quantile(s, q_short))
    # 一応、ショートは負側を優先的に使う。正になってしまう場合は対称値も候補に表示。
    thr_short = thr_short_raw if thr_short_raw < 0 else -abs(thr_long)

    exit_long = thr_long * exit_mult
    exit_short = thr_short * exit_mult

    def r3(x): return float(np.round(x, 3))

    print("=== threshold suggestion ===")
    print(f"date       : {date_jst} (JST 09:00-10:00)")
    print(f"symbol     : {symbol}")
    print(f"ema_span   : {ema_span}")
    print(f"quantiles  : long q={q_long}, short q={q_short}, exit_mult={exit_mult}")
    print()
    print(f"stats S_ema: min={s.min():.6f}  p25={np.quantile(s,0.25):.6f}  "
          f"median={np.quantile(s,0.5):.6f}  p75={np.quantile(s,0.75):.6f}  max={s.max():.6f}")
    print()
    print(f"suggested  : thr_long={r3(thr_long)}  thr_short={r3(thr_short)}  "
          f"exit_long={r3(exit_long)}  exit_short={r3(exit_short)}")
    if thr_short_raw >= 0:
        print(f"  (note) raw q_short={thr_short_raw:.6f} ≥ 0 → 対称値 {-abs(thr_long):.3f} を暫定ショート閾値として提示")
    print()
    print("tips:")
    print("- エントリ過多で削りが出る場合は  q_long↑ / |q_short|↑ / exit_mult↓ / confirm_bars↑ / cooldown_bars↑ を検討")
    print("- 取りこぼしが多い場合は        q_long↓ / |q_short|↓ / exit_mult↑ を検討")


# ------------------------------
# 5) テーブル情報
# ------------------------------
def table_info(conn: sqlite3.Connection, table: str):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    if not rows:
        print(f"table not found: {table}")
        return
    print(f"columns in {table}:")
    for r in rows:
        print(f" - {r[1]} ({r[2]})")


def row_count(conn: sqlite3.Connection, table: str):
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT COUNT(1) FROM {table}")
        n = cur.fetchone()[0]
        print(f"{table}: {n} rows")
    except sqlite3.Error as e:
        print(f"error: {e}")


# ==============================
# main
# ==============================
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap0 = sub.add_parser("ensure-signals-pack3")
    ap0.add_argument("--db", default="runtime.db")

    ap1 = sub.add_parser("check-pack3")
    ap1.add_argument("--db", default="runtime.db")
    ap1.add_argument("--symbol", required=True)
    ap1.add_argument("--date", required=True)

    ap2 = sub.add_parser("show-signals")
    ap2.add_argument("--db", default="runtime.db")
    ap2.add_argument("--symbol", required=True)
    ap2.add_argument("--date", required=True)
    ap2.add_argument("--ema-span", type=int, default=3)

    # NEW: しきい値提案
    ap3 = sub.add_parser("suggest-thresholds")
    ap3.add_argument("--db", default="runtime.db")
    ap3.add_argument("--symbol", required=True)
    ap3.add_argument("--date", required=True)
    ap3.add_argument("--ema-span", type=int, default=3)
    ap3.add_argument("--q-long", type=float, default=0.75)
    ap3.add_argument("--q-short", type=float, default=0.25)
    ap3.add_argument("--exit-mult", type=float, default=0.6)

    ap4 = sub.add_parser("table-info")
    ap4.add_argument("--db", default="runtime.db")
    ap4.add_argument("--table", required=True)

    ap5 = sub.add_parser("row-count")
    ap5.add_argument("--db", default="runtime.db")
    ap5.add_argument("--table", required=True)

    args = ap.parse_args()
    conn = sqlite3.connect(args.db)

    if args.cmd == "ensure-signals-pack3":
        ensure_signals_pack3(conn)
    elif args.cmd == "check-pack3":
        check_pack3(conn, args.symbol, args.date)
    elif args.cmd == "show-signals":
        show_signals(conn, args.symbol, args.date, args.ema_span)
    elif args.cmd == "suggest-thresholds":
        suggest_thresholds(conn, args.symbol, args.date,
                           ema_span=args.ema_span,
                           q_long=args.q_long, q_short=args.q_short,
                           exit_mult=args.exit_mult)
    elif args.cmd == "table-info":
        table_info(conn, args.table)
    elif args.cmd == "row-count":
        row_count(conn, args.table)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

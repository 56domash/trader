# -*- coding: utf-8 -*-
from __future__ import annotations
"""
strategy_loop.py (Pack1+Pack2+Pack3, SはPack2ベース)
- DBの bars_1m から 1分足を取得（UTC保存前提）
- 特徴量は features.compute_packs() で Pack1/2/3 を一括生成
- S は Pack2 の buy*_base / sell*_base の平均差で算出（互換）
- signals_1m と features_1m に UPSERT
"""

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Optional, Tuple, List
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from tq.features_packs import *  # ← Pack1+Pack2+Pack3 を返す

JST = ZoneInfo("Asia/Tokyo")


# =========================
# Config
# =========================
@dataclass
class Config:
    db_path: str = "./runtime.db"
    symbol: str = "7203.T"
    jst_start: str = "09:00"
    jst_end: str = "10:00"
    weights: dict = None


def load_config(path: str) -> Config:
    raw = {}
    try:
        raw = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
    except FileNotFoundError:
        pass
    c = Config()
    c.db_path = raw.get("db_path", c.db_path)
    c.symbol = raw.get("symbol", c.symbol)
    c.jst_start = raw.get("jst_start", c.jst_start)
    c.jst_end = raw.get("jst_end", c.jst_end)
    raw_weights = raw.get("weights", None)
    if raw_weights:
        # pack1, pack2 ... の順で20個に変換
        weights = []
        for i in range(1, 21):
            key = f"pack{i}"
            weights.append(float(raw_weights.get(key, 1.0)))
        c.weights = weights
    return c


# =========================
# Time helpers
# =========================
def jst_day_from_arg(target_date: Optional[str], use_last_session: bool,
                     conn: sqlite3.Connection, symbol: str) -> date:
    if target_date:
        return datetime.strptime(target_date, "%Y-%m-%d").date()
    if use_last_session:
        row = conn.execute(
            "SELECT ts FROM bars_1m WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (symbol,)
        ).fetchone()
        if row:
            return pd.to_datetime(row[0], utc=True).tz_convert(JST).date()
    return datetime.now(JST).date()


def jst_window_utc(d: date, start_hm: Tuple[int, int], end_hm: Tuple[int, int]) -> Tuple[datetime, datetime]:
    s = datetime.combine(
        d, time(start_hm[0], start_hm[1]), JST).astimezone(timezone.utc)
    e = datetime.combine(
        d, time(end_hm[0], end_hm[1]), JST).astimezone(timezone.utc)
    return s, e


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(timezone.utc)
    else:
        idx = idx.tz_convert(timezone.utc)
    out = df.copy()
    out.index = idx
    return out


# =========================
# DB helpers (CREATE/UPSERT)
# =========================
CREATE_SIGNALS_SQL = """
CREATE TABLE IF NOT EXISTS signals_1m(
  symbol TEXT NOT NULL,
  ts     TEXT NOT NULL,  -- UTC ISO8601
  b1 REAL, b2 REAL, b3 REAL, b4 REAL, b5 REAL,
  s1 REAL, s2 REAL, s3 REAL, s4 REAL, s5 REAL,
  S  REAL,
  PRIMARY KEY(symbol, ts)
);
"""

CREATE_FEATURES_SQL_PREFIX = """
CREATE TABLE IF NOT EXISTS features_1m(
  symbol TEXT NOT NULL,
  ts     TEXT NOT NULL
"""
CREATE_FEATURES_SQL_SUFFIX = """,
  PRIMARY KEY(symbol, ts)
);
"""

# def upsert_signals(conn: sqlite3.Connection, symbol: str, df_sig: pd.DataFrame) -> int:
#     if df_sig.empty:
#         return 0
#     conn.execute(CREATE_SIGNALS_SQL)
#     rows = []
#     for ts, r in df_sig.iterrows():
#         rows.append((
#             symbol,
#             pd.Timestamp(ts).tz_convert(timezone.utc).isoformat(),
#             float(r.get("b1")) if pd.notna(r.get("b1")) else None,
#             float(r.get("b2")) if pd.notna(r.get("b2")) else None,
#             float(r.get("b3")) if pd.notna(r.get("b3")) else None,
#             float(r.get("b4")) if pd.notna(r.get("b4")) else None,
#             float(r.get("b5")) if pd.notna(r.get("b5")) else None,
#             float(r.get("s1")) if pd.notna(r.get("s1")) else None,
#             float(r.get("s2")) if pd.notna(r.get("s2")) else None,
#             float(r.get("s3")) if pd.notna(r.get("s3")) else None,
#             float(r.get("s4")) if pd.notna(r.get("s4")) else None,
#             float(r.get("s5")) if pd.notna(r.get("s5")) else None,
#             float(r.get("S"))  if pd.notna(r.get("S"))  else None,
#         ))
#     sql = """
#     INSERT INTO signals_1m(symbol, ts, b1,b2,b3,b4,b5, s1,s2,s3,s4,s5, S)
#     VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
#     ON CONFLICT(symbol, ts) DO UPDATE SET
#       b1=excluded.b1, b2=excluded.b2, b3=excluded.b3, b4=excluded.b4, b5=excluded.b5,
#       s1=excluded.s1, s2=excluded.s2, s3=excluded.s3, s4=excluded.s4, s5=excluded.s5,
#       S =excluded.S
#     """
#     conn.executemany(sql, rows)
#     conn.commit()
#     return len(rows)
# def upsert_signals(conn: sqlite3.Connection, symbol: str, df: pd.DataFrame):
#     """
#     signals_1m に Pack1〜20の b/s と S を保存
#     """
#     if df.empty:
#         print("[STRATEGY][INFO] skip empty signals")
#         return 0

#     # テーブルを作り直す場合は DROP TABLE してから実行
#     conn.execute("""
# CREATE TABLE IF NOT EXISTS signals_1m (
#     symbol TEXT NOT NULL,
#     ts TEXT NOT NULL,
#     S_buy REAL,
#     S_sell REAL,
#     S REAL,
#     b_pack1 REAL, s_pack1 REAL,
#     b_pack2 REAL, s_pack2 REAL,
#     b_pack3 REAL, s_pack3 REAL,
#     b_pack4 REAL, s_pack4 REAL,
#     b_pack5 REAL, s_pack5 REAL,
#     b_pack6 REAL, s_pack6 REAL,
#     b_pack7 REAL, s_pack7 REAL,
#     b_pack8 REAL, s_pack8 REAL,
#     b_pack9 REAL, s_pack9 REAL,
#     b_pack10 REAL, s_pack10 REAL,
#     b_pack11 REAL, s_pack11 REAL,
#     b_pack12 REAL, s_pack12 REAL,
#     b_pack13 REAL, s_pack13 REAL,
#     b_pack14 REAL, s_pack14 REAL,
#     b_pack15 REAL, s_pack15 REAL,
#     b_pack16 REAL, s_pack16 REAL,
#     b_pack17 REAL, s_pack17 REAL,
#     b_pack18 REAL, s_pack18 REAL,
#     b_pack19 REAL, s_pack19 REAL,
#     b_pack20 REAL, s_pack20 REAL,
#     PRIMARY KEY (symbol, ts)
#     );
#     """)

#     # DataFrameから行を作る
#     rows = []
#     for ts, r in df.iterrows():
#         row = [
#             symbol,
#             pd.Timestamp(ts).tz_convert("UTC").isoformat(),
#             float(r.get("S_buy", None)) if pd.notna(r.get("S_buy", None)) else None,
#             float(r.get("S_sell", None)) if pd.notna(r.get("S_sell", None)) else None,
#             float(r.get("S", None)) if pd.notna(r.get("S", None)) else None,
#         ]
#         # b_pack1〜20, s_pack1〜20 を順番に追加
#         for i in range(1, 21):
#             row.append(float(r.get(f"b_pack{i}", None)) if pd.notna(r.get(f"b_pack{i}", None)) else None)
#             row.append(float(r.get(f"s_pack{i}", None)) if pd.notna(r.get(f"s_pack{i}", None)) else None)
#         rows.append(row)

#     placeholders = ",".join(["?"] * (3 + 2 * 20 + 2))  # symbol, ts, S_buy, S_sell, S + 40列
#     sql = f"""
#     INSERT INTO signals_1m VALUES ({placeholders})
#     ON CONFLICT(symbol, ts) DO UPDATE SET
#       S_buy=excluded.S_buy,
#       S_sell=excluded.S_sell,
#       S=excluded.S,
#       {", ".join([f"b_pack{i}=excluded.b_pack{i}, s_pack{i}=excluded.s_pack{i}" for i in range(1,21)])}
#     """
#     conn.executemany(sql, rows)
#     conn.commit()
#     return len(rows)


def upsert_signals(conn: sqlite3.Connection, symbol: str, sig_out: pd.DataFrame):
    """
    signals_1m にシグナルを upsert する
    - S_buy, S_sell, S
    - b_pack1..20, s_pack1..20
    - w_pack1..20 (初期値=1.0)
    """
    if sig_out.empty:
        print("[STRATEGY][INFO] skip empty df for signals_1m")
        return 0

    # signals_1m テーブル作成（なければ）
    cols = []
    for i in range(1, 21):
        cols.append(f"b_pack{i} REAL")
    for i in range(1, 21):
        cols.append(f"s_pack{i} REAL")
    for i in range(1, 21):
        cols.append(f"w_pack{i} REAL")

    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS signals_1m (
        symbol TEXT,
        ts TEXT,
        S_buy REAL,
        S_sell REAL,
        S REAL,
        {", ".join(cols)},
        PRIMARY KEY(symbol, ts)
    )
    """)

    # SQL 構築
    insert_cols = ["symbol", "ts", "S_buy", "S_sell", "S"]
    for i in range(1, 21):
        insert_cols.append(f"b_pack{i}")
    for i in range(1, 21):
        insert_cols.append(f"s_pack{i}")
    for i in range(1, 21):
        insert_cols.append(f"w_pack{i}")

    placeholders = ",".join(["?"] * len(insert_cols))

    update_cols = [
        f"{c}=excluded.{c}" for c in insert_cols if c not in ("symbol", "ts")]

    sql = f"""
    INSERT INTO signals_1m ({",".join(insert_cols)})
    VALUES ({placeholders})
    ON CONFLICT(symbol, ts) DO UPDATE SET
    {",".join(update_cols)}
    """

    # rows 構築
    rows = []
    for ts, r in sig_out.iterrows():
        row = [
            symbol,
            pd.Timestamp(ts).isoformat(),
            r.get("S_buy"),
            r.get("S_sell"),
            r.get("S"),
        ]
        # b_pack
        for i in range(1, 21):
            row.append(r.get(f"b_pack{i}", None))
        # s_pack
        for i in range(1, 21):
            row.append(r.get(f"s_pack{i}", None))
        # w_pack (初期は全部1.0)
        for i in range(1, 21):
            row.append(1.0)

        rows.append(tuple(row))

    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def upsert_features(conn: sqlite3.Connection, symbol: str, feat: pd.DataFrame) -> int:
    if feat.empty:
        return 0
    feat = feat.copy()
    feat = ensure_utc_index(feat)
    # 動的に列を作成
    cols = list(feat.columns)
    # 既に存在しない場合は CREATE
    ddl = CREATE_FEATURES_SQL_PREFIX
    for col in cols:
        ddl += f",\n  \"{col}\" REAL"
    ddl += CREATE_FEATURES_SQL_SUFFIX
    conn.execute(ddl)

    # UPSERT
    placeholders = ",".join(["?"] * (2 + len(cols)))
    set_clause = ",".join([f"\"{c}\"=excluded.\"{c}\"" for c in cols])
    sql = f"""
    INSERT INTO features_1m(symbol, ts, {",".join([f'"{c}"' for c in cols])})
    VALUES({placeholders})
    ON CONFLICT(symbol, ts) DO UPDATE SET {set_clause}
    """
    rows = []
    for ts, r in feat.iterrows():
        base = [symbol, pd.Timestamp(ts).tz_convert(timezone.utc).isoformat()]
        vals = [(float(r[c]) if pd.notna(r[c]) else None) for c in cols]
        rows.append(base + vals)
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


# =========================
# Core
# =========================
def load_bars(conn: sqlite3.Connection, symbol: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    q = """
    SELECT ts, open, high, low, close, volume
    FROM bars_1m
    WHERE symbol=? AND ts>=? AND ts<?
    ORDER BY ts
    """
    df = pd.read_sql_query(
        q, conn,
        params=(symbol, start_utc.isoformat(), end_utc.isoformat()),
        parse_dates=["ts"]
    ).set_index("ts")
    return ensure_utc_index(df)


def load_ext(conn: sqlite3.Connection, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    q = """
    SELECT ts, symbol, close
    FROM ext_1m
    WHERE ts>=? AND ts<?
    """
    df = pd.read_sql_query(q, conn, params=(
        start_utc.isoformat(), end_utc.isoformat()), parse_dates=["ts"])
    if df.empty:
        return pd.DataFrame()
    df = df.pivot(index="ts", columns="symbol", values="close")
    return ensure_utc_index(df)


def build_signals_from_pack2(feat_all: pd.DataFrame) -> pd.DataFrame:
    """Pack2の *_base から 5買い/5売りの平均差で S を作る"""
    need = ["buy1_base", "buy2_base", "buy3_base", "buy4_base", "buy5_base",
            "sell1_base", "sell2_base", "sell3_base", "sell4_base", "sell5_base"]
    missing = [c for c in need if c not in feat_all.columns]
    if missing:
        raise KeyError(f"Pack2 base columns missing: {missing}")
    sig = pd.DataFrame(index=feat_all.index)
    # 0〜1 の想定（features側で01化済）
    buys = feat_all[["buy1_base", "buy2_base",
                     "buy3_base", "buy4_base", "buy5_base"]].clip(0, 1)
    sells = feat_all[["sell1_base", "sell2_base",
                      "sell3_base", "sell4_base", "sell5_base"]].clip(0, 1)
    sig["b1"], sig["b2"], sig["b3"], sig["b4"], sig["b5"] = [
        buys.iloc[:, i] for i in range(5)]
    sig["s1"], sig["s2"], sig["s3"], sig["s4"], sig["s5"] = [
        sells.iloc[:, i] for i in range(5)]
    sig["S"] = buys.mean(axis=1) - sells.mean(axis=1)  # [-1,1]想定
    return sig


def build_signals_from_all(feat_all: pd.DataFrame, weights: dict) -> pd.DataFrame:
    sig = pd.DataFrame(index=feat_all.index)

    # Pack2
    buys = feat_all[["buy1_base", "buy2_base",
                     "buy3_base", "buy4_base", "buy5_base"]].clip(0, 1)
    sells = feat_all[["sell1_base", "sell2_base",
                      "sell3_base", "sell4_base", "sell5_base"]].clip(0, 1)
    sig["b_pack2"] = buys.mean(axis=1)
    sig["s_pack2"] = sells.mean(axis=1)

    # Pack1
    sig["b_pack1"] = feat_all["p1_rsi14_01"]
    sig["s_pack1"] = 1 - feat_all["p1_rsi14_01"]

    # Pack3
    sig["b_pack3"] = feat_all["p3_macd_hist01"]
    sig["s_pack3"] = 1 - feat_all["p3_macd_hist01"]

    # Pack4
    sig["b_pack4"] = (feat_all["p4_ret5m"].clip(
        lower=-0.01, upper=0.01) + 0.01) / 0.02
    sig["s_pack4"] = 1 - sig["b_pack4"]

    # 重み付き合成
    w1, w2, w3, w4 = weights["pack1"], weights["pack2"], weights["pack3"], weights["pack4"]
    num = (w1*(sig["b_pack1"]-sig["s_pack1"])
           + w2*(sig["b_pack2"]-sig["s_pack2"])
           + w3*(sig["b_pack3"]-sig["s_pack3"])
           + w4*(sig["b_pack4"]-sig["s_pack4"]))
    den = (w1+w2+w3+w4)
    sig["S"] = (num/den).clip(-1, 1)

    return sig

# def build_signals_from_all20(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
#     """
#     Pack1〜20 の特徴量からシグナルを合成する
#     - b_packN / s_packN を作成
#     - NaNは0.5埋め
#     - weightsで加重平均（なければデフォルト=1）
#     """
#     out = pd.DataFrame(index=df.index)

#     # 個別Packごとのbuy/sellスコアを作成
#     for i in range(1, 21):
#         col = f"pack{i}"
#         if col in df.columns:
#             # NaNは0.5埋め
#             b = df[col].fillna(0.5)
#             s = 1.0 - b
#             out[f"b_{col}"] = b
#             out[f"s_{col}"] = s

#     # Packごとの重みを正規化して合成
#     bcols = [c for c in out.columns if c.startswith("b_")]
#     scols = [c for c in out.columns if c.startswith("s_")]

#     wb = []
#     for c in bcols:
#         pack = c.replace("b_", "")
#         wb.append(weights.get(pack, 1.0))  # デフォルト重み=1.0
#     ws = wb  # buyとsellは同じ重みで

#     b_vals = out[bcols].multiply(wb, axis=1).sum(axis=1) / sum(wb)
#     s_vals = out[scols].multiply(ws, axis=1).sum(axis=1) / sum(ws)

#     out["S_buy"] = b_vals
#     out["S_sell"] = s_vals
#     out["S"] = out["S_buy"] - out["S_sell"]

#     return out

# def build_signals_from_all20(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
#     out = pd.DataFrame(index=df.index)

#     b_cols = [f"b_pack{i}" for i in range(1, 21)]
#     s_cols = [f"s_pack{i}" for i in range(1, 21)]

#     # 🚩 NaNを必ず0.5で補正
#     df[b_cols] = df[b_cols].fillna(0.5)
#     df[s_cols] = df[s_cols].fillna(0.5)

#     # 加重平均
#     b_score = sum(weights.get(f"pack{i}", 1.0) * df[f"b_pack{i}"] for i in range(1, 21))
#     s_score = sum(weights.get(f"pack{i}", 1.0) * df[f"s_pack{i}"] for i in range(1, 21))
#     w_total = sum(weights.get(f"pack{i}", 1.0) for i in range(1, 21))

#     out["S_buy"] = (b_score / w_total).fillna(0.5)
#     out["S_sell"] = (s_score / w_total).fillna(0.5)
#     out["S"] = out["S_buy"] - out["S_sell"]

#     return out


def build_signals_from_all20(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    b_cols = [f"b_pack{i}" for i in range(1, 21)]
    s_cols = [f"s_pack{i}" for i in range(1, 21)]

    # 🚩 NaNを必ず0.5で補正
    df[b_cols] = df[b_cols].fillna(0.5)
    df[s_cols] = df[s_cols].fillna(0.5)

    # 加重平均
    b_score = sum(weights.get(f"pack{i}", 1.0)
                  * df[f"b_pack{i}"] for i in range(1, 21))
    s_score = sum(weights.get(f"pack{i}", 1.0)
                  * df[f"s_pack{i}"] for i in range(1, 21))
    w_total = sum(weights.get(f"pack{i}", 1.0) for i in range(1, 21))

    out["S_buy"] = (b_score / w_total).fillna(0.5)
    out["S_sell"] = (s_score / w_total).fillna(0.5)
    out["S"] = out["S_buy"] - out["S_sell"]

    # 🚩 各 Pack の b/s もそのまま保存
    for i in range(1, 21):
        out[f"b_pack{i}"] = df[f"b_pack{i}"].fillna(0.5).clip(0, 1)
        out[f"s_pack{i}"] = df[f"s_pack{i}"].fillna(0.5).clip(0, 1)

    return out


def build_signals_from_all10(feat_all: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Pack1〜10に対応した重み付き合成。
    Pack5〜Pack10はscaffoldとして0.5固定（features側で順次実装予定）。
    """
    sig = pd.DataFrame(index=feat_all.index)

    # Pack1
    if "p1_rsi14_01" in feat_all:
        sig["b_pack1"] = feat_all["p1_rsi14_01"]
        sig["s_pack1"] = 1 - feat_all["p1_rsi14_01"]
    else:
        sig["b_pack1"] = sig["s_pack1"] = 0.5

    # Pack2
    if all(c in feat_all.columns for c in ["buy1_base", "buy2_base", "buy3_base", "buy4_base", "buy5_base"]):
        buys = feat_all[["buy1_base", "buy2_base",
                         "buy3_base", "buy4_base", "buy5_base"]].clip(0, 1)
        sells = feat_all[["sell1_base", "sell2_base",
                          "sell3_base", "sell4_base", "sell5_base"]].clip(0, 1)
        sig["b_pack2"] = buys.mean(axis=1)
        sig["s_pack2"] = sells.mean(axis=1)
    else:
        sig["b_pack2"] = sig["s_pack2"] = 0.5

    # Pack3
    if "p3_macd_hist01" in feat_all:
        sig["b_pack3"] = feat_all["p3_macd_hist01"]
        sig["s_pack3"] = 1 - feat_all["p3_macd_hist01"]
    else:
        sig["b_pack3"] = sig["s_pack3"] = 0.5

    # Pack4
    if "p4_ret5m" in feat_all:
        sig["b_pack4"] = (feat_all["p4_ret5m"].clip(-0.01, 0.01) + 0.01) / 0.02
        sig["s_pack4"] = 1 - sig["b_pack4"]
    else:
        sig["b_pack4"] = sig["s_pack4"] = 0.5

    # Pack5〜Pack10: ダミー
    for i in range(5, 11):
        sig[f"b_pack{i}"] = 0.5
        sig[f"s_pack{i}"] = 0.5

    # ---- 重み付き合成 ----
    num, den = 0.0, 0.0
    for key, w in weights.items():
        if w == 0:
            continue
        b_col, s_col = f"b_{key}", f"s_{key}"
        if b_col in sig.columns and s_col in sig.columns:
            num += w * (sig[b_col] - sig[s_col])
            den += w
    sig["S"] = (num/den).clip(-1, 1) if den > 0 else 0.0

    return sig


def generate_signals(conn, cfg, target_date):
    s_h, s_m = map(int, cfg.jst_start.split(":"))
    e_h, e_m = map(int, cfg.jst_end.split(":"))
    out_s_utc, out_e_utc = jst_window_utc(target_date, (s_h, s_m), (e_h, e_m))
    warm_s_utc, warm_e_utc = jst_window_utc(target_date, (8, 55), (10, 5))

    bars = load_bars(conn, cfg.symbol, warm_s_utc, warm_e_utc)
    df_ext = load_ext(conn, warm_s_utc, warm_e_utc)
    feat_all = compute_packs(bars, df_mkt=df_ext)
    pack_scores = compute_pack_scores_from_all_features(feat_all)
    sig_out = build_signals_from_all20(pack_scores, cfg.weights)

    return sig_out.loc[out_s_utc:out_e_utc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--use-last-session", action="store_true")
    ap.add_argument("--target-date", help="YYYY-MM-DD（JST）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    conn = sqlite3.connect(cfg.db_path, timeout=10_000)

    try:
        tgt = jst_day_from_arg(
            args.target_date, args.use_last_session, conn, cfg.symbol)

        # 出力窓（JST 09:00–10:00）
        s_h, s_m = map(int, cfg.jst_start.split(":"))
        e_h, e_m = map(int, cfg.jst_end.split(":"))
        out_s_utc, out_e_utc = jst_window_utc(tgt, (s_h, s_m), (e_h, e_m))

        # 計算窓（JST 08:55–10:05）: ウォームアップ込み
        warm_s_utc, warm_e_utc = jst_window_utc(tgt, (8, 55), (10, 5))

        print(
            f"[STRATEGY] window JST {cfg.jst_start}-{cfg.jst_end} -> UTC {out_s_utc.isoformat()} ~ {out_e_utc.isoformat()} | symbol={cfg.symbol}")

        # 1) バー読み込み（ウォームアップ窓）
        bars = load_bars(conn, cfg.symbol, warm_s_utc, warm_e_utc)
        if bars.empty:
            print("[STRATEGY][WARN] no bars in warm window -> skip")
            return

        # 外部データ
        df_ext = load_ext(conn, warm_s_utc, warm_e_utc)

        # 2) 特徴量（Pack1〜4）
        # feat_all = compute_packs(bars, df_mkt=df_ext)

        # sig_all = build_signals_from_all20(feat_all, cfg.weights)

        feat_all = compute_packs(bars, df_mkt=df_ext)
        pack_scores = compute_pack_scores_from_all_features(feat_all)

        # from strategy_utils import generate_signals

        # sig_out = generate_signals(conn, cfg, target_date)
        # signals_1m に書き込み

        sig_out = build_signals_from_all20(pack_scores, cfg.weights)

        # 4) 出力窓にスライス（UTC）
        # sig_out = sig_all[(sig_all.index >= out_s_utc) & (sig_all.index < out_e_utc)]
        feat_out = feat_all.loc[sig_out.index]  # 同じIndexのみ保存
        if sig_out.empty:
            print("[STRATEGY][WARN] no rows in output window")
            return

        print("[DEBUG] sig_out sample:")
        print(sig_out.head().to_string())

        # 5) 保存
        n_sig = upsert_signals(conn, cfg.symbol, sig_out)
        n_feat = upsert_features(conn, cfg.symbol, feat_out)

        print(
            f"[STRATEGY] done -> {n_sig} rows to signals_1m, {n_feat} rows to features_1m")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

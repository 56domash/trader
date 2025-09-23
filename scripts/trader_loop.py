# -*- coding: utf-8 -*-
from __future__ import annotations
from tq.features import PACK3  # すでにALL_SPECSに含めるならこのimportは不要

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd
import yaml
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")

# =============================================================================
# Config
# =============================================================================


@dataclass
class Config:
    db_path: str = "./runtime.db"
    symbol: str = "7203.T"
    jst_start: str = "09:00"
    jst_end: str = "10:00"

    # thresholds（ここが“最終使用値”。自動上書き等はしない）
    thr_long: float = 0.10
    thr_short: float = -0.50
    exit_long: float = 0.10
    exit_short: float = -0.10

    # 判定・運用
    ema_span: int = 3
    confirm_bars: int = 1
    vwap_gate: bool = False
    min_edge: float = 0.0
    min_slope: float = 0.0
    cooldown_bars: int = 0
    max_trades_per_day: int = 50
    side_filter: str = "both"  # "long" / "short" / "both"

    # サブ時間帯（null で完全無効）
    sub_start: Optional[str] = None  # "09:03"
    sub_end: Optional[str] = None    # "09:55"

    # 価格・数量・TTL
    lot_size: int = 100
    ttl_min: int = 8
    fee_rate: float = 0.0002  # 約定ごとに price*qty*fee_rate


def _maybe_float(x, default):
    try:
        return float(x)
    except Exception:
        return default


def load_config(path: str, case: Optional[str] = None) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # case指定ありなら、そのブロックを掘り出す
    if case is not None:
        if case not in raw:
            raise ValueError(f"Case '{case}' not found in {path}")
        raw = raw[case] or {}

    c = Config()
    c.db_path = raw.get("db_path", c.db_path)
    c.symbol = raw.get("symbol", c.symbol)
    c.jst_start = raw.get("jst_start", c.jst_start)
    c.jst_end = raw.get("jst_end", c.jst_end)

    c.thr_long = _maybe_float(raw.get("thr_long", c.thr_long), c.thr_long)
    c.thr_short = _maybe_float(raw.get("thr_short", c.thr_short), c.thr_short)
    c.exit_long = _maybe_float(raw.get("exit_long", c.exit_long), c.exit_long)
    c.exit_short = _maybe_float(
        raw.get("exit_short", c.exit_short), c.exit_short)

    c.ema_span = int(raw.get("ema_span", c.ema_span))
    c.confirm_bars = int(raw.get("confirm_bars", c.confirm_bars))
    c.vwap_gate = bool(raw.get("vwap_gate", c.vwap_gate))
    c.min_edge = _maybe_float(raw.get("min_edge", c.min_edge), c.min_edge)
    c.min_slope = _maybe_float(raw.get("min_slope", c.min_slope), c.min_slope)
    c.cooldown_bars = int(raw.get("cooldown_bars", c.cooldown_bars))
    c.max_trades_per_day = int(
        raw.get("max_trades_per_day", c.max_trades_per_day))
    c.side_filter = str(raw.get("side_filter", c.side_filter)).lower()

    sub_start = raw.get("sub_start", None)
    sub_end = raw.get("sub_end", None)
    c.sub_start = sub_start if (sub_start and str(sub_start).strip()) else None
    c.sub_end = sub_end if (sub_end and str(sub_end).strip()) else None

    c.lot_size = int(raw.get("lot_size", c.lot_size))
    c.ttl_min = int(raw.get("ttl_min", c.ttl_min))
    c.fee_rate = _maybe_float(raw.get("fee_rate", c.fee_rate), c.fee_rate)

    # case名を記録（後で DB に保存したりログに残せるように）
    setattr(c, "_case_name", case or "default")

    return c


# =============================================================================
# Time helpers
# =============================================================================


def _to_jst(ts_utc: datetime) -> datetime:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    return ts_utc.astimezone(JST)


def _as_utc(ts) -> datetime:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC").to_pydatetime()
    return ts.tz_convert("UTC").to_pydatetime()


def _hmm_to_time(s: str) -> time:
    hh, mm = map(int, s.split(":"))
    return time(hh, mm)


def jst_window_utc(day_jst: date, start_hm: str, end_hm: str) -> Tuple[datetime, datetime]:
    s_jst = datetime.combine(day_jst, _hmm_to_time(start_hm), JST)
    e_jst = datetime.combine(day_jst, _hmm_to_time(end_hm), JST)
    return s_jst.astimezone(timezone.utc), e_jst.astimezone(timezone.utc)


def decide_target_date(conn: sqlite3.Connection, symbol: str,
                       use_last_session: bool,
                       target_date_str: Optional[str]) -> date:
    if target_date_str:
        return datetime.strptime(target_date_str, "%Y-%m-%d").date()
    if use_last_session:
        row = conn.execute(
            "SELECT ts FROM bars_1m WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (symbol,)
        ).fetchone()
        if row:
            return pd.to_datetime(row[0], utc=True).tz_convert(JST).date()
    return datetime.now(JST).date()

# =============================================================================
# DB helpers
# =============================================================================


def ensure_tables(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS fills_1m (
      symbol   TEXT NOT NULL,
      ts       TEXT NOT NULL,        -- UTC ISO8601
      side     TEXT NOT NULL,        -- BUY/SELL
      action   TEXT NOT NULL,        -- OPEN/CLOSE/EOD
      price    REAL NOT NULL,
      qty      INTEGER NOT NULL,
      pnl      REAL,
      position INTEGER NOT NULL,
      note     TEXT,
      PRIMARY KEY(symbol, ts, side, action, price, qty, position, note)
    );
    """)
    conn.commit()

    # 既存の fills_1m などのCREATE文があるはず
    conn.execute("""
    CREATE TABLE IF NOT EXISTS trader_state_1m (
        symbol TEXT,
        ts TEXT,
        S_ema REAL,
        can_long INTEGER,
        can_short INTEGER,
        below_run INTEGER,
        above_run INTEGER,
        pos INTEGER,
        case_name TEXT,
        PRIMARY KEY (symbol, ts, case_name)
    )
    """)
    conn.commit()


def read_signals(conn: sqlite3.Connection, symbol: str, s_utc: datetime, e_utc: datetime) -> pd.DataFrame:
    q = """
    SELECT ts, S
    FROM signals_1m
    WHERE symbol=? AND ts>=? AND ts<?
    ORDER BY ts
    """
    df = pd.read_sql_query(q, conn, params=(symbol, s_utc.isoformat(), e_utc.isoformat()),
                           parse_dates=["ts"])
    if df.empty:
        return df
    df.set_index("ts", inplace=True)
    return df


def read_prices(conn: sqlite3.Connection, symbol: str, s_utc: datetime, e_utc: datetime) -> pd.DataFrame:
    q = """
    SELECT ts, close
    FROM bars_1m
    WHERE symbol=? AND ts>=? AND ts<?
    ORDER BY ts
    """
    px = pd.read_sql_query(q, conn, params=(symbol, s_utc.isoformat(), e_utc.isoformat()),
                           parse_dates=["ts"])
    if px.empty:
        return px
    px.set_index("ts", inplace=True)
    return px


def emit(conn: sqlite3.Connection, symbol: str, ts, side, action, price, qty, pnl, position, note):
    conn.execute(
        """
        INSERT OR REPLACE INTO fills_1m(symbol, ts, side, action, price, qty, pnl, position, note)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (symbol, _as_utc(ts).isoformat(), side, action,
         float(price), int(qty), None if pnl is None else float(pnl),
         int(position), str(note))
    )

# =============================================================================
# Debug helpers
# =============================================================================


def _fmt_jst(ts_utc: datetime) -> str:
    return _to_jst(ts_utc).strftime("%H:%M")


def _dbg_bar(ts, s, prev_s, thrL, thrS, below_run, above_run,
             in_subwin, on_cooldown, vwap_ok, verbose, cfg: Config):
    if not verbose:
        return
    prev_txt = "nan" if prev_s is None else f"{prev_s:.3f}"
    cross_up = (prev_s is not None and prev_s < thrL and s >= thrL)
    cross_down = (prev_s is not None and prev_s > thrS and s <= thrS)
    can_long = (below_run >= max(1, cfg.confirm_bars)) and (
        s >= thrL + cfg.min_edge) and (prev_s is not None and (s - prev_s) >= cfg.min_slope)
    can_short = (above_run >= max(1, cfg.confirm_bars)) and (
        s <= thrS - cfg.min_edge) and (prev_s is not None and (prev_s - s) >= cfg.min_slope)
    print(
        f"[DBG {_fmt_jst(ts)}] S_ema={s:.3f} prev={prev_txt}  "
        f"below_run={below_run} above_run={above_run}  "
        f"in_subwin={in_subwin} cooldown={on_cooldown} vwap_ok={vwap_ok}  "
        f"thrL={thrL:.2f} thrS={thrS:.2f} edge={cfg.min_edge:.3f} slope={cfg.min_slope:.3f}  "
        f"cross_up={cross_up} cross_down={cross_down}  "
        f"can_long={can_long} can_short={can_short}"
    )

# =============================================================================
# Core trading
# =============================================================================


def run_trader(conn: sqlite3.Connection, cfg: Config, tgt: date, verbose: bool = False):
    pnl_total = 0.0
    s_utc, e_utc = jst_window_utc(tgt, cfg.jst_start, cfg.jst_end)
    print(
        f"[TRADER][{tgt} 09:00 JST] start window {cfg.jst_start}-{cfg.jst_end}, symbol={cfg.symbol}")
    print(f"[TRADER] thresholds(src=config): thr_long={cfg.thr_long}, thr_short={cfg.thr_short}, "
          f"exit_long={cfg.exit_long}, exit_short={cfg.exit_short}, ema_span={cfg.ema_span}")
    print(f"[TRADER] gates: confirm_bars={cfg.confirm_bars}, min_edge={cfg.min_edge}, min_slope={cfg.min_slope}, "
          f"vwap_gate={cfg.vwap_gate}, cooldown_bars={cfg.cooldown_bars}, side_filter={cfg.side_filter}")
    print(
        f"[TRADER] subwindow: {cfg.sub_start or 'OFF'} ~ {cfg.sub_end or 'OFF'}")

    ensure_tables(conn)

    sig = read_signals(conn, cfg.symbol, s_utc, e_utc)
    if sig.empty:
        print("[TRADER][WARN] no signals in window -> exit")
        return
    # S_ema 計算（span=cfg.ema_span）
    sig = sig.copy()
    sig["S_ema"] = sig["S"].ewm(span=cfg.ema_span, adjust=False).mean()

    px = read_prices(conn, cfg.symbol, s_utc, e_utc)
    if px.empty:
        print("[TRADER][WARN] no prices in window -> exit")
        return

    # サブ窓（JST）有効化判定
    use_sub = bool(cfg.sub_start and cfg.sub_end)
    if use_sub:
        s2_utc, e2_utc = jst_window_utc(tgt, cfg.sub_start, cfg.sub_end)
        sig = sig[(sig.index >= s2_utc) & (sig.index < e2_utc)]
        px = px[(px.index >= s2_utc) & (px.index < e2_utc)]

    # 時間基準で inner join（安全のため）
    df = sig.join(px, how="inner")
    if df.empty:
        print("[TRADER][WARN] no joined rows (signals ∩ prices) -> exit")
        return

    # 参考：VWAP gate（signals_1m に 'vwap_ok' が入っている場合のみ使う）
    has_vwap_ok = ("vwap_ok" in df.columns)
    if cfg.vwap_gate and not has_vwap_ok:
        print(
            "[TRADER][INFO] vwap_gate requested but 'vwap_ok' column not found -> gate ignored")
    use_vwap = cfg.vwap_gate and has_vwap_ok

    # 状態
    pos = 0
    open_ts = None
    open_price = None
    cooldown = 0
    trades = 0

    # 直前までの連続カウント（修正ポイント：前バー prev_s で更新）
    below_run = 0  # 直前まで thr_long 未満が連続した本数
    above_run = 0  # 直前まで thr_short 超が連続した本数
    prev_s = None

    print(f"[TRADER][{_to_jst(s_utc):%H:%M} JST] initial position={pos}")

    for ts, row in df.iterrows():
        s = float(row["S_ema"])
        price = float(row["close"])

        # サブ窓・クールダウン・VWAP
        in_subwin = True if not use_sub else (s2_utc <= ts < e2_utc)
        on_cd = cooldown > 0
        vwap_ok = bool(row["vwap_ok"]) if use_vwap else True

        # --- 1) 前バーまでの連続をカウント更新 ---
        if prev_s is None:
            below_run = 0
            above_run = 0
        else:
            if prev_s < cfg.thr_long:
                below_run += 1
            else:
                below_run = 0

            if prev_s > cfg.thr_short:
                above_run += 1
            else:
                above_run = 0

        # デバッグ出力
        _dbg_bar(ts, s, prev_s, cfg.thr_long, cfg.thr_short,
                 below_run, above_run,
                 in_subwin, on_cd, vwap_ok, verbose, cfg)

        # TTL/EODフラット（ポジション解消条件）
        ttl_hit = False
        if pos != 0 and open_ts is not None:
            if (_to_jst(ts) - _to_jst(open_ts)) >= timedelta(minutes=cfg.ttl_min):
                ttl_hit = True
        eod = (ts >= e_utc)

        # --- 2) エグジット判定 ---
        if pos > 0:
            exit_now = (s <= cfg.exit_long) or ttl_hit or eod
            if exit_now:
                fee = price * cfg.lot_size * cfg.fee_rate
                gross = (price - open_price) * cfg.lot_size
                pnl = gross - fee
                pnl_total += pnl   # ← 累計
                emit(conn, cfg.symbol, ts, "SELL", "CLOSE", price, cfg.lot_size, pnl, 0,
                     f"S_ema={s:.3f}, ttl={int((_to_jst(ts)-_to_jst(open_ts)).total_seconds()/60)}m, gross={gross:.2f}, fee={fee:.2f}")
                pos = 0
                open_ts = None
                open_price = None
                cooldown = cfg.cooldown_bars
                trades += 1
                if verbose:
                    print(
                        f"[TRADER][{_fmt_jst(ts)} JST] CLOSE LONG  @ {price:.1f} | pos -> 0 | pnl={pnl:.2f}")

        elif pos < 0:
            exit_now = (s >= cfg.exit_short) or ttl_hit or eod
            if exit_now:
                fee = price * cfg.lot_size * cfg.fee_rate
                gross = (open_price - price) * cfg.lot_size
                pnl = gross - fee
                pnl_total += pnl   # ← 累計
                emit(conn, cfg.symbol, ts, "BUY", "CLOSE", price, cfg.lot_size, pnl, 0,
                     f"S_ema={s:.3f}, ttl={int((_to_jst(ts)-_to_jst(open_ts)).total_seconds()/60)}m, gross={gross:.2f}, fee={fee:.2f}")
                pos = 0
                open_ts = None
                open_price = None
                cooldown = cfg.cooldown_bars
                trades += 1
                if verbose:
                    print(
                        f"[TRADER][{_fmt_jst(ts)} JST] CLOSE SHORT @ {price:.1f} | pos -> 0 | pnl={pnl:.2f}")

        if eod:
            break  # ウィンドウ終端

        # クールダウン進行
        if cooldown > 0:
            cooldown -= 1

        # --- 3) エントリー判定（前バーまでの連続を使用） ---
        if pos == 0 and trades < cfg.max_trades_per_day and (not on_cd) and in_subwin and vwap_ok:
            long_ok = (cfg.side_filter in ("long", "both"))
            long_ok &= (below_run >= max(1, cfg.confirm_bars))
            long_ok &= (s >= cfg.thr_long + cfg.min_edge)
            long_ok &= (prev_s is not None and (s - prev_s) >= cfg.min_slope)

            short_ok = (cfg.side_filter in ("short", "both"))
            short_ok &= (above_run >= max(1, cfg.confirm_bars))
            short_ok &= (s <= cfg.thr_short - cfg.min_edge)
            short_ok &= (prev_s is not None and (prev_s - s) >= cfg.min_slope)

            if long_ok:
                emit(conn, cfg.symbol, ts, "BUY", "OPEN", price,
                     cfg.lot_size, 0.0, +100, f"S_ema={s:.3f}")
                pos = +100
                open_ts = ts
                open_price = price
                if verbose:
                    print(
                        f"[TRADER][{_fmt_jst(ts)} JST] OPEN LONG  @ {price:.1f} | pos 0 -> 100 | S_ema={s:.3f}")

            elif short_ok:
                emit(conn, cfg.symbol, ts, "SELL", "OPEN", price,
                     cfg.lot_size, 0.0, -100, f"S_ema={s:.3f}")
                pos = -100
                open_ts = ts
                open_price = price
                if verbose:
                    print(
                        f"[TRADER][{_fmt_jst(ts)} JST] OPEN SHORT @ {price:.1f} | pos 0 -> -100 | S_ema={s:.3f}")

                case_name = getattr(cfg, "_case_name", "default")
                # 状態保存（★ここで呼ぶ）
                record_state(conn, cfg.symbol, ts, s, long_ok, short_ok,
                             below_run, above_run, pos, case_name)

        # 次バーへ
        prev_s = s

    # EOD：残っていればフラット
    if pos != 0 and open_ts is not None:
        last_ts = df.index[-1]
        price = float(df.iloc[-1]["close"])
        fee = price * cfg.lot_size * cfg.fee_rate
        if pos > 0:
            gross = (price - open_price) * cfg.lot_size
            emit(conn, cfg.symbol, last_ts, "SELL", "EOD", price, cfg.lot_size, gross - fee, 0,
                 f"EOD, gross={gross:.2f}, fee={fee:.2f}")
            if verbose:
                print(
                    f"[TRADER][{_fmt_jst(last_ts)} JST] EOD FLATTEN LONG  @ {price:.1f}")
        else:
            gross = (open_price - price) * cfg.lot_size
            emit(conn, cfg.symbol, last_ts, "BUY", "EOD", price, cfg.lot_size, gross - fee, 0,
                 f"EOD, gross={gross:.2f}, fee={fee:.2f}")
            if verbose:
                print(
                    f"[TRADER][{_fmt_jst(last_ts)} JST] EOD FLATTEN SHORT @ {price:.1f}")

    conn.commit()
    print(f"[TRADER] done for {tgt} ({cfg.symbol})")
    return pnl_total  # ✅ 追加


# 先頭付近

# ...systems_10 内 or 直後

def systems_10(bars: pd.DataFrame, feat: pd.DataFrame, use_pack3_in_systems: bool = False, pack3_weight: float = 0.2):
    # 既存の buy/sell 計算 ...
    # buy_sum, sell_sum を既に作っている前提

    if use_pack3_in_systems:
        pack3_buy = feat[[
            "or_pos", "or_break_up", "dir_streak_up", "rsi_slope3", "macd_hist_slope3", "price_to_prev_close"
        ]].mean(axis=1)
        pack3_sell = feat[[
            "or_break_dn", "dir_streak_dn", "entropy10", "ret_skew20"
        ]].mean(axis=1)
        buy_sum = (buy_sum + pack3_weight * pack3_buy) / (1.0 + pack3_weight)
        sell_sum = (sell_sum + pack3_weight * pack3_sell) / \
            (1.0 + pack3_weight)

    S = (buy_sum - sell_sum).clip(-1, 1)
    # 戻り値は既存の形式に合わせて
    out = pd.DataFrame({
        "S": S
    }, index=bars.index)
    return out


def record_state(conn, symbol, ts, S_ema, can_long, can_short,
                 below_run, above_run, pos, case_name):
    conn.execute("""
        INSERT OR REPLACE INTO trader_state_1m
        (symbol, ts, S_ema, can_long, can_short,
         below_run, above_run, pos, case_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, ts.isoformat(), float(S_ema), int(can_long), int(can_short),
          below_run, above_run, pos, case_name))
    conn.commit()

# =============================================================================
# CLI
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--use-last-session", action="store_true")
    ap.add_argument("--target-date", help="YYYY-MM-DD（JST）")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--case", type=str,
                    help="Name of test case in YAML config")
    args = ap.parse_args()

    # cfg = load_config(args.config)
    cfg = load_config(args.config, args.case)
    # cfg["_case_name"] = args.case

    conn = sqlite3.connect(cfg.db_path, timeout=5_000)
    try:
        tgt = decide_target_date(
            conn, cfg.symbol, args.use_last_session, args.target_date)
        run_trader(conn, cfg, tgt, verbose=args.verbose)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

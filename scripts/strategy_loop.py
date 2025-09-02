from __future__ import annotations
import argparse, datetime as dt, json, traceback
import numpy as np
import pandas as pd
from tq.config import load
from tq.db import connect, get_meta, put_meta, day_bars_df, day_series_df
from tq.features import compute_feature_snapshot
from tq.scheduler import sleep_to_next_minute

def _collapse_score(x) -> float:
    try:
        arr = np.asarray(x, dtype=np.float64)
        val = float(np.nanmean(arr)) if arr.ndim >= 1 else float(arr)
        if not np.isfinite(val): return 0.5
        return float(np.clip(val, 0.0, 1.0))
    except Exception:
        return 0.5

def _as5(x) -> list[float]:
    try:
        lst = list(x) if isinstance(x, (list, tuple, np.ndarray, pd.Series)) else [x]
    except Exception:
        lst = [x]
    scalars = [_collapse_score(xi) for xi in lst]
    if len(scalars) == 5:
        return scalars
    if len(scalars) == 10:
        return [float(np.nanmean(scalars[i:i+2])) for i in range(0,10,2)]
    if len(scalars) > 5:
        return scalars[:5]
    return scalars + [0.5] * (5 - len(scalars))

def _insert_signal_sql(conn, symbol: str, ts_iso_utc: str, Buy: list[float], Sell: list[float], S: float, meta: dict):
    b1,b2,b3,b4,b5 = _as5(Buy)
    s1,s2,s3,s4,s5 = _as5(Sell)
    S = float(_collapse_score(S))
    meta_json = json.dumps(meta, ensure_ascii=False, default=float)
    conn.execute("""
        INSERT INTO signals_1m
        (symbol, ts, buy1,buy2,buy3,buy4,buy5, sell1,sell2,sell3,sell4,sell5, S, meta_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(symbol, ts) DO UPDATE SET
          buy1=excluded.buy1, buy2=excluded.buy2, buy3=excluded.buy3, buy4=excluded.buy4, buy5=excluded.buy5,
          sell1=excluded.sell1, sell2=excluded.sell2, sell3=excluded.sell3, sell4=excluded.sell4, sell5=excluded.sell5,
          S=excluded.S, meta_json=excluded.meta_json
    """, (symbol, ts_iso_utc, b1,b2,b3,b4,b5, s1,s2,s3,s4,s5, S, meta_json))
    conn.commit()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    s = load(args.config)
    conn = connect(s.db_path)
    print("[STRATEGY] start")

    JST = dt.timezone(dt.timedelta(hours=9))

    while True:
        try:
            jst_now = dt.datetime.now(JST)
            date_str = jst_now.date().isoformat()

            df_day = day_bars_df(conn, s.symbol, date_str)
            if df_day is None or df_day.empty:
                raise RuntimeError("no bars today")

            # テスト中は between_time を外してもOK
            focus = df_day.between_time("09:00", "10:00")
            if focus is None or focus.empty:
                raise RuntimeError("no 9-10 yet")

            last_ts = focus.index.max()

            last_done = get_meta(conn, "strategy_last_ts", None)
            if last_done:
                last_done_ts = pd.to_datetime(last_done, utc=True, errors="coerce")
                if last_done_ts is not None and last_done_ts.tz_convert("Asia/Tokyo") >= last_ts:
                    raise RuntimeError("nothing new")

            r_nk   = day_series_df(conn, "market_1m", "kind", "NK225_FUT",   date_str)
            r_auto = day_series_df(conn, "market_1m", "kind", "SECTOR_AUTO", date_str)
            usdjpy = day_series_df(conn, "fx_1m",     "pair", "USDJPY",      date_str)

            ext = {
                "r_nk":   r_nk.pct_change().fillna(0.0)   if r_nk   is not None and not r_nk.empty   else None,
                "r_auto": r_auto.pct_change().fillna(0.0) if r_auto is not None and not r_auto.empty else None,
                "usdjpy": usdjpy                          if usdjpy is not None and not usdjpy.empty else None,
            }

            feat, Buy_raw, Sell_raw, S_raw, meta = compute_feature_snapshot(df_day, last_ts, ext)

            Buy  = _as5(Buy_raw)
            Sell = _as5(Sell_raw)
            S    = float(_collapse_score(S_raw))

            ts_iso_utc = last_ts.tz_convert("UTC").replace(microsecond=0).isoformat()
            _insert_signal_sql(conn, s.symbol, ts_iso_utc, Buy, Sell, S, meta)
            put_meta(conn, "strategy_last_ts", ts_iso_utc)

            print(f"[STRATEGY] {last_ts} S={S:.3f} B={Buy} S={Sell}")

        except Exception as e:
            print("[STRATEGY][INFO]", type(e).__name__, str(e))
            traceback.print_exc()

        sleep_to_next_minute(2)

if __name__ == "__main__":
    main()

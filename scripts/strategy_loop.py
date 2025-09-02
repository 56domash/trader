from __future__ import annotations
import argparse, datetime as dt, json
import numpy as np
import pandas as pd
from tq.config import load
from tq.db import connect, get_meta, put_meta, day_bars_df, day_series_df, insert_signal
from tq.features import compute_feature_snapshot
from tq.scheduler import sleep_to_next_minute

def _to_float_list(x, n=5):
    """シーケンス/ndarray/スカラを、長さnの純Python floatリストに強制整形。足りなければ0.5で埋める。"""
    if x is None:
        return [0.5]*n
    if isinstance(x, (float, int, np.floating, np.integer)):
        return [float(x)]*n
    try:
        lst = [float(v) for v in list(x)]
    except Exception:
        return [0.5]*n
    if len(lst) < n:
        lst = lst + [0.5]*(n-len(lst))
    elif len(lst) > n:
        lst = lst[:n]
    return lst

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

            # 当日1分足
            df_day = day_bars_df(conn, s.symbol, date_str)
            if df_day is None or df_day.empty:
                raise RuntimeError("no bars today")

            # 9-10時のみ（テスト時は between_time を 00:00-23:59 に）
            focus = df_day.between_time("09:00", "10:00")
            if focus is None or focus.empty:
                raise RuntimeError("no 9-10 yet")

            last_ts = focus.index.max()

            # 直近処理済み時刻（UTC保存）をTZ安全に比較
            last_done = get_meta(conn, "strategy_last_ts", None)
            if last_done:
                last_done_ts = pd.to_datetime(last_done, utc=True, errors="coerce")
                if last_done_ts is not None:
                    if last_done_ts.tz_convert("Asia/Tokyo") >= last_ts:
                        raise RuntimeError("nothing new")

            # 追加系列
            r_nk   = day_series_df(conn, "market_1m", "kind", "NK225_FUT",   date_str)
            r_auto = day_series_df(conn, "market_1m", "kind", "SECTOR_AUTO", date_str)
            usdjpy = day_series_df(conn, "fx_1m",     "pair", "USDJPY",      date_str)

            ext = {
                "r_nk":   r_nk.pct_change().fillna(0.0)   if r_nk   is not None and not r_nk.empty   else None,
                "r_auto": r_auto.pct_change().fillna(0.0) if r_auto is not None and not r_auto.empty else None,
                "usdjpy": usdjpy                          if usdjpy is not None and not usdjpy.empty else None,
            }

            feat, Buy, Sell, S, meta = compute_feature_snapshot(df_day, last_ts, ext)

            # --- ここが重要：完全スカラー化 ---
            Buy  = _to_float_list(Buy,  n=5)
            Sell = _to_float_list(Sell, n=5)
            S    = float(S)

            # デバッグ（もしまた失敗するなら中身を見たい）
            # print("[DEBUG] Buy:", Buy, "Sell:", Sell, "S:", S)

            ts_iso_utc = last_ts.tz_convert("UTC").replace(microsecond=0).isoformat()
            insert_signal(conn, s.symbol, ts_iso_utc, Buy, Sell, S, meta)
            put_meta(conn, "strategy_last_ts", ts_iso_utc)

            print(f"[STRATEGY] {last_ts} S={S:.3f} B={Buy} S={Sell}")

        except Exception as e:
            # 例外の型とメッセージを出す（原因追跡用）
            print("[STRATEGY][INFO]", type(e).__name__, str(e))

        # 次の分足確定+2秒で起動
        sleep_to_next_minute(2)

if __name__ == "__main__":
    main()

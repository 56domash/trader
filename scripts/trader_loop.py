
from __future__ import annotations
import argparse, datetime as dt, hashlib, pandas as pd
from tq.config import load
from tq.db import connect, get_meta, get_position, upsert_position, day_bars_df, insert_order, fill_order, get_risk_flag
from tq.exec_loop import decide_entry_exit, update_pnl_on_exit, iso_utc
from tq.scheduler import sleep_to_next_minute

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args(); s = load(args.config); conn = connect(s.db_path)
    print("[TRADER] start")
    while True:
        try:
            if get_risk_flag(conn, "HALT", "0") == "1":
                print("[TRADER] HALTED")
                sleep_to_next_minute(2); continue
            jst = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
            date_str = jst.date().isoformat()
            df_day = day_bars_df(conn, s.symbol, date_str)
            if df_day is None or df_day.empty: raise RuntimeError("no bars")
            last_sig = get_meta(conn, "strategy_last_ts", None)
            if not last_sig: raise RuntimeError("no signal yet")
            sig_ts = pd.to_datetime(last_sig).tz_convert("Asia/Tokyo")
            if sig_ts not in df_day.index: raise RuntimeError("bar not found for signal")
            px = float(df_day.loc[sig_ts, 'close'])
            cur = conn.execute("SELECT S FROM signals_1m WHERE symbol=? AND ts=?", (s.symbol, pd.to_datetime(last_sig).tz_convert('UTC').replace(microsecond=0).isoformat()))
            row = cur.fetchone()
            if not row: raise RuntimeError("signal row missing")
            S = float(row[0])
            pos = get_position(conn, s.symbol)
            side = None if not pos else pos['side']
            params = dict(thr_long=s.thr_long, thr_short=s.thr_short, exit_long=s.exit_long, exit_short=s.exit_short)
            decision = decide_entry_exit(S, side, params)
            corr_id = hashlib.sha1(f"{s.symbol}|{last_sig}".encode()).hexdigest()
            if decision == 'ENTER_LONG':
                insert_order(conn, iso_utc(), s.symbol, 'LONG', s.lot_size, px, corr_id, 'Enter')
                upsert_position(conn, s.symbol, 'LONG', s.lot_size, px, iso_utc(sig_ts.tz_convert('UTC')))
                fill_order(conn, corr_id, iso_utc(), px, None, s.symbol, 'LONG', s.lot_size)
                print(f"[TRADER] ENTER LONG @{px}")
                print(f"[TRADER] DEBUG S={S:.3f} side={side}")
            elif decision == 'ENTER_SHORT':
                insert_order(conn, iso_utc(), s.symbol, 'SHORT', s.lot_size, px, corr_id, 'Enter')
                upsert_position(conn, s.symbol, 'SHORT', s.lot_size, px, iso_utc(sig_ts.tz_convert('UTC')))
                fill_order(conn, corr_id, iso_utc(), px, None, s.symbol, 'SHORT', s.lot_size)
                print(f"[TRADER] ENTER SHORT @{px}")
                print(f"[TRADER] DEBUG S={S:.3f} side={side}")
            elif decision == 'EXIT':
                if pos and pos['side'] in ('LONG','SHORT') and pos['qty']>0 and pos['avg_px'] is not None:
                    pnl = update_pnl_on_exit(pos['side'], float(pos['avg_px']), px, int(pos['qty']), s.fee_rate)
                    insert_order(conn, iso_utc(), s.symbol, 'EXIT', int(pos['qty']), px, corr_id+"-EXIT", 'Exit')
                    fill_order(conn, corr_id+"-EXIT", iso_utc(), px, pnl, s.symbol, pos['side'], int(pos['qty']))
                    upsert_position(conn, s.symbol, None, 0, None, None)
                    print(f"[TRADER] EXIT {pos['side']} @{px} PnL={pnl:.0f}")
                    print(f"[TRADER] DEBUG S={S:.3f} side={side}")
                else:
                    print("[TRADER] nothing to exit")
                    print(f"[TRADER] DEBUG S={S:.3f} side={side}")
            else:
                print("[TRADER] HOLD")
                print(f"[TRADER] DEBUG S={S:.3f} side={side}")
        except Exception as e:
            print("[TRADER][INFO]", e)
        sleep_to_next_minute(2)

if __name__ == "__main__":
    main()

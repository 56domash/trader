
from __future__ import annotations
import argparse, datetime as dt
from tq.config import load
from tq.db import connect, set_risk_flag
from tq.scheduler import sleep_to_next_minute

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--daily_loss_limit", type=float, default=50000.0)
    args = ap.parse_args(); s = load(args.config); conn = connect(s.db_path)
    print("[RISK] start")
    while True:
        try:
            jst_date = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date().isoformat()
            cur = conn.execute("SELECT IFNULL(SUM(pnl),0.0) FROM executions WHERE substr(ts,1,10)=?", (jst_date,))
            pnl = float(cur.fetchone()[0] or 0.0)
            if pnl <= -abs(args.daily_loss_limit):
                set_risk_flag(conn, "HALT", "1"); print(f"[RISK] HALT PnL={pnl:.0f}")
            else:
                set_risk_flag(conn, "HALT", "0"); print(f"[RISK] OK PnL={pnl:.0f}")
        except Exception as e:
            print("[RISK][INFO]", e)
        sleep_to_next_minute(5)

if __name__ == "__main__":
    main()

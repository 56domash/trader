
from __future__ import annotations
import datetime as dt
from .risk import position_pnl

def iso_utc(ts=None):
    if ts is None: ts = dt.datetime.now(dt.timezone.utc)
    if ts.tzinfo is None: ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()

def decide_entry_exit(S: float, side: str|None, params):
    if side is None:
        if S >= params['thr_long']: return 'ENTER_LONG'
        if S <= params['thr_short']: return 'ENTER_SHORT'
        return None
    else:
        if side=='LONG' and (S <= params['exit_long']): return 'EXIT'
        if side=='SHORT' and (S >= params['exit_short']): return 'EXIT'
        return None

def update_pnl_on_exit(side: str, entry_px: float, exit_px: float, qty: int, fee_rate: float) -> float:
    return position_pnl(side, entry_px, exit_px, qty, fee_rate)

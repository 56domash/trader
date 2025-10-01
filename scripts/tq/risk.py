from __future__ import annotations
def position_pnl(side, entry, exitp, qty, fee_rate):
    return (exitp-entry)*qty - (entry+exitp)*qty*fee_rate if side=='LONG' else (entry-exitp)*qty - (entry+exitp)*qty*fee_rate

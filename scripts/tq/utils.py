# === FILE: tq/utils.py ===
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, timezone, time as dt_time
from zoneinfo import ZoneInfo
from typing import Optional, Tuple

import numpy as np
import pandas as pd

JST = ZoneInfo("Asia/Tokyo")

# -------- Time helpers --------

def jst_window_utc(date_jst, start_hm: Tuple[int, int]=(9,0), end_hm: Tuple[int, int]=(10,0)) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start_utc, end_utc) for a JST intraday window on a given date.
    Fix: construct times with datetime.time instead of pd.Timestamp(hour=..).
    """
    s_jst = datetime.combine(date_jst, dt_time(start_hm[0], start_hm[1])).replace(tzinfo=JST)
    e_jst = datetime.combine(date_jst, dt_time(end_hm[0], end_hm[1])).replace(tzinfo=JST)
    s = pd.Timestamp(s_jst).tz_convert("UTC")
    e = pd.Timestamp(e_jst).tz_convert("UTC")
    return s, e

# -------- Price math --------

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out = df.copy()
    out.index = idx
    return out


def vwap_day(df: pd.DataFrame, jst: ZoneInfo = JST) -> pd.Series:
    """Intraday VWAP anchored at each JST session start (00:00 UTC≈09:00 JST)."""
    if df.empty:
        return pd.Series([], dtype=float)
    x = df.copy()
    x = ensure_utc_index(x)
    j = x.index.tz_convert(jst)
    # group by YYYY-MM-DD in JST
    day_key = j.date
    tp = (x["high"] + x["low"] + x["close"]) / 3.0
    vol = x["volume"].fillna(0)
    pv = tp * vol
    # cumsum per day
    g = pd.Series(1, index=x.index).groupby(pd.Index(day_key)).cumsum()  # per-day running counter
    # Using groupby on day_key via map to align
    df_pv = pv.groupby(pd.Index(day_key)).cumsum()
    df_v  = vol.groupby(pd.Index(day_key)).cumsum()
    vwap = (df_pv / df_v).ffill().fillna(x["close"])  # if zero vol, fall back close
    vwap.index = x.index
    return vwap


def rolling_zscore(s: pd.Series, win: int, clip: Optional[Tuple[float,float]]=(-3,3)) -> pd.Series:
    m = s.rolling(win, min_periods=max(2, win//2)).mean()
    sd = s.rolling(win, min_periods=max(2, win//2)).std()
    z = (s - m) / sd.replace(0, np.nan)
    if clip:
        z = z.clip(lower=clip[0], upper=clip[1])
    return z.fillna(0.0)


def norm01(x: pd.Series, lo: float=-3.0, hi: float=3.0) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    x = x.clip(lower=lo, upper=hi)
    return (x - lo) / (hi - lo)


def safe_mean(cols: list[pd.Series], default: float=0.5) -> pd.Series:
    cols = [pd.to_numeric(c, errors="coerce") for c in cols if c is not None]
    if not cols:
        return pd.Series(default, index=None)
    df = pd.concat(cols, axis=1)
    return df.mean(axis=1).fillna(default)

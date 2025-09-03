
# === FILE: tq/features.py ===
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_utc_index, vwap_day, rolling_zscore, norm01, safe_mean

@dataclass
class FeatureSpec:
    name: str
    fn: Callable[[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]], pd.Series]
    kind: str = "tech"
    norm: Tuple[float, float] = (-3.0, 3.0)  # for norm01; use (0,1) to skip

# ---- base helpers ----

def _ret(df: pd.DataFrame, n: int=1) -> pd.Series:
    return df["close"].pct_change(n)


def _slope(s: pd.Series, n: int) -> pd.Series:
    return s.diff(n) / max(n, 1)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


# ---- Feature functions (first ~25 already) ----

def f_ret1(df, *_):  return _ret(df, 1)

def f_ret3(df, *_):  return _ret(df, 3)

def f_ret5(df, *_):  return _ret(df, 5)

def f_ret10(df,*_):  return _ret(df, 10)

def f_ret20(df,*_):  return _ret(df, 20)


def f_mom_slope3(df,*_):
    r = _ret(df, 1).fillna(0)
    return _slope(r, 3)


def f_vwap_gap(df,*_):
    vwap = df.get("vwap")
    if vwap is None or vwap.isna().all():
        vwap = vwap_day(df)
    return (df["close"] - vwap) / vwap.replace(0, np.nan)


def f_bb_pos(df,*_):
    mid = df["close"].rolling(20).mean()
    sd  = df["close"].rolling(20).std()
    up = mid + 2*sd
    lo = mid - 2*sd
    return (df["close"] - lo) / (up - lo)


def f_bb_bw(df,*_):
    mid = df["close"].rolling(20).mean()
    sd  = df["close"].rolling(20).std()
    up = mid + 2*sd
    lo = mid - 2*sd
    return (up - lo) / mid.replace(0, np.nan)


def f_keltner_pos(df,*_):
    ema20 = _ema(df["close"], 20)
    tr = (df[["high","low","close"]].max(axis=1) - df[["high","low","close"]].min(axis=1)).rolling(20).mean()
    up = ema20 + 2*tr
    lo = ema20 - 2*tr
    return (df["close"] - lo) / (up - lo)


def f_rsi14(df,*_):
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100.0


def f_stoch_k(df,*_):
    ll = df["low"].rolling(14).min()
    hh = df["high"].rolling(14).max()
    return (df["close"] - ll) / (hh - ll)


def f_stoch_d(df,*_):
    return f_stoch_k(df).rolling(3).mean()


def f_atr14_z(df,*_):
    tr = (df[["high","low","close"]].max(axis=1) - df[["high","low","close"]].min(axis=1))
    atr = tr.rolling(14).mean()
    return rolling_zscore(atr, 60)


def f_true_range_z(df,*_):
    tr = (df[["high","low","close"]].max(axis=1) - df[["high","low","close"]].min(axis=1))
    return rolling_zscore(tr, 60)


def f_or_pos(df,*_):
    # Opening Range 5分（09:00-09:05 JST）に対する位置
    j = df.index.tz_convert("Asia/Tokyo")
    first5 = (j.time >= pd.Timestamp("09:00").time()) & (j.time < pd.Timestamp("09:05").time())
    or_high = df["high"].where(first5).groupby(j.date).transform("max")
    or_low  = df["low"].where(first5).groupby(j.date).transform("min")
    return (df["close"] - or_low) / (or_high - or_low)


def f_vol_z60(df,*_):
    return rolling_zscore(df["volume"].fillna(0), 60)


def f_vol_ratio_5_20(df,*_):
    v5 = df["volume"].rolling(5).mean()
    v20= df["volume"].rolling(20).mean()
    return (v5 / v20.replace(0, np.nan)) - 1.0


def f_obv_slope(df,*_):
    delta = np.sign(df["close"].diff().fillna(0))
    obv = (delta * df["volume"].fillna(0)).cumsum()
    return _slope(obv.fillna(0), 5)


def f_ma_fast_slow(df,*_):
    ema5 = _ema(df["close"], 5)
    ema20= _ema(df["close"], 20)
    return (ema5 - ema20) / ema20.replace(0, np.nan)


def f_price_above_vwap(df,*_):
    vwap = df.get("vwap")
    if vwap is None or vwap.isna().all():
        vwap = vwap_day(df)
    return (df["close"] >= vwap).astype(float)

# ---- FX / exogenous (first pack) ----

def f_fx_ret5(df, df_fx: Optional[pd.DataFrame], *_):
    if df_fx is None or df_fx.empty or "close" not in df_fx:
        return pd.Series(0.0, index=df.index)
    fx = df_fx["close"].reindex(df.index, method="nearest")
    return fx.pct_change(5)


def f_fx_corr20(df, df_fx: Optional[pd.DataFrame], *_):
    if df_fx is None or df_fx.empty or "close" not in df_fx:
        return pd.Series(0.0, index=df.index)
    px = df["close"].pct_change().fillna(0)
    fx = df_fx["close"].reindex(df.index, method="nearest").pct_change().fillna(0)
    return px.rolling(20).corr(fx)

# ---- NEW: Next 25 features ----

def f_gap_open(df,*_):
    prev_close = df["close"].shift(1)
    return (df["open"] - prev_close) / prev_close


def f_or_range_pct(df,*_):
    j = df.index.tz_convert("Asia/Tokyo")
    first5 = (j.time >= pd.Timestamp("09:00").time()) & (j.time < pd.Timestamp("09:05").time())
    or_high = df["high"].where(first5).groupby(j.date).transform("max")
    or_low  = df["low"].where(first5).groupby(j.date).transform("min")
    rng = (or_high - or_low)
    return rng / df["close"].replace(0, np.nan)


def f_or_break_up(df,*_):
    j = df.index.tz_convert("Asia/Tokyo")
    first5 = (j.time >= pd.Timestamp("09:00").time()) & (j.time < pd.Timestamp("09:05").time())
    or_high = df["high"].where(first5).groupby(j.date).transform("max")
    or_low  = df["low"].where(first5).groupby(j.date).transform("min")
    base = (or_high - or_low).replace(0, np.nan)
    return (df["close"] - or_high) / base


def f_or_break_down(df,*_):
    j = df.index.tz_convert("Asia/Tokyo")
    first5 = (j.time >= pd.Timestamp("09:00").time()) & (j.time < pd.Timestamp("09:05").time())
    or_high = df["high"].where(first5).groupby(j.date).transform("max")
    or_low  = df["low"].where(first5).groupby(j.date).transform("min")
    base = (or_high - or_low).replace(0, np.nan)
    return (or_low - df["close"]) / base


def f_ema20_slope(df,*_):
    e20 = _ema(df["close"], 20)
    return e20.diff(1)


def f_hl_range_pct20(df,*_):
    hi = df["high"].rolling(20).max()
    lo = df["low"].rolling(20).min()
    return (hi - lo) / df["close"].replace(0, np.nan)


def f_tr_ratio_5_20(df,*_):
    tr = (df[["high","low","close"]].max(axis=1) - df[["high","low","close"]].min(axis=1))
    tr5 = tr.rolling(5).mean()
    tr20= tr.rolling(20).mean()
    return (tr5 / tr20.replace(0, np.nan)) - 1.0


def f_bb_bw_slope(df,*_):
    bw = f_bb_bw(df)
    return bw.diff(3)


def f_pullback_depth5(df,*_):
    m5 = df["high"].rolling(5).max()
    return (m5 - df["close"]) / m5.replace(0, np.nan)


def f_breakout_strength5(df,*_):
    m5 = df["high"].rolling(5).max()
    return (df["close"] - m5) / df["close"].replace(0, np.nan)


def f_vwap_resid_z60(df,*_):
    vwap = df.get("vwap")
    if vwap is None or vwap.isna().all():
        vwap = vwap_day(df)
    resid = df["close"] - vwap
    return rolling_zscore(resid, 60)


def f_vwap_cross_recent3(df,*_):
    vwap = df.get("vwap")
    if vwap is None or vwap.isna().all():
        vwap = vwap_day(df)
    sign = np.sign((df["close"] - vwap).fillna(0))
    crossed = (sign != sign.shift(1)).astype(float)
    return crossed.rolling(3).max()  # 直近3本でクロスあれば1


def f_ema5_dist(df,*_):
    e5 = _ema(df["close"], 5)
    return (df["close"] - e5) / e5.replace(0, np.nan)


def f_ema20_dist(df,*_):
    e20 = _ema(df["close"], 20)
    return (df["close"] - e20) / e20.replace(0, np.nan)


def f_ema5_above20(df,*_):
    e5 = _ema(df["close"], 5)
    e20= _ema(df["close"], 20)
    return (e5 >= e20).astype(float)


def f_macd_norm(df,*_):
    e12 = _ema(df["close"], 12)
    e26 = _ema(df["close"], 26)
    macd = e12 - e26
    return macd / e26.replace(0, np.nan)


def f_macd_sig_diff(df,*_):
    e12 = _ema(df["close"], 12)
    e26 = _ema(df["close"], 26)
    macd = e12 - e26
    sig = _ema(macd, 9)
    return macd - sig

# ADX / DI (簡易)

def _dx(df):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = (df[["high","low","close"]].max(axis=1) - df[["high","low","close"]].min(axis=1))
    atr14 = tr.rolling(14).mean()
    pdi = 100 * (plus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    mdi = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    dx = ( (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan) ) * 100
    return dx, pdi, mdi


def f_adx14_norm(df,*_):
    dx, _, _ = _dx(df)
    adx = dx.rolling(14).mean() / 100.0
    return adx


def f_di_diff(df,*_):
    _, pdi, mdi = _dx(df)
    return (pdi - mdi) / 100.0

# Money Flow系

def f_mfi14(df,*_):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    delta = tp.diff()
    pos_mf = (tp * df["volume"]).where(delta > 0, 0.0)
    neg_mf = (tp * df["volume"]).where(delta < 0, 0.0)
    pm = pos_mf.rolling(14).sum()
    nm = neg_mf.rolling(14).sum()
    mfi = 100 - 100 / (1 + (pm / nm.replace(0, np.nan)))
    return (mfi / 100.0)


def f_cmf20(df,*_):
    mf_mult = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    mf_vol = mf_mult * df["volume"].fillna(0)
    return (mf_vol.rolling(20).sum() / df["volume"].rolling(20).sum().replace(0, np.nan))


def f_obv_z60(df,*_):
    delta = np.sign(df["close"].diff().fillna(0))
    obv = (delta * df["volume"].fillna(0)).cumsum()
    return rolling_zscore(obv, 60)


def f_vol_ma_slope20(df,*_):
    vma = df["volume"].rolling(20).mean()
    return vma.diff(3)

# FX追加

def f_fx_ret1(df, df_fx: Optional[pd.DataFrame], *_):
    if df_fx is None or df_fx.empty or "close" not in df_fx:
        return pd.Series(0.0, index=df.index)
    fx = df_fx["close"].reindex(df.index, method="nearest")
    return fx.pct_change(1)


def f_fx_corr60(df, df_fx: Optional[pd.DataFrame], *_):
    if df_fx is None or df_fx.empty or "close" not in df_fx:
        return pd.Series(0.0, index=df.index)
    px = df["close"].pct_change().fillna(0)
    fx = df_fx["close"].reindex(df.index, method="nearest").pct_change().fillna(0)
    return px.rolling(60).corr(fx)


FEATURES: list[FeatureSpec] = [
    # Momentum / Price
    FeatureSpec("ret1", f_ret1, "tech", (-0.01, 0.01)),
    FeatureSpec("ret3", f_ret3, "tech", (-0.02, 0.02)),
    FeatureSpec("ret5", f_ret5, "tech", (-0.03, 0.03)),
    FeatureSpec("ret10", f_ret10, "tech", (-0.05, 0.05)),
    FeatureSpec("ret20", f_ret20, "tech", (-0.08, 0.08)),
    FeatureSpec("mom_slope3", f_mom_slope3, "tech", (-0.01, 0.01)),

    # VWAP / Bands / Trend
    FeatureSpec("vwap_gap", f_vwap_gap, "tech", (-0.005, 0.005)),
    FeatureSpec("bb_pos", f_bb_pos, "tech", (0.0, 1.0)),
    FeatureSpec("bb_bw", f_bb_bw, "tech", (0.0, 0.06)),
    FeatureSpec("bb_bw_slope", f_bb_bw_slope, "tech", (-0.01, 0.01)),
    FeatureSpec("kelt_pos", f_keltner_pos, "tech", (0.0, 1.0)),
    FeatureSpec("ma_fast_slow", f_ma_fast_slow, "tech", (-0.01, 0.01)),
    FeatureSpec("ema20_slope", f_ema20_slope, "tech", (-3, 3)),
    FeatureSpec("ema5_dist", f_ema5_dist, "tech", (-0.01, 0.01)),
    FeatureSpec("ema20_dist", f_ema20_dist, "tech", (-0.02, 0.02)),
    FeatureSpec("ema5_above20", f_ema5_above20, "tech", (0.0, 1.0)),
    FeatureSpec("price_above_vwap", f_price_above_vwap, "tech", (0.0, 1.0)),


    # Oscillators
    FeatureSpec("rsi14", f_rsi14, "tech", (0.0, 1.0)),
    FeatureSpec("stoch_k", f_stoch_k, "tech", (0.0, 1.0)),
    FeatureSpec("stoch_d", f_stoch_d, "tech", (0.0, 1.0)),
    FeatureSpec("macd_norm", f_macd_norm, "tech", (-0.01, 0.01)),
    FeatureSpec("macd_sig_diff", f_macd_sig_diff, "tech", (-0.01, 0.01)),

    # Volatility / Range
    FeatureSpec("atr14_z", f_atr14_z, "tech", (-3, 3)),
    FeatureSpec("tr_z", f_true_range_z, "tech", (-3, 3)),
    FeatureSpec("tr_ratio_5_20", f_tr_ratio_5_20, "tech", (-0.5, 0.5)),
    FeatureSpec("hl_range_pct20", f_hl_range_pct20, "tech", (0.0, 0.05)),

    # Opening range / Breakouts
    FeatureSpec("or_pos", f_or_pos, "tech", (0.0, 1.0)),
    FeatureSpec("or_range_pct", f_or_range_pct, "tech", (0.0, 0.02)),
    FeatureSpec("or_break_up", f_or_break_up, "tech", (-0.02, 0.02)),
    FeatureSpec("or_break_down", f_or_break_down, "tech", (-0.02, 0.02)),

    # Pullback / Breakout micro
    FeatureSpec("pullback_depth5", f_pullback_depth5, "tech", (0.0, 0.02)),
    FeatureSpec("breakout_strength5", f_breakout_strength5, "tech", (-0.02, 0.02)),

    # Volume / Flow
    FeatureSpec("vol_z60", f_vol_z60, "tech", (-3, 3)),
    FeatureSpec("vol_ratio_5_20", f_vol_ratio_5_20, "tech", (-0.5, 0.5)),
    FeatureSpec("vol_ma_slope20", f_vol_ma_slope20, "tech", (-3, 3)),
    FeatureSpec("obv_slope", f_obv_slope, "tech", (-3, 3)),
    FeatureSpec("obv_z60", f_obv_z60, "tech", (-3, 3)),
    FeatureSpec("mfi14", f_mfi14, "tech", (0.0, 1.0)),
    FeatureSpec("cmf20", f_cmf20, "tech", (-1.0, 1.0)),

    # VWAP extras
    FeatureSpec("vwap_resid_z60", f_vwap_resid_z60, "tech", (-3, 3)),
    FeatureSpec("vwap_cross3", f_vwap_cross_recent3, "tech", (0.0, 1.0)),

    # FX
    FeatureSpec("fx_ret1", f_fx_ret1, "fx", (-0.01, 0.01)),
    FeatureSpec("fx_ret5", f_fx_ret5, "fx", (-0.01, 0.01)),
    FeatureSpec("fx_corr20", f_fx_corr20, "fx", (-1.0, 1.0)),
    FeatureSpec("fx_corr60", f_fx_corr60, "fx", (-1.0, 1.0)),
]


def compute_all(df_bars: pd.DataFrame, df_fx: Optional[pd.DataFrame]=None, df_mkt: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    if df_bars is None or df_bars.empty:
        return pd.DataFrame()
    bars = ensure_utc_index(df_bars)
    out = {}
    for spec in FEATURES:
        try:
            s = spec.fn(bars, df_fx, df_mkt)
        except Exception:
            s = pd.Series(np.nan, index=bars.index)
        if spec.norm == (0.0, 1.0):
            out[spec.name] = s.astype(float).clip(0.0, 1.0).fillna(0.5)
        else:
            out[spec.name] = norm01(s, *spec.norm).fillna(0.5)
    return pd.DataFrame(out, index=bars.index)

def f_price_above_vwap(df, *_):
    vwap = df.get("vwap")
    if vwap is None or vwap.isna().all():
        from .utils import vwap_day
        vwap = vwap_day(df)
    return (df["close"] >= vwap).astype(float)

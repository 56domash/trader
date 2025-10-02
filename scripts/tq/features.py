
# === FILE: tq/features.py ===
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
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



# =========================
# 共通ヘルパ
# =========================
def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
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

def _ema(s: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").ewm(span=span, adjust=False).mean()

def _rolling_mean(s: pd.Series, win: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rolling(win, min_periods=max(2, win//2)).mean()

def _rolling_std(s: pd.Series, win: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rolling(win, min_periods=max(2, win//2)).std()

def _rolling_max(s: pd.Series, win: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rolling(win, min_periods=max(2, win//2)).max()

def _rolling_min(s: pd.Series, win: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rolling(win, min_periods=max(2, win//2)).min()

def _rolling_z(s: pd.Series, win: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m = _rolling_mean(s, win)
    sd = _rolling_std(s, win)
    z = (s - m) / sd.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3, 3)

def _norm01(s: pd.Series, lo=-3.0, hi=3.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float).clip(lo, hi)
    return (s - lo) / (hi - lo)

def _vwap_day(df: pd.DataFrame, tz="Asia/Tokyo") -> pd.Series:
    if df.empty: return pd.Series([], dtype=float)
    x = _ensure_utc(df)
    j = x.index.tz_convert(tz)
    day_key = pd.Index(j.date)
    tp = (x["high"] + x["low"] + x["close"]) / 3.0
    vol = pd.to_numeric(x["volume"], errors="coerce").fillna(0.0)
    pv = tp * vol
    pv_c = pv.groupby(day_key).cumsum()
    v_c  = vol.groupby(day_key).cumsum()
    vwap = (pv_c / v_c).ffill().fillna(x["close"])
    vwap.index = x.index
    return vwap

def _true_range(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low  = pd.to_numeric(df["low"], errors="coerce")
    close_prev = pd.to_numeric(df["close"].shift(1), errors="coerce")
    tr = pd.concat([
        (high - low).abs(),
        (high - close_prev).abs(),
        (low  - close_prev).abs()
    ], axis=1).max(axis=1)
    return tr

# =========================
# Pack1（初期の25系：モメンタム/ボラ/OR/VWAP/出来高）
# =========================
def compute_pack1(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_utc(df)
    out = pd.DataFrame(index=df.index)

    close = pd.to_numeric(df["close"], errors="coerce")
    high  = pd.to_numeric(df["high"], errors="coerce")
    low   = pd.to_numeric(df["low"], errors="coerce")
    vol   = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    # モメンタム類
    ret1 = close.pct_change().fillna(0.0)
    ret3 = close.pct_change(3).fillna(0.0)
    ret5 = close.pct_change(5).fillna(0.0)
    out["p1_ret1_z"] = _rolling_z(ret1, 10)
    out["p1_ret3_z"] = _rolling_z(ret3, 20)
    out["p1_ret5_z"] = _rolling_z(ret5, 30)

    # ATRとボラ正規化
    tr = _true_range(df)
    atr14 = _rolling_mean(tr, 14)
    out["p1_atr14_norm"] = (atr14 / close.replace(0, np.nan)).fillna(0.0).clip(0, 0.05)

    # 当日OR（9-10の高安レンジを日中で参照：窓内でも安全に）
    # 累積の当日高安で近似（JST日でグループしたcummax/cummin）
    j = df.index.tz_convert("Asia/Tokyo")
    g = pd.Index(j.date)
    day_hi = high.groupby(g).cummax()
    day_lo = low.groupby(g).cummin()
    rng = (day_hi - day_lo).replace(0, np.nan)
    out["p1_or_pos"] = ((close - day_lo) / rng).clip(0, 1).fillna(0.5)

    # VWAP 乖離
    vwap = _vwap_day(df)
    out["p1_vwap_dist01"] = _norm01(_rolling_z((close - vwap) / close.replace(0, np.nan), 20))

    # ボリンジャー幅・位置
    m20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std().replace(0, np.nan)
    out["p1_bb_pos01"] = _norm01(((close - m20) / (2 * sd20)).clip(-3, 3))
    out["p1_bb_bw01"]  = _norm01((sd20 / close.replace(0, np.nan)).fillna(0.0), lo=0, hi=0.05)

    # 出来高変化・偏り
    vol_ma20 = vol.rolling(20).mean()
    out["p1_vol_ratio20"] = ((vol / vol_ma20.replace(0, np.nan)).clip(0, 5)).fillna(0.0)  # 0〜5を想定
    up = (close.diff() > 0).astype(int)
    dn = (close.diff() < 0).astype(int)
    upv = (vol * up).rolling(10).sum()
    dnv = (vol * dn).rolling(10).sum()
    out["p1_vol_imb01"] = ((upv - dnv) / (upv + dnv).replace(0, np.nan)).fillna(0.0) * 0.5 + 0.5

    # RSI / Stoch（軽量版）
    gain = (close.diff().clip(lower=0)).rolling(14).mean()
    loss = (-close.diff().clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    out["p1_rsi14_01"] = (rsi.fillna(50.0) / 100.0)

    hh14 = high.rolling(14).max()
    ll14 = low.rolling(14).min()
    stoch_k = (close - ll14) / (hh14 - ll14).replace(0, np.nan)
    out["p1_stoch_k01"] = stoch_k.fillna(0.5).clip(0, 1)

    return out

# =========================
# Pack2（既存：従来Sのベース）
# =========================
def compute_pack2(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_utc(df)
    out = pd.DataFrame(index=df.index)

    close = pd.to_numeric(df["close"], errors="coerce")
    vol   = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    # 短期モメンタム
    ret1 = close.pct_change().fillna(0.0)
    out["p2_ret1_z"] = _rolling_z(ret1, 10)
    out["p2_ret1"]   = _norm01(out["p2_ret1_z"])

    # VWAP 乖離
    vwap = _vwap_day(df)
    out["p2_vwap_gap"] = (close - vwap) / close.replace(0, np.nan)
    out["p2_vwap_gap_z"] = _rolling_z(out["p2_vwap_gap"], 20)
    out["p2_vwap_gap01"] = _norm01(out["p2_vwap_gap_z"])

    # MAクロス
    ma5  = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    out["p2_ma_fast_slow"] = _norm01((ma5 - ma20) / close.replace(0, np.nan))

    # BB位置
    m20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std().replace(0, np.nan)
    bb_pos = (close - m20) / (2 * sd20)
    out["p2_bb_pos"] = _norm01(bb_pos.clip(-3, 3))

    # 出来高z
    vol_z = _rolling_z(vol, 20)
    out["p2_vol_z01"] = _norm01(vol_z)

    # ---- 既存Sの5買い/5売りの“素点”をここで定義（従来互換）
    out["buy1_base"]  = out["p2_ret1"]
    out["buy2_base"]  = out["p2_vwap_gap01"]
    out["buy3_base"]  = out["p2_ma_fast_slow"]
    out["buy4_base"]  = out["p2_bb_pos"]
    out["buy5_base"]  = out["p2_vol_z01"]

    out["sell1_base"] = 1.0 - out["p2_ret1"]
    out["sell2_base"] = 1.0 - out["p2_vwap_gap01"]
    out["sell3_base"] = 1.0 - out["p2_ma_fast_slow"]
    out["sell4_base"] = 1.0 - out["p2_bb_pos"]
    out["sell5_base"] = 1.0 - out["p2_vol_z01"]

    return out

# =========================
# Pack3（追加：参考列のみ、Sは触らない）
# =========================
def compute_pack3(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_utc(df)
    out = pd.DataFrame(index=df.index)

    close = pd.to_numeric(df["close"], errors="coerce")
    high  = pd.to_numeric(df["high"], errors="coerce")
    low   = pd.to_numeric(df["low"], errors="coerce")
    vol   = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    # MACD(12,26,9)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd  = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    out["p3_macd_hist"] = hist.fillna(0.0)
    out["p3_macd_hist01"] = _norm01(_rolling_z(out["p3_macd_hist"], 30))

    # Williams %R(10)（-100〜0）→ 0〜1
    high10 = high.rolling(10).max()
    low10  = low.rolling(10).min()
    wr10 = -100 * (high10 - close) / (high10 - low10).replace(0, np.nan)
    out["p3_wr10"] = wr10.fillna(0.0).clip(-100, 0)
    out["p3_wr10_01"] = (out["p3_wr10"] - (-100.0)) / 100.0

    # 出来高スパイク
    out["p3_vol_spike20"] = _rolling_z(vol, 20)
    out["p3_vol_spike01"] = _norm01(out["p3_vol_spike20"])

    # 追加の参考指標（軽量版）
    # Keltner位置（EMA20 ± ATR*1.5）
    ema20 = _ema(close, 20)
    tr = _true_range(df)
    atr20 = _rolling_mean(tr, 20)
    kel_up = ema20 + 1.5 * atr20
    kel_dn = ema20 - 1.5 * atr20
    out["p3_keltner_pos01"] = ((close - kel_dn) / (kel_up - kel_dn).replace(0, np.nan)).clip(0, 1).fillna(0.5)

    # CCI(20)っぽい指標の01化（簡易）
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(20).mean()
    md = (tp - sma_tp).abs().rolling(20).mean()
    cci20 = (tp - sma_tp) / (0.015 * md.replace(0, np.nan))
    out["p3_cci20_01"] = _norm01(cci20.clip(-200, 200), lo=-200, hi=200)

    return out


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


# === FILE: tq/features.py ===
# ...（既存のimport, utilsなどは省略）

# =========================
# Pack4（追加：外部市場・日次・マルチTF・季節性）
# =========================
def compute_pack4(df: pd.DataFrame,
                  df_fx: Optional[pd.DataFrame]=None,
                  df_mkt: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    df = ensure_utc_index(df)
    out = pd.DataFrame(index=df.index)

    close = pd.to_numeric(df["close"], errors="coerce")
    high  = pd.to_numeric(df["high"], errors="coerce")
    low   = pd.to_numeric(df["low"], errors="coerce")
    vol   = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    # --- マルチタイムフレーム（5分足近似: rolling 5本）
    out["p4_ret5m"] = _rolling_mean(close.pct_change(), 5)
    out["p4_ret15m"] = _rolling_mean(close.pct_change(), 15)
    out["p4_vol5m_z"] = rolling_zscore(vol.rolling(5).mean(), 60)
    out["p4_hl_range5m"] = (high.rolling(5).max() - low.rolling(5).min()) / close.replace(0, np.nan)
    ma5m = close.rolling(5).mean()
    ma20m = close.rolling(20).mean()
    out["p4_ema5m_cross"] = (ma5m - ma20m) / ma20m.replace(0, np.nan)

    # --- 日次（日足っぽい特徴: JST日グループ）
    jst = df.index.tz_convert("Asia/Tokyo")
    g = pd.Index(jst.date)
    day_close = close.groupby(g).transform("last")
    day_vol = vol.groupby(g).transform("sum")
    day_hi = high.groupby(g).transform("max")
    day_lo = low.groupby(g).transform("min")

    out["p4_daily_ret"] = (day_close - day_close.shift(1)) / day_close.shift(1)
    out["p4_daily_vol_z"] = rolling_zscore(day_vol, 20)
    out["p4_daily_range_pct"] = (day_hi - day_lo) / day_close.replace(0, np.nan)
    out["p4_prev_close_gap"] = (close - day_close.shift(1)) / day_close.shift(1)

    dow = pd.Series(jst.weekday, index=df.index)
    out["p4_dow_sin"] = np.sin(2*np.pi*dow/7)
    # 曜日cosを入れたければここで追加可能

    # --- 外部市場（df_mkt: NK225_FUT, SECTOR_AUTOなど）
    if df_mkt is not None and not df_mkt.empty:
        mkt = df_mkt.reindex(df.index, method="nearest")
        if "NK225_FUT" in mkt:
            out["p4_corr_nk225"] = close.pct_change().rolling(30).corr(mkt["NK225_FUT"].pct_change())
        if "SECTOR_AUTO" in mkt:
            out["p4_corr_auto"] = close.pct_change().rolling(30).corr(mkt["SECTOR_AUTO"].pct_change())

    # --- ダミーで他市場（例: SP500, GOLD, OIL）は0.5埋め（拡張可能）
    # out["p4_corr_sp500"] = 0.5
    # out["p4_corr_gold"] = 0.5
    # out["p4_corr_oil"] = 0.5
    # out["p4_corr_vix"] = 0.5
    # out["p4_corr_bond"] = 0.5

        # --- 外部市場（df_mkt: SP500, GOLD, OIL, VIX, BOND10Y）
    if df_mkt is not None and not df_mkt.empty:
        if "SP500" in df_mkt:
            out["p4_corr_sp500"] = close.pct_change().rolling(30).corr(df_mkt["SP500"].pct_change())
        if "GOLD" in df_mkt:
            out["p4_corr_gold"] = close.pct_change().rolling(30).corr(df_mkt["GOLD"].pct_change())
        if "OIL" in df_mkt:
            out["p4_corr_oil"] = close.pct_change().rolling(30).corr(df_mkt["OIL"].pct_change())
        if "VIX" in df_mkt:
            out["p4_corr_vix"] = close.pct_change().rolling(30).corr(df_mkt["VIX"].pct_change())
        if "BOND10Y" in df_mkt:
            out["p4_corr_bond"] = close.pct_change().rolling(30).corr(df_mkt["BOND10Y"].pct_change())
    else:
        out["p4_corr_sp500"] = 0.5
        out["p4_corr_gold"]  = 0.5
        out["p4_corr_oil"]   = 0.5
        out["p4_corr_vix"]   = 0.5
        out["p4_corr_bond"]  = 0.5


    # --- リスク指標
    out["p4_beta_mkt30"] = close.pct_change().rolling(30).cov(day_close.pct_change()) / (
        day_close.pct_change().rolling(30).var().replace(0, np.nan)
    )
    out["p4_vol_regime_mkt"] = (close.pct_change().rolling(30).std() /
                                close.pct_change().rolling(120).std().replace(0, np.nan))

    # --- 時間・シーズナリティ
    # mins = jst.hour*60 + jst.minute
    # out["p4_mins_to_close"] = (60 - (mins - 9*60)).clip(0, 60) / 60.0
    # week_of_month = ((jst.day-1)//7 + 1).astype(float)
    # out["p4_week_of_month"] = week_of_month/4.0
    # # holiday_dummyは簡易に週末前 = 金曜とする
    # out["p4_holiday_dummy"] = (dow==4).astype(float)
    # month = jst.month
    # out["p4_month_sin"] = np.sin(2*np.pi*month/12)
    # out["p4_month_cos"] = np.cos(2*np.pi*month/12)

    # --- 時間・シーズナリティ
    mins = pd.Series(jst.hour*60 + jst.minute, index=df.index)
    out["p4_mins_to_close"] = (60 - (mins - 9*60)).clip(0, 60) / 60.0

    week_of_month = ((jst.day-1)//7 + 1).astype(float)
    out["p4_week_of_month"] = week_of_month/4.0
    out["p4_holiday_dummy"] = (dow==4).astype(float)

    month = jst.month
    out["p4_month_sin"] = np.sin(2*np.pi*month/12)
    out["p4_month_cos"] = np.cos(2*np.pi*month/12)


    return out


# ========= Pack4 spec =========
PACK4 = [
    FeatureSpec("ret5m", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_ret5m"]),
    FeatureSpec("ret15m", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_ret15m"]),
    FeatureSpec("vol5m_z", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_vol5m_z"]),
    FeatureSpec("hl_range5m", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_hl_range5m"]),
    FeatureSpec("ema5m_cross", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_ema5m_cross"]),
    FeatureSpec("daily_ret", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_daily_ret"]),
    FeatureSpec("daily_vol_z", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_daily_vol_z"]),
    FeatureSpec("daily_range_pct", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_daily_range_pct"]),
    FeatureSpec("prev_close_gap", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_prev_close_gap"]),
    FeatureSpec("dow_sin", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_dow_sin"]),
    FeatureSpec("corr_nk225", lambda df,fx,mkt: compute_pack4(df,fx,mkt).get("p4_corr_nk225", pd.Series(0.5, index=df.index))),
    FeatureSpec("corr_auto", lambda df,fx,mkt: compute_pack4(df,fx,mkt).get("p4_corr_auto", pd.Series(0.5, index=df.index))),
    FeatureSpec("corr_sp500", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_corr_sp500"]),
    FeatureSpec("corr_gold", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_corr_gold"]),
    FeatureSpec("corr_oil", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_corr_oil"]),
    FeatureSpec("corr_vix", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_corr_vix"]),
    FeatureSpec("corr_bond", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_corr_bond"]),
    FeatureSpec("beta_mkt30", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_beta_mkt30"]),
    FeatureSpec("vol_regime_mkt", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_vol_regime_mkt"]),
    FeatureSpec("mins_to_close", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_mins_to_close"]),
    FeatureSpec("week_of_month", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_week_of_month"]),
    FeatureSpec("holiday_dummy", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_holiday_dummy"]),
    FeatureSpec("month_sin", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_month_sin"]),
    FeatureSpec("month_cos", lambda df,fx,mkt: compute_pack4(df,fx,mkt)["p4_month_cos"]),
]

# =========================
# エクスポート：4 Packをまとめる
# =========================
def compute_packs(df: pd.DataFrame,
                  use_pack1: bool=True,
                  use_pack2: bool=True,
                  use_pack3: bool=True,
                  use_pack4: bool=True,
                  df_fx: Optional[pd.DataFrame]=None,
                  df_mkt: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    parts = []
    if use_pack1:
        parts.append(compute_pack1(df))
    if use_pack2:
        parts.append(compute_pack2(df))
    if use_pack3:
        parts.append(compute_pack3(df))
    if use_pack4:
        parts.append(compute_pack4(df, df_fx, df_mkt))
    if not parts:
        return pd.DataFrame(index=ensure_utc_index(df).index)
    return pd.concat(parts, axis=1)



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

# ========= Pack3 helpers =========
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Tuple, Dict

try:
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
except Exception:
    JST = None

def _safe_series(val, index):
    """valをSeriesにしてindexに合わせる。例外時は0.5で返す"""
    try:
        s = pd.Series(val, index=index)
        s = pd.to_numeric(s, errors="coerce")
        return s.reindex(index)
    except Exception:
        return pd.Series(0.5, index=index)

def _nan01(s):
    """0-1にクリップ。NaNは0.5"""
    s = pd.to_numeric(s, errors="coerce")
    s = (s - s.min(skipna=True)) / (s.max(skipna=True) - s.min(skipna=True) + 1e-9)
    s = s.clip(0.0, 1.0)
    return s.fillna(0.5)

def _z01(z):
    """zスコアを0-1へ（tanh圧縮）。NaNは0.5"""
    z = pd.to_numeric(z, errors="coerce")
    return (0.5 + 0.5 * np.tanh(z / 3.0)).fillna(0.5)

def _roll_z(x, win):
    x = pd.to_numeric(x, errors="coerce")
    m = x.rolling(win, min_periods=max(3, win//3)).mean()
    s = x.rolling(win, min_periods=max(3, win//3)).std(ddof=0)
    return (x - m) / (s + 1e-9)

def _dir(x):
    """上昇:1, 下降:0"""
    return (x.diff() > 0).astype(float)

def _streak_up(x, cap=10):
    u = (x > 0).astype(int)
    run = []
    c = 0
    for v in u:
        c = c + 1 if v == 1 else 0
        run.append(min(c, cap))
    return pd.Series(run, index=x.index)

def _streak_down(x, cap=10):
    d = (x < 0).astype(int)
    run = []
    c = 0
    for v in d:
        c = c + 1 if v == 1 else 0
        run.append(min(c, cap))
    return pd.Series(run, index=x.index)

def _minute_of_day_sin_cos(idx_utc):
    try:
        jst = idx_utc.tz_convert("Asia/Tokyo")
    except Exception:
        jst = idx_utc
    minute = jst.hour * 60 + jst.minute
    ang = 2*np.pi*minute/(24*60)
    return np.sin(ang), np.cos(ang), minute

def _prev_session_close(df):
    """
    当日JSTの前営業日の最終closeをブロードキャスト
    前日が取れない時は直近60分の最後を代用
    """
    try:
        jst = df.index.tz_convert("Asia/Tokyo")
        d = pd.Series(df["close"].values, index=jst)
        dates = jst.date
        out = []
        cache = {}
        for i, dt in enumerate(d.index):
            day = dt.date()
            if day not in cache:
                # 前日の最終値
                prev = d.loc[str(pd.Timestamp(day, tz="Asia/Tokyo") - pd.Timedelta(days=1)).split()[0]]
                prev_close = prev.iloc[-1] if len(prev) else np.nan
                cache[day] = prev_close
            out.append(cache[day])
        s = pd.Series(out, index=df.index)
        # 代替
        s = s.fillna(df["close"].rolling(60, min_periods=1).apply(lambda x: x[-1], raw=True))
        return s
    except Exception:
        return pd.Series(np.nan, index=df.index)

# ========= Pack3 feature fns (25) =========

def f_open_gap(df, ctx):  # 1
    prev = _prev_session_close(df)
    gap = (df["open"] - prev) / (prev.replace(0, np.nan))
    return _z01(_roll_z(gap, 60))

def f_price_to_prev_close(df, ctx):  # 2
    prev = _prev_session_close(df)
    rel = (df["close"] - prev) / (prev.replace(0, np.nan))
    return _z01(_roll_z(rel, 60))

def f_or_high(df, ctx, n=5):
    try:
        # 当日JST 09:00〜の最初n分
        jst = df.index.tz_convert("Asia/Tokyo")
        first = (jst.hour == 9) & (jst.minute < (0+n))
        orh = df["high"].where(first).cummax().ffill()
        orh = orh.fillna(method="ffill")
        return orh
    except Exception:
        return pd.Series(np.nan, index=df.index)

def f_or_low(df, ctx, n=5):
    try:
        jst = df.index.tz_convert("Asia/Tokyo")
        first = (jst.hour == 9) & (jst.minute < (0+n))
        orl = df["low"].where(first).cummin().ffill()
        orl = orl.fillna(method="ffill")
        return orl
    except Exception:
        return pd.Series(np.nan, index=df.index)

def f_or_pos(df, ctx):  # 3 位置
    orh = f_or_high(df, ctx)
    orl = f_or_low(df, ctx)
    pos = (df["close"] - orl) / (orh - orl + 1e-9)
    return pos.clip(0, 1).fillna(0.5)

def f_or_break_up(df, ctx):  # 4
    orh = f_or_high(df, ctx)
    return (df["high"] > orh).astype(float)

def f_or_break_dn(df, ctx):  # 5
    orl = f_or_low(df, ctx)
    return (df["low"] < orl).astype(float)

def f_std5_over_20(df, ctx):  # 6
    s5 = df["close"].rolling(5).std(ddof=0)
    s20 = df["close"].rolling(20).std(ddof=0)
    r = (s5 / (s20 + 1e-9))
    return r.clip(0, 2).fillna(1.0).pipe(_nan01)

def f_atr_ratio_7_21(df, ctx):  # 7
    tr = (pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1))
    atr7 = tr.rolling(7).mean()
    atr21 = tr.rolling(21).mean()
    r = atr7 / (atr21 + 1e-9)
    return r.clip(0, 2).fillna(1.0).pipe(_nan01)

def f_bb_bw_pct(df, ctx):  # 8 ボリンジャーバンド幅パーセンタイル（20）
    m = df["close"].rolling(20).mean()
    s = df["close"].rolling(20).std(ddof=0)
    bw = (2*s*2) / (m + 1e-9)  # (UB-LB)/price ≈ 4σ/price
    # 過去60の順位
    pct = bw.rolling(60, min_periods=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) else np.nan, raw=False)
    return pct.fillna(0.5).clip(0,1)

def f_rsi_slope3(df, ctx):  # 9
    # 簡易RSI
    chg = df["close"].diff()
    up = chg.clip(lower=0).rolling(14).mean()
    dn = (-chg.clip(upper=0)).rolling(14).mean()
    rsi = 100 * up / (up + dn + 1e-9)
    slope = rsi.diff(3)
    return _z01(_roll_z(slope, 60))

def f_macd_hist_slope3(df, ctx):  # 10
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig
    slope = hist.diff(3)
    return _z01(_roll_z(slope, 60))

def f_dir_streak_up(df, ctx):  # 11
    d = _dir(df["close"])
    s = _streak_up(d, cap=8)
    return (s / 8.0).clip(0,1).fillna(0.5)

def f_dir_streak_dn(df, ctx):  # 12
    d = _dir(df["close"]).replace({1:0, 0:1})
    s = _streak_up(d, cap=8)
    return (s / 8.0).clip(0,1).fillna(0.5)

def f_up_ratio10(df, ctx):  # 13
    d = _dir(df["close"])
    r = d.rolling(10).mean()
    return r.clip(0,1).fillna(0.5)

def f_entropy10(df, ctx):  # 14
    d = _dir(df["close"])
    p = d.rolling(10).mean().clip(1e-4, 1-1e-4)
    ent = -(p*np.log2(p) + (1-p)*np.log2(1-p))  # 0〜1bit
    return (ent / 1.0).fillna(0.5)

def f_ret_skew20(df, ctx):  # 15
    r = df["close"].pct_change()
    skew = r.rolling(20).skew()
    return _z01(skew)

def f_ret_kurt20(df, ctx):  # 16
    r = df["close"].pct_change()
    kurt = r.rolling(20).kurt()
    return _z01(kurt)

def f_vol_regime(df, ctx):  # 17 短長ボラ比
    s30 = df["close"].pct_change().rolling(30).std(ddof=0)
    s120 = df["close"].pct_change().rolling(120).std(ddof=0)
    reg = s30 / (s120 + 1e-9)
    return reg.clip(0,2).pipe(_nan01)

def f_fx_corr30(df, ctx):  # 18
    if "fx_close" not in df.columns:
        return pd.Series(0.5, index=df.index)
    a = df["close"].pct_change()
    b = df["fx_close"].pct_change()
    corr = a.rolling(30).corr(b)
    return _z01(corr)

def f_fx_beta30(df, ctx):  # 19
    if "fx_close" not in df.columns:
        return pd.Series(0.5, index=df.index)
    a = df["close"].pct_change()
    b = df["fx_close"].pct_change().replace(0, np.nan)
    cov = a.rolling(30).cov(b)
    var = b.rolling(30).var()
    beta = cov / (var + 1e-9)
    return _z01(beta)

def f_aroon_up14(df, ctx):  # 20
    HH = df["high"].rolling(14, min_periods=3).apply(lambda x: len(x)-1-np.argmax(x), raw=True)
    up = 1 - (HH / 13.0)
    return up.clip(0,1).fillna(0.5)

def f_aroon_dn14(df, ctx):  # 21
    LL = df["low"].rolling(14, min_periods=3).apply(lambda x: len(x)-1-np.argmin(x), raw=True)
    dn = 1 - (LL / 13.0)
    return dn.clip(0,1).fillna(0.5)

def f_mod_slope5(df, ctx):  # 22 単純傾き（5本）
    x = np.arange(len(df))
    y = df["close"].values
    win = 5
    out = np.full(len(df), np.nan)
    for i in range(win, len(df)):
        xx = x[i-win+1:i+1]
        yy = y[i-win+1:i+1]
        k = np.polyfit(xx, yy, 1)[0]
        out[i] = k
    return _z01(pd.Series(out, index=df.index))

def f_mins_since_open_norm(df, ctx):  # 23 09:00からの経過分（0〜1）
    try:
        jst = df.index.tz_convert("Asia/Tokyo")
        mins = (jst.hour*60 + jst.minute) - 9*60
        mins = mins.clip(lower=0)
        return (mins / 60.0).clip(0,1).fillna(0.0)
    except Exception:
        return pd.Series(0.0, index=df.index)

def f_mod_sin(df, ctx):  # 24 一日のsin（連続周期特徴）
    s, c, _ = _minute_of_day_sin_cos(df.index)
    return _nan01(pd.Series(s, index=df.index))

def f_mod_cos(df, ctx):  # 25
    s, c, _ = _minute_of_day_sin_cos(df.index)
    return _nan01(pd.Series(c, index=df.index))


# ========= Pack3 spec =========
@dataclass
class FeatureSpec:
    name: str
    fn: Callable
    category: str
    bounds: Tuple[float, float] = (0.0, 1.0)

PACK3 = [
    FeatureSpec("open_gap",               f_open_gap,            "gap"),
    FeatureSpec("price_to_prev_close",   f_price_to_prev_close, "gap"),
    FeatureSpec("or_pos",                 f_or_pos,              "or"),
    FeatureSpec("or_break_up",            f_or_break_up,         "or"),
    FeatureSpec("or_break_dn",            f_or_break_dn,         "or"),
    FeatureSpec("std5_over_20",           f_std5_over_20,        "vol"),
    FeatureSpec("atr_ratio_7_21",         f_atr_ratio_7_21,      "vol"),
    FeatureSpec("bb_bw_pct",              f_bb_bw_pct,           "vol"),
    FeatureSpec("rsi_slope3",             f_rsi_slope3,          "mom"),
    FeatureSpec("macd_hist_slope3",       f_macd_hist_slope3,    "mom"),
    FeatureSpec("dir_streak_up",          f_dir_streak_up,       "mom"),
    FeatureSpec("dir_streak_dn",          f_dir_streak_dn,       "mom"),
    FeatureSpec("up_ratio10",             f_up_ratio10,          "mom"),
    FeatureSpec("entropy10",              f_entropy10,           "mom"),
    FeatureSpec("ret_skew20",             f_ret_skew20,          "dist"),
    FeatureSpec("ret_kurt20",             f_ret_kurt20,          "dist"),
    FeatureSpec("vol_regime",             f_vol_regime,          "regime"),
    FeatureSpec("fx_corr30",              f_fx_corr30,           "fx"),
    FeatureSpec("fx_beta30",              f_fx_beta30,           "fx"),
    FeatureSpec("aroon_up14",             f_aroon_up14,          "trend"),
    FeatureSpec("aroon_dn14",             f_aroon_dn14,          "trend"),
    FeatureSpec("mod_slope5",             f_mod_slope5,          "trend"),
    FeatureSpec("mins_since_open",        f_mins_since_open_norm,"time"),
    FeatureSpec("mod_sin",                f_mod_sin,             "time"),
    FeatureSpec("mod_cos",                f_mod_cos,             "time"),
]

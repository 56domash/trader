from __future__ import annotations
import numpy as np, pandas as pd
from .utils import clip01, z_to_01, safe_div, to5m, anchored_vwap, robust_z

def _last(series: pd.Series, k: int = 1):
    """末尾k番目を安全に返す。足りなければ np.nan を返す。"""
    if series is None or len(series) < k:
        return np.nan
    return series.iloc[-k]

def session(df, start='09:00', end='15:00'): return df.between_time(start, end)
def within_9_10(df): return df.between_time('09:00','10:00')

def compute_feature_snapshot(df_day: pd.DataFrame, now_idx: pd.Timestamp, ext: dict):
    """
    Parameters
    ----------
    df_day : 1分足の当日DataFrame（列: open, high, low, close, volume / tz-aware index）
    now_idx: 現在時刻（tz-aware Timestamp, JSTを想定）
    ext    : 追加系列辞書 {"r_nk": Series(リターン), "r_auto": Series(リターン), "usdjpy": Series(レート)}

    Returns
    -------
    feat : dict   # 100特徴（b1_1..b5_10, s1_1..s5_10）
    Buy  : list   # 5式（各10特徴の平均）買い側 [buy1..buy5]
    Sell : list   # 5式（各10特徴の平均）売り側 [sell1..sell5]
    S    : float  # 総合スコア (= sum(Buy) - sum(Sell))
    meta : dict   # ログ用（vwap, atr5, ORH, ORL, close, rvol5）
    """
    # フォーカス（9:00-10:00の1分足）。最小本数のウォームアップを要求（例: 6本）
    focus = df_day.loc[:now_idx].between_time('09:00','10:00')
    if focus is None or focus.empty or len(focus) < 6:
        raise RuntimeError("warming up (need >=6 bars in 09:00-10:00)")

    # 代表値
    close = float(focus['close'].iloc[-1])
    high  = float(focus['high'].iloc[-1])
    low   = float(focus['low'].iloc[-1])
    vol   = float(focus['volume'].iloc[-1])
    open0 = float(focus['open'].iloc[0])
    ORH   = float(focus['high'].iloc[:min(len(focus), 30)].max())   # 9:00~9:30高値（例）
    ORL   = float(focus['low'].iloc[:min(len(focus), 30)].min())    # 9:00~9:30安値（例）

    # 5分足集計（必要に応じて使用）
    agg5 = to5m(focus)  # open/high/low/close/volume を5分に集約
    atr5 = (agg5['high'] - agg5['low']).rolling(5, min_periods=3).mean()
    atr_now = float(atr5.iloc[-1]) if len(atr5) else 0.0

    # VWAP（日中アンカー）
    vwap_day = anchored_vwap(session(df_day))
    # VWAP を focus の index に整列（← reindex('ffill') の誤用を修正）
    vwap_aligned = vwap_day.reindex(focus.index, method='ffill')

    prev_close = _last(focus['close'], 2)
    prev_vwap  = _last(vwap_aligned, 2)
    vwap_now   = _last(vwap_aligned, 1)

    # クロス判定（前バーの位置関係 + 現在の位置関係）
    vw_cross_up = (
        pd.notna(prev_close) and pd.notna(prev_vwap) and prev_close <= prev_vwap
    ) and (
        pd.notna(close) and pd.notna(vwap_now) and close > vwap_now
    )
    vw_cross_dn = (
        pd.notna(prev_close) and pd.notna(prev_vwap) and prev_close >= prev_vwap
    ) and (
        pd.notna(close) and pd.notna(vwap_now) and close < vwap_now
    )

    def recent_share(cond, n=5):
        s = pd.Series(cond, index=focus.index).tail(n)
        return float(np.mean(s.astype(float))) if len(s) > 0 else 0.5

    def bb_pos_last():
        ma20  = focus['close'].rolling(20, min_periods=5).mean()
        std20 = focus['close'].rolling(20, min_periods=5).std()
        posb  = (focus['close'] - ma20) / (2 * std20.replace(0, np.nan))
        v = posb.iloc[-1] if np.isfinite(posb.iloc[-1]) else 0.0
        return float(clip01(v * 0.5 + 0.5))

    def up_down_vol(n=5):
        delta = focus['close'].diff()
        upv = focus['volume'].where(delta >= 0, 0.0).tail(n).sum()
        dnv = focus['volume'].where(delta <  0, 0.0).tail(n).sum()
        tot = upv + dnv
        return (safe_div(upv, tot), safe_div(dnv, tot))

    # 追加系列（存在すれば使用）
    r_nk   = ext.get("r_nk",   None)
    r_auto = ext.get("r_auto", None)
    usdjpy = ext.get("usdjpy", None)

    # 例：ボラ・出来高などの基礎特徴
    rvol5 = float(safe_div(focus['volume'].tail(5).mean(), focus['volume'].rolling(20, min_periods=5).mean().iloc[-1]))
    hh = np.sum(focus['high'].diff().fillna(0) > 0)
    ll = np.sum(focus['low'].diff().fillna(0)  < 0)

    # ここから各ブロック（10特徴×5群＝50）×2=100
    # ------ Buy 群（例） ------
    # 1群：モメンタム/ボラ/バンド/クロス/プルバック 等
    tanh = np.tanh
    macd = focus['close'].ewm(span=12, adjust=False).mean() - focus['close'].ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_os = macd.iloc[-1] - macd_signal.iloc[-1] if len(macd) >= 2 else 0.0

    # open-range 超え比率（直近5本でORH上にいた割合など）
    above_orh_share5 = recent_share(focus['close'] > ORH, n=5)
    pullback_depth_norm = clip01(safe_div((ORH - close), max(1e-9, ORH - ORL))) if ORH > ORL else 0.0

    f11 = clip01(robust_z(focus['close'].pct_change().tail(5).mean()))
    f12 = clip01(robust_z((high - low) / max(1e-9, low)))
    f13 = clip01(robust_z(macd_os))
    f14 = clip01(robust_z(rvol5))
    f15 = clip01(((hh - ll) / 5 + 1) / 2)
    f16 = bb_pos_last()
    f17 = clip01((tanh(macd_os) + 1) / 2)
    f18 = clip01(above_orh_share5)
    f19 = 0.7 if vw_cross_up else 0.3
    f110 = 1.0 - clip01(pullback_depth_norm)

    up_share, dn_share = up_down_vol(5)
    f21 = clip01(rvol5)
    f22 = clip01(up_share)
    f23 = 0.8  # プレースホルダ（調整余地）
    f24 = 0.5
    f25 = 0.5
    f26 = 0.7 if (len(focus) >= 2 and focus['close'].iloc[-1] > focus['close'].iloc[-2]) else (0.3 if len(focus) >= 2 else 0.5)
    f27 = clip01(robust_z(focus['close'].pct_change().rolling(3).mean().iloc[-1]))
    f28 = clip01(robust_z((close - ORL) / max(1e-9, ORH - ORL))) if ORH > ORL else 0.5
    f29 = clip01(robust_z(safe_div(close, vwap_now) - 1.0)) if pd.notna(vwap_now) else 0.5
    f210= clip01(robust_z(atr_now)) if np.isfinite(atr_now) else 0.5

    # 以降も同様に特徴を定義（例では簡略化）
    # 群2
    f31=f32=f33=f34=f35=f36=f37=f38=f39=f310 = 0.5
    # 群3
    f41=f42=f43=f44=f45=f46=f47=f48=f49=f410 = 0.5
    # 群4
    f51=f52=f53=f54=f55=f56=f57=f58=f59=f510 = 0.5
    # 群5
    f61=f62=f63=f64=f65=f66=f67=f68=f69=f610 = 0.5

    # ------ Sell 群（例） ------
    # VWAPダウンクロスやオーバーシュートなど
    g11 = 0.7 if vw_cross_dn else 0.3
    g12 = 1.0 - f16
    g13 = clip01(robust_z(-macd_os))
    g14 = clip01(1.0 - up_share)
    g15 = clip01(robust_z((ORH - close) / max(1e-9, ORH - ORL))) if ORH > ORL else 0.5
    g16 = clip01(robust_z(-rvol5))
    g17 = clip01(robust_z(safe_div(vwap_now, close) - 1.0)) if pd.notna(vwap_now) else 0.5
    g18 = 0.5
    g19 = 0.5
    g110= 0.5

    # 以下、群2-群5は簡略化プレースホルダ
    g21=g22=g23=g24=g25=g26=g27=g28=g29=g210 = 0.5
    g31=g32=g33=g34=g35=g36=g37=g38=g39=g310 = 0.5
    g41=g42=g43=g44=g45=g46=g47=g48=g49=g410 = 0.5
    g51=g52=g53=g54=g55=g56=g57=g58=g59=g510 = 0.5

    # 5群×各10特徴→各群の平均 = 5式
    B  = [[f11,f12,f13,f14,f15,f16,f17,f18,f19,f110],
          [f21,f22,f23,f24,f25,f26,f27,f28,f29,f210],
          [f31,f32,f33,f34,f35,f36,f37,f38,f39,f310],
          [f41,f42,f43,f44,f45,f46,f47,f48,f49,f410],
          [f51,f52,f53,f54,f55,f56,f57,f58,f59,f510]]
    Sg = [[g11,g12,g13,g14,g15,g16,g17,g18,g19,g110],
          [g21,g22,g23,g24,g25,g26,g27,g28,g29,g210],
          [g31,g32,g33,g34,g35,g36,g37,g38,g39,g310],
          [g41,g42,g43,g44,g45,g46,g47,g48,g49,g410],
          [g51,g52,g53,g54,g55,g56,g57,g58,g59,g510]]

    Buy  = [float(np.mean(gp)) for gp in B]
    Sell = [float(np.mean(gp)) for gp in Sg]
    S_total = sum(Buy) - sum(Sell)

    # 100特徴を書き出し用にフラット化
    feat = {}
    for k in range(5):
        for i in range(10):
            feat[f"b{k+1}_{i+1}"] = float(B[k][i])
            feat[f"s{k+1}_{i+1}"] = float(Sg[k][i])

    meta = dict(
        vwap=float(vwap_now) if pd.notna(vwap_now) else float('nan'),
        atr5=float(atr_now),
        ORH=float(ORH),
        ORL=float(ORL),
        close=float(close),
        rvol5=float(rvol5),
    )
    return feat, Buy, Sell, S_total, meta

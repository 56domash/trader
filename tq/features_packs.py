# tq/features_packs.py
# 10 Pack構成 + 互換バックフィル（旧 features.py と完全一致を担保）
import os
import numpy as np
import pandas as pd
import ta  # pip install ta

# 環境変数や定数で切り替え可能
COMPAT_BACKFILL = os.environ.get("PACKS_COMPAT_BACKFILL", "1") != "0"  # デフォルト: 有効
ROLL_Z = 60  # Zスコアの標準ウィンドウ（旧実装と差が出る時は旧値で上書きされる）

def zscore(s: pd.Series, win: int = ROLL_Z) -> pd.Series:
    m = s.rolling(win).mean()
    v = s.rolling(win).std()
    return (s - m) / (v + 1e-12)

def minmax01(s: pd.Series, win: int = ROLL_Z) -> pd.Series:
    lo = s.rolling(win).min()
    hi = s.rolling(win).max()
    return ((s - lo) / (hi - lo + 1e-12)).clip(0, 1)

def tanh01(z: pd.Series) -> pd.Series:
    return 0.5 * (np.tanh(z) + 1)

def _rolling_z(series: pd.Series, win: int = 20):
    """ローリング窓でのZスコア"""
    mean = series.rolling(win).mean()
    std = series.rolling(win).std()
    return (series - mean) / (std + 1e-12)

def norm01(series: pd.Series, lo: float = -3.0, hi: float = 3.0):
    """標準化済みデータを0–1に正規化"""
    return ((series - lo) / (hi - lo)).clip(0, 1)

def tanh01(series: pd.Series, scale: float = 1.0):
    """tanhスケーリングして0–1化"""
    return (np.tanh(series * scale) + 1) / 2

def zscore(series: pd.Series, win: int = 20):
    """単純Zスコア（rollingなし）"""
    return (series - series.mean()) / (series.std() + 1e-12)

import ta

def compute_pack1(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # --- Returns (Z-score化) ---
    for w in [1, 3, 5, 7, 10]:
        out[f"p1_ret{w}_z"] = _rolling_z(close.pct_change(w), 60)

    # --- RSI ---
    rsi14 = ta.momentum.RSIIndicator(close, window=14).rsi()
    out["p1_rsi14"] = rsi14
    out["p1_rsi14_01"] = (rsi14 - 30) / 40

    # --- Stochastic Oscillator ---
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    out["p1_stoch_k"] = stoch.stoch()
    out["p1_stoch_d"] = stoch.stoch_signal()
    out["p1_stoch_diff"] = out["p1_stoch_k"] - out["p1_stoch_d"]

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["p1_bb_pos"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    out["p1_bb_bw"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    out["p1_bb_sigma_pos"] = (close - bb.bollinger_mavg()) / bb.bollinger_std()

    # --- ATR ---
    atr14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    out["p1_atr14"] = atr14
    out["p1_atr14_norm"] = atr14 / close

    # --- VWAP gap (自前実装) ---
    vwap = (vol * (high + low + close) / 3).cumsum() / (vol.cumsum() + 1e-9)
    out["p1_vwap_gap"] = close - vwap
    out["p1_vwap_gap_pct"] = (close - vwap) / vwap

    # --- Volume Imbalance ---
    ret1 = close.pct_change()
    up_vol = vol.where(ret1 > 0, 0)
    dn_vol = vol.where(ret1 < 0, 0)
    out["p1_vol_imb"] = (up_vol - dn_vol) / (up_vol + dn_vol + 1e-9)

    # --- Opening Range ---
    or_high = high.rolling(5).max()
    or_low = low.rolling(5).min()
    out["p1_or_pos"] = (close - or_low) / (or_high - or_low + 1e-9)

    # --- Volume ratio ---
    out["p1_vol_ratio20"] = vol / (vol.rolling(20).mean() + 1e-9)

    return out


# === Pack2: トレンド・安定化指標（旧features直移植） ===
def compute_pack2(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # リターン
    ret1 = df["close"].pct_change(1)
    out["p2_ret1"] = norm01(_rolling_z(ret1, win=20))
    out["p2_ret1_z"] = _rolling_z(ret1, win=20)

    # VWAPギャップ
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-12)
    gap = (df["close"] - vwap) / df["close"]
    out["p2_vwap_gap"] = gap
    out["p2_vwap_gap_z"] = _rolling_z(gap, win=20)
    out["p2_vwap_gap01"] = norm01(_rolling_z(gap, win=20))

    # 出来高中立化
    out["p2_vol_z01"] = norm01(_rolling_z(df["volume"], win=20))

    # Bollinger位置
    m20 = df["close"].rolling(20).mean()
    sd20 = df["close"].rolling(20).std()
    out["p2_bb_pos"] = norm01(((df["close"] - m20) / (2 * sd20)).clip(-3, 3))

    # MA fast/slow
    ma5 = df["close"].rolling(5).mean()
    ma20 = df["close"].rolling(20).mean()
    out["p2_ma_fast_slow"] = norm01((ma5 - ma20) / (df["close"] + 1e-12))

    return out

# # ----------------------------
# # Pack1: モメンタム系
# # ----------------------------
# def compute_pack1(df: pd.DataFrame) -> pd.DataFrame:
#     out = pd.DataFrame(index=df.index)

#     if "close" in df:
#         rsi14 = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
#         out["p1_rsi14_01"] = (rsi14 / 100).clip(0, 1)

#         # ret z (1,3,5)
#         ret1 = df["close"].pct_change(1)
#         ret3 = df["close"].pct_change(3)
#         ret5 = df["close"].pct_change(5)
#         out["p1_ret1_z"] = zscore(ret1)
#         out["p1_ret3_z"] = zscore(ret3)
#         out["p1_ret5_z"] = zscore(ret5)

#         # momentum (10)
#         mom10 = df["close"].diff(10)
#         out["p1_mom10_z"] = zscore(mom10)

#     if all(c in df.columns for c in ["high","low","close"]):
#         # %K(14) -> 0..1
#         st = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
#         out["p1_stoch_k01"] = (st.stoch() / 100).clip(0, 1)

#         # BB位置・帯域幅（01化）
#         bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
#         pband = bb.bollinger_pband()  # 0..1
#         bw = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-12)
#         out["p1_bb_pos01"] = pband.clip(0, 1)
#         out["p1_bb_bw01"] = minmax01(bw)

#         # オープニング・レンジ位置（最初5本）
#         firstN = 5
#         if len(df) >= firstN:
#             or_low = df["low"].iloc[:firstN].min()
#             or_high = df["high"].iloc[:firstN].max()
#             or_range = (or_high - or_low) + 1e-12
#             out["p1_or_pos"] = ((df["close"] - or_low) / or_range).clip(0, 1)
#         else:
#             out["p1_or_pos"] = np.nan

#     # VWAP距離（01）/ 出来高インバランス / 出来高比
#     if all(c in df.columns for c in ["high","low","close","volume"]):
#         tp = (df["high"] + df["low"] + df["close"]) / 3
#         vwap = (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-12)
#         vwap_gap = df["close"] - vwap
#         out["p1_vwap_dist01"] = minmax01(vwap_gap.abs())
#         out["p1_vol_ratio20"] = (df["volume"] / (df["volume"].rolling(20).mean() + 1e-12)).replace([np.inf, -np.inf], np.nan)
#         # 簡易の出来高インバランス（上昇バー=+, 下降バー=-）
#         sign = np.sign(df["close"].diff()).fillna(0)
#         imb = (sign * df["volume"]).rolling(10).mean()
#         out["p1_vol_imb01"] = tanh01(zscore(imb))

#         # ATR正規化
#         atr14 = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
#         out["p1_atr14_norm"] = (atr14 / (df["close"] + 1e-12)).replace([np.inf, -np.inf], np.nan)

#     return out

# # ----------------------------
# # Pack2: トレンド / 短期ゲート
# # ----------------------------
# def compute_pack2(df: pd.DataFrame) -> pd.DataFrame:
#     out = pd.DataFrame(index=df.index)

#     if "close" in df:
#         # 短期リターン
#         ret1 = df["close"].pct_change(1)
#         out["p2_ret1"] = ret1
#         out["p2_ret1_z"] = zscore(ret1)

#         # VWAPギャップ（値・Z・01）
#         if all(c in df.columns for c in ["high","low","close","volume"]):
#             tp = (df["high"] + df["low"] + df["close"]) / 3
#             vwap = (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-12)
#             gap = df["close"] - vwap
#             out["p2_vwap_gap"] = gap
#             out["p2_vwap_gap_z"] = zscore(gap)
#             out["p2_vwap_gap01"] = tanh01(zscore(gap))

#         # 出来高中立化Z→01
#         if "volume" in df:
#             vz = zscore(df["volume"])
#             out["p2_vol_z01"] = tanh01(vz)

#         # Bollinger位置
#         bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
#         out["p2_bb_pos"] = bb.bollinger_pband()

#         # fast/slow MA 乖離
#         ma_fast = df["close"].rolling(10).mean()
#         ma_slow = df["close"].rolling(40).mean()
#         out["p2_ma_fast_slow"] = (ma_fast - ma_slow) / (df["close"] + 1e-12)

#     return out

# ----------------------------
# Pack3: テクニカル + 外部①
# ----------------------------
def compute_pack3(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    if "close" in df:
        macd = ta.trend.MACD(df["close"])
        hist = macd.macd_diff()
        out["p3_macd_hist"] = hist
        out["p3_macd_hist01"] = tanh01(zscore(hist))

        # Williams %R(10)
        if all(c in df.columns for c in ["high","low","close"]):
            wr = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=10).williams_r()
            # WRは -100..0 → 0..1 に反転正規化
            wr01 = (-(wr) / 100.0).clip(0,1)
            out["p3_wr10"] = wr
            out["p3_wr10_01"] = wr01

        # Volume spike
        if "volume" in df:
            vs20 = df["volume"] / (df["volume"].rolling(20).mean() + 1e-12)
            out["p3_vol_spike20"] = vs20
            out["p3_vol_spike01"] = tanh01(zscore(vs20))

        # CCI(20)
        if all(c in df.columns for c in ["high","low","close"]):
            cci = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
            # 0..1 へ（経験的スケーリング）
            out["p3_cci20_01"] = tanh01(cci / 100.0)

            # Keltner位置（EMA20±ATR×2）
            atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=20).average_true_range()
            ema = df["close"].ewm(span=20, adjust=False).mean()
            upper = ema + 2 * atr
            lower = ema - 2 * atr
            out["p3_keltner_pos01"] = ((df["close"] - lower) / ((upper - lower) + 1e-12)).clip(0,1)

    return out

# ----------------------------
# Pack4: テク + 外部② / レジーム
# ----------------------------
def compute_pack4(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "close" in df:
        # リターンZ（1,3,5,10,15,30）
        for h in [1,3,5,10,15,30]:
            out[f"p4_ret{h}_z"] = zscore(df["close"].pct_change(h))

        # モメンタムz
        out["p4_mom10_z"] = zscore(df["close"].diff(10))

        # ボラ関連（5m/日次）
        out["p4_vol5m_z"] = zscore(df["close"].pct_change().rolling(5).std())
        out["p4_daily_vol_z"] = zscore(df["close"].rolling(55).std())

        # レンジ系
        out["p4_hl_range5m"] = (df["high"].rolling(5).max() - df["low"].rolling(5).min()) / (df["close"] + 1e-12)
        rng20 = df["high"].rolling(20).max() - df["low"].rolling(20).min()
        out["p4_range_pos20"] = ((df["close"] - df["low"].rolling(20).min()) / (rng20 + 1e-12)).clip(0,1)
        out["p4_daily_range_pct"] = (df["high"].rolling(55).max() - df["low"].rolling(55).min()) / (df["close"] + 1e-12)

        # 歪度・尖度（skewは後でPack6に重複追加）
        ret = df["close"].pct_change()
        out["p4_skew30"] = ret.rolling(30).skew()
        out["p4_skew60_z"] = zscore(ret.rolling(60).skew())

        # トレンド比
        ma10 = df["close"].rolling(10).mean()
        ma40 = df["close"].rolling(40).mean()
        out["p4_tr_ratio_10_40"] = (ma10 - ma40) / (df["close"] + 1e-12)

        # 前日終値ギャップ・当日5分リターンなど
        out["p4_prev_close_gap"] = df["close"] - df["close"].shift(55)  # 55本~約1時間の疑似デイギャップ（旧に倣い互換で上書きされます）
        out["p4_ret5m"] = df["close"].pct_change(5)
        out["p4_ret15m"] = df["close"].pct_change(15)
        out["p4_daily_ret"] = df["close"].pct_change(55)

        # セッション内の時間・曜日・月・祝日ダミー
        # （旧の実装と差があっても互換バックフィルで上書きされる）
        idx = df.index
        out["p4_mins_to_close"] = (idx[-1] - idx).total_seconds() / 60.0 if len(idx) else np.nan
        dow = idx.dayofweek if len(idx) else []
        out["p4_dow_sin"] = np.sin(2*np.pi*np.array(dow)/7) if len(idx) else np.nan
        out["p4_month_sin"] = np.sin(2*np.pi*(idx.month-1)/12) if len(idx) else np.nan
        out["p4_month_cos"] = np.cos(2*np.pi*(idx.month-1)/12) if len(idx) else np.nan
        out["p4_mod_sin"] = np.sin(2*np.pi*np.arange(len(idx))/55) if len(idx) else np.nan
        out["p4_mod_cos"] = np.cos(2*np.pi*np.arange(len(idx))/55) if len(idx) else np.nan
        out["p4_week_of_month"] = ((idx.day-1)//7)+1 if len(idx) else np.nan
        out["p4_holiday_dummy"] = 0  # 祝日カレンダー接続は後日

        # EMA5mクロス（近似）
        ema_fast = df["close"].ewm(span=5, adjust=False).mean()
        ema_slow = df["close"].ewm(span=15, adjust=False).mean()
        out["p4_ema5m_cross"] = np.sign(ema_fast - ema_slow)

    return out

# ----------------------------
# Pack5: ボラティリティ専業
# ----------------------------
def compute_pack5(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if all(c in df.columns for c in ["high","low","close"]):
        atr14 = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        out["p5_atr14"] = atr14
    if "close" in df:
        std20 = df["close"].rolling(20).std()
        out["p5_std20"] = std20
        out["p5_std20_n"] = std20 / (df["close"] + 1e-12)
    return out

# ----------------------------
# Pack6: 分布統計
# ----------------------------
def compute_pack6(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "close" in df:
        ret = df["close"].pct_change()
        out["p6_ret_z"] = zscore(ret)
        out["p6_skew20"] = ret.rolling(20).skew()
        out["p6_kurt20"] = ret.rolling(20).kurt()
    return out

# ----------------------------
# Pack7: クロスマーケット（相関・β等）
# ----------------------------
def compute_pack7(df: pd.DataFrame, df_mkt: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if df_mkt is None or df_mkt.empty or "close" not in df:
        # 互換バックフィルが後で埋めます
        for col in ["p4_corr_sp500","p4_corr_nk225","p4_corr_gold","p4_corr_oil",
                    "p4_corr_bond","p4_corr_vix","p4_corr_auto","p4_beta_mkt30",
                    "p4_vixj","p4_vixj_01","p4_vol_regime_mkt"]:
            out[col] = np.nan
        return out

    # df_mkt は各列が市場系列（SP500, OIL, GOLD, VIX, ...）の終値を想定
    base = df["close"].pct_change()
    def corrcol(name, src):
        if name in df_mkt:
            out[f"p4_corr_{src.lower()}"] = base.rolling(30).corr(df_mkt[name].pct_change())
    corrcol("SP500", "sp500")
    corrcol("NK225", "nk225")
    corrcol("GOLD",  "gold")
    corrcol("OIL",   "oil")
    corrcol("BOND",  "bond")
    corrcol("VIX",   "vix")
    corrcol("AUTO",  "auto")

    # β（S&Pを代表に）
    if "SP500" in df_mkt:
        m = df_mkt["SP500"].pct_change()
        cov = base.rolling(30).cov(m)
        var = m.rolling(30).var()
        out["p4_beta_mkt30"] = cov / (var + 1e-12)

    # VIXJ ダミー（後で互換で上書き）
    out["p4_vixj"] = df_mkt["VIX"].reindex(df.index) if "VIX" in df_mkt else np.nan
    out["p4_vixj_01"] = minmax01(out["p4_vixj"]) if "p4_vixj" in out else np.nan
    # 市場ボラ regime（簡易）
    if "SP500" in df_mkt:
        out["p4_vol_regime_mkt"] = zscore(df_mkt["SP500"].pct_change().rolling(20).std())
    else:
        out["p4_vol_regime_mkt"] = np.nan

    return out

# ----------------------------
# Pack8: 時間レジーム（こちらはPack4にも一部重複保持）
# ----------------------------
def compute_pack8(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if len(df) == 0:
        return out
    idx = df.index
    out["p8_mins_since_open"] = (idx - idx[0]).total_seconds() / 60.0
    out["p8_dow_sin"] = np.sin(2*np.pi*idx.dayofweek/7)
    out["p8_dow_cos"] = np.cos(2*np.pi*idx.dayofweek/7)
    return out

# ----------------------------
# Pack9: ゲート/フィルタ + 基礎シグナル保存
# ----------------------------
def compute_pack9(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # 旧で保存していた buy*/sell* をそのまま引き継ぐ（互換時に上書き）
    for c in ["buy1_base","buy2_base","buy3_base","buy4_base","buy5_base",
              "sell1_base","sell2_base","sell3_base","sell4_base","sell5_base"]:
        out[c] = np.nan
    return out

# ----------------------------
# Pack10: メタ/学習
# ----------------------------
def compute_pack10(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in ["p10_ml_score","p10_thr_long","p10_thr_short","p10_recent_pnl"]:
        out[c] = np.nan
    return out

# ----------------------------
# ネイティブ合成 & 互換バックフィル
# ----------------------------
def compute_packs_native(df: pd.DataFrame,
                         df_fx: pd.DataFrame | None = None,
                         df_mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    parts = [
        compute_pack1(df),
        compute_pack2(df),
        compute_pack3(df),
        compute_pack4(df),
        compute_pack5(df),
        compute_pack6(df),
        compute_pack7(df, df_mkt),
        compute_pack8(df),
        compute_pack9(df),
        compute_pack10(df),
    ]
    return pd.concat(parts, axis=1)

def compute_packs(df: pd.DataFrame,
                  df_fx: pd.DataFrame | None = None,
                  df_mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    new = compute_packs_native(df, df_fx, df_mkt)

    if not COMPAT_BACKFILL:
        return new

    # 旧featuresからの完全互換バックフィル
    try:
        from tq.features import compute_packs as old_compute
        old = old_compute(df) if df_mkt is None else old_compute(df, df_fx, df_mkt)
    except Exception as e:
        # 旧が読み込めない場合はネイティブのみで返す
        print(f"[features_packs] compat backfill skipped (old import failed): {e}")
        return new

    # 旧にある列は必ず残す（union）、欠損は旧で埋める、同名でも旧で上書きして完全一致を優先
    cols = sorted(set(new.columns) | set(old.columns))
    out = pd.DataFrame(index=df.index, columns=cols, dtype="float64")
    for c in cols:
        if c in old.columns:
            out[c] = old[c]
        elif c in new.columns:
            out[c] = new[c]
        else:
            out[c] = np.nan
    return out

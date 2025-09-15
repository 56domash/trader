from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
# from ta.volume import OnBalanceVolumeIndicator
# from ta.trend import SMAIndicator, EMAIndicator, MACD
import ta
import os
import numpy as np
import pandas as pd
from typing import Optional
import os


# 環境変数や定数で切り替え可能
COMPAT_BACKFILL = os.environ.get(
    "PACKS_COMPAT_BACKFILL", "1") != "0"  # デフォルト: 有効
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


def _rolling_z(series: pd.Series, window: int = 20) -> pd.Series:
    """ローリング zスコア"""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std


def norm01(series: pd.Series, lo: float = -3.0, hi: float = 3.0):
    """標準化済みデータを0–1に正規化"""
    return ((series - lo) / (hi - lo)).clip(0, 1)


def tanh01(series: pd.Series, scale: float = 1.0):
    """tanhスケーリングして0–1化"""
    return (np.tanh(series * scale) + 1) / 2


def zscore(series: pd.Series, win: int = 20):
    """単純Zスコア（rollingなし）"""
    return (series - series.mean()) / (series.std() + 1e-12)


def _ewm_zscore(series: pd.Series, span: int = 20) -> pd.Series:
    """
    EWMA (指数加重移動平均) に基づく zスコアを計算
    series: 入力データ
    span: 平滑化スパン
    """
    mean = series.ewm(span=span, min_periods=span).mean()
    std = series.ewm(span=span, min_periods=span).std()
    z = (series - mean) / std
    return z.fillna(0)

# ---------- 小さなヘルパー（安全に何度定義してもOK） ----------


def _drop_dupe_index(df_or_s):
    """インデックスの重複を後勝ちで落とす（相関計算前の安全策）"""
    obj = df_or_s.copy()
    idx = obj.index
    if idx.has_duplicates:
        obj = obj[~idx.duplicated(keep="last")]
    return obj


def _align_series(s: pd.Series, index: pd.Index, method="ffill"):
    """外部シリーズを自シンボルのindexに合わせて揃える"""
    s2 = _drop_dupe_index(s)
    return s2.reindex(index).fillna(method=method)

# def _ewm_zscore(x: pd.Series, span=20):
#     mu = x.ewm(span=span, adjust=False).mean()
#     sd = x.ewm(span=span, adjust=False).std().replace(0, np.nan)
#     return (x - mu) / sd


def _rolling_beta(y: pd.Series, x: pd.Series, window=60):
    """beta = cov(y,x)/var(x) のローリング版"""
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / (var + 1e-12)


def _realized_vol(ret: pd.Series, window=20):
    return ret.rolling(window).std()


def _downside_vol(ret: pd.Series, window=20):
    neg = ret.clip(upper=0)
    return (neg.pow(2).rolling(window).mean().pow(0.5))


def _autocorr(ret: pd.Series, lag=1, window=60):
    # 簡易自己相関（ラグ付き相関）
    return ret.rolling(window).corr(ret.shift(lag))


def _rolling_max_drawdown(close: pd.Series, window=60):
    roll_max = close.rolling(window).max()
    dd = close / roll_max - 1.0
    return dd


def compute_pack1(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # --- Returns (Z-score化) ---
    for w in [1, 3, 5]:
        out[f"p1_ret{w}_z"] = _rolling_z(close.pct_change(w), 60)

    # --- RSI (0–1正規化だけ残す) ---
    rsi14 = ta.momentum.RSIIndicator(close, window=14).rsi()
    out["p1_rsi14_01"] = (rsi14 - 30) / 40  # 旧式と同じスケーリング

    # --- Stochastic Oscillator (Kを0–1化) ---
    stoch = ta.momentum.StochasticOscillator(
        high, low, close, window=14, smooth_window=3)
    k = stoch.stoch()
    out["p1_stoch_k01"] = k / 100.0  # 0–1に正規化

    # --- Bollinger Bands (旧名に合わせる) ---
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    hband, lband, mavg = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    out["p1_bb_pos01"] = (close - lband) / (hband - lband + 1e-9)
    out["p1_bb_bw01"] = (hband - lband) / (mavg + 1e-9)

    # --- ATR ---
    atr14 = ta.volatility.AverageTrueRange(
        high, low, close, window=14).average_true_range()

    out["p1_atr14_norm"] = atr14 / close

    # --- VWAP distance (旧名に合わせる) ---
    vwap = (vol * (high + low + close) / 3).cumsum() / (vol.cumsum() + 1e-9)
    out["p1_vwap_dist01"] = (close - vwap) / vwap

    # --- Volume Imbalance (旧名に合わせる) ---
    ret1 = close.pct_change()
    up_vol = vol.where(ret1 > 0, 0)
    dn_vol = vol.where(ret1 < 0, 0)
    imb = (up_vol - dn_vol) / (up_vol + dn_vol + 1e-9)
    out["p1_vol_imb01"] = imb

    # --- Opening Range ---
    or_high = high.rolling(5).max()
    or_low = low.rolling(5).min()
    out["p1_or_pos"] = (close - or_low) / (or_high - or_low + 1e-9)

    # --- Volume ratio ---
    out["p1_vol_ratio20"] = vol / (vol.rolling(20).mean() + 1e-9)

    return out


# ============================
# Pack2: ゲート・安定化系（修正版）
# ============================
def compute_pack2(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    volume = df["volume"]

    # --- 旧仕様の再現（必須） ---
    # リターン系
    out["p2_ret1"] = close.pct_change(1)
    out["p2_ret1_z"] = _rolling_z(out["p2_ret1"], 20)

    # ボリンジャーバンド位置
    bb = BollingerBands(close, window=20, window_dev=2)
    bb_pos = (close - bb.bollinger_lband()) / \
        (bb.bollinger_hband() - bb.bollinger_lband())
    out["p2_bb_pos"] = bb_pos

    # VWAP gap
    vwap = (df["close"] * df["volume"]).cumsum() / \
        (df["volume"].cumsum() + 1e-9)
    vwap_gap = close - vwap
    out["p2_vwap_gap"] = vwap_gap
    out["p2_vwap_gap_z"] = _rolling_z(vwap_gap, 20)
    out["p2_vwap_gap01"] = (vwap_gap > 0).astype(int)  # 旧仕様では符号フラグに近い挙動だった

    # Volume Z-score
    out["p2_vol_z01"] = np.tanh(_rolling_z(volume, 20))

    # 移動平均差分
    ma_fast = close.rolling(10).mean()
    ma_slow = close.rolling(30).mean()
    out["p2_ma_fast_slow"] = ma_fast - ma_slow

    # --- 改善版の追加（新カラム） ---
    out["p2_vwap_gap_pct"] = vwap_gap / vwap.replace(0, np.nan)
    out["p2_ma_diff_z"] = _rolling_z(out["p2_ma_fast_slow"], 20)
    out["p2_ma20_slope"] = ma_slow.diff()
    out["p2_vol_z"] = _rolling_z(volume, 20)
    out["p2_vol_spike"] = (out["p2_vol_z"] > 2).astype(int)

    return out

# ============================
# Pack3: テクニカル + 外部①
# ============================


def compute_pack3(df: pd.DataFrame, df_mkt: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    out["p3_macd_hist"] = hist
    out["p3_macd_hist01"] = np.tanh(hist)
    out["p3_macd"] = macd
    out["p3_macd_signal"] = signal

    # --- Williams %R (10期間) ---
    wr10 = -100 * (high.rolling(10).max() - close) / \
        (high.rolling(10).max() - low.rolling(10).min() + 1e-9)
    out["p3_wr10"] = wr10
    out["p3_wr10_01"] = wr10 / -50.0  # [-1, 1] に近似

    # --- 出来高スパイク ---
    vol_z20 = _rolling_z(volume, 20)
    out["p3_vol_spike20"] = vol_z20
    out["p3_vol_spike01"] = (vol_z20 > 2).astype(int)

    # --- CCI (20) ---
    tp = (high + low + close) / 3
    cci20 = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std(ddof=0))
    out["p3_cci20"] = cci20
    out["p3_cci20_01"] = np.tanh(cci20 / 100)

    # --- Keltner Channel ポジション ---
    atr = (high - low).rolling(20).mean()
    kel_upper = close.rolling(20).mean() + 2 * atr
    kel_lower = close.rolling(20).mean() - 2 * atr
    out["p3_keltner_pos01"] = (close - kel_lower) / \
        (kel_upper - kel_lower + 1e-9)

    # --- 外部市場 (VIXJ) ---
    if df_mkt is not None and "VIXJ" in df_mkt.columns:
        vix = df_mkt["VIXJ"].reindex(df.index).ffill()
        out["p3_vixj"] = vix
        out["p3_vixj_01"] = _rolling_z(vix, 60)

    return out


def compute_pack4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pack4: マクロ・市場全体・日次ベース特徴量
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # 日次リターン & ボラ
    out["p4_daily_ret"] = close.pct_change().fillna(0)
    out["p4_daily_range_pct"] = ((high - low) / close.shift()).fillna(0)
    out["p4_daily_vol_z"] = _ewm_zscore(out["p4_daily_ret"], span=30)

    # intraday系
    out["p4_ret5m"] = close.pct_change(5).fillna(0)
    out["p4_ret15m"] = close.pct_change(15).fillna(0)
    out["p4_vol5m_z"] = _ewm_zscore(vol.pct_change().fillna(0), span=30)

    # EMAクロス
    ema_fast = close.ewm(span=5).mean()
    ema_slow = close.ewm(span=20).mean()
    out["p4_ema5m_cross"] = (ema_fast > ema_slow).astype(int)

    # High-Low レンジ比
    out["p4_hl_range5m"] = (high - low) / close.shift()

    # 時間・暦要素
    idx = df.index
    out["p4_dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["p4_month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    out["p4_month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    out["p4_week_of_month"] = ((idx.day - 1) // 7 + 1).astype(int)
    mins_to_close = 15*60 - (idx.hour*60 + idx.minute)
    out["p4_mins_to_close"] = np.clip(mins_to_close, a_min=0, a_max=None)

    # 前日終値とのギャップ
    out["p4_prev_close_gap"] = close / close.shift(1) - 1

    # マーケット系（外部df_mkt必須）
    # TODO: 実運用では df_mkt["bond"], df_mkt["gold"], ... を同期して相関
    # 仮でダミーを返す
    out["p4_beta_mkt30"] = np.nan
    out["p4_corr_bond"] = np.nan
    out["p4_corr_gold"] = np.nan
    out["p4_corr_oil"] = np.nan
    out["p4_corr_sp500"] = np.nan
    out["p4_corr_vix"] = np.nan

    # 改良追加
    out["p4_range_pos20"] = (close - close.rolling(20).min()) / \
        (close.rolling(20).max() - close.rolling(20).min())
    out["p4_skew30"] = close.pct_change().rolling(30).skew()
    out["p4_skew60_z"] = close.pct_change().rolling(60).skew().fillna(0)

    return out

# ---------- Pack5: ボラティリティ / レンジ（拡張） ----------


def compute_pack5(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # ATR系
    atr14 = AverageTrueRange(h, l, c, window=14).average_true_range()
    out["p5_atr14"] = atr14
    out["p5_atr14_n"] = atr14 / c

    # 標準偏差・実現ボラ
    std20 = c.pct_change().rolling(20).std()
    out["p5_std20"] = std20
    out["p5_std20_n"] = std20 / \
        (c.pct_change().rolling(60).std() + 1e-12)  # 相対化

    # Garman–Klass / Parkinson（高低レンジ由来）
    hl = np.log(h / l).replace([np.inf, -np.inf], np.nan)
    co = np.log(c / o).replace([np.inf, -np.inf], np.nan)
    gk20 = (0.5 * hl.pow(2) - (2*np.log(2)-1) * co.pow(2)
            ).rolling(20).mean().clip(lower=0).pow(0.5)
    out["p5_vol_gk20"] = gk20

    pk20 = (hl.pow(2) / (4*np.log(2))).rolling(20).mean().clip(lower=0).pow(0.5)
    out["p5_vol_pk20"] = pk20

    # Bollinger 幅 / Keltner 幅（バンド幅）
    bb = BollingerBands(close=c, window=20, window_dev=2)
    bbw = (bb.bollinger_hband() - bb.bollinger_lband()) / \
        (bb.bollinger_mavg() + 1e-9)
    out["p5_bb_bw20"] = bbw

    kc = KeltnerChannel(high=h, low=l, close=c, window=20,
                        window_atr=14, original_version=True)
    kc_bw = (kc.keltner_channel_hband() - kc.keltner_channel_lband()
             ) / (kc.keltner_channel_mband() + 1e-9)
    out["p5_kc_bw20"] = kc_bw

    # レンジ系
    out["p5_avg_range20_n"] = (h - l).rolling(20).mean() / c
    out["p5_range_pos20"] = (c - l.rolling(20).min()) / \
        ((h.rolling(20).max() - l.rolling(20).min()) + 1e-9)

    return out


# ---------- Pack6: リターン分布 / 自己相関 / リスク因子（拡張） ----------
def compute_pack6(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["Close"]
    ret1 = c.pct_change()

    # Z化（EWM）
    out["p6_ret_z20"] = _ewm_zscore(ret1, span=20)
    out["p6_ret_z60"] = _ewm_zscore(ret1, span=60)

    # 歪度・尖度（複数窓）
    out["p6_skew20"] = ret1.rolling(20).skew()
    out["p6_kurt20"] = ret1.rolling(20).kurt()
    out["p6_skew60"] = ret1.rolling(60).skew()

    # ダウンサイドボラ / ボラのボラ
    out["p6_downside_vol20"] = _downside_vol(ret1, window=20)
    vol20 = ret1.rolling(20).std()
    out["p6_vol_of_vol20"] = vol20.rolling(20).std()

    # 自己相関（短中期）
    out["p6_autocorr1_60"] = _autocorr(ret1, lag=1, window=60)
    out["p6_autocorr5_60"] = _autocorr(ret1, lag=5, window=60)

    # ローリング最大DD（短期）
    out["p6_mdd60"] = _rolling_max_drawdown(c, window=60)

    # 上昇比率
    out["p6_up_ratio20"] = (ret1 > 0).rolling(20).mean()

    return out


# ---------- Pack7: クロスマーケット / ベータ・相関（拡張・安全整列） ----------
def compute_pack7(df: pd.DataFrame, df_fx: pd.DataFrame | None = None, df_mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["Close"]
    ret = c.pct_change()

    def _mk(name) -> pd.Series | None:
        if df_mkt is not None and name in df_mkt.columns:
            return _align_series(df_mkt[name], df.index)
        return None

    def _fx(name) -> pd.Series | None:
        if df_fx is not None and name in df_fx.columns:
            return _align_series(df_fx[name], df.index)
        return None

    # 相関 & ベータ（60）
    for label, s in {
        "SP500": _mk("SP500"),
        "NK225": _mk("NK225"),
        "VIX": _mk("VIX"),
        "OIL": _mk("OIL"),
        "GOLD": _mk("GOLD"),
        "AUTO": _mk("AUTO"),
        "BOND": _mk("BOND"),
    }.items():
        if s is None:
            continue
        r = s.pct_change()
        out[f"p7_corr_{label.lower()}"] = ret.rolling(60).corr(r)
        out[f"p7_beta_{label.lower()}"] = _rolling_beta(ret, r, window=60)
        # 残差のZ（簡易）
        resid = ret - out[f"p7_beta_{label.lower()}"] * r
        out[f"p7_resid_{label.lower()}_z"] = _ewm_zscore(resid, span=60)

        # リード/ラグ相関（外部が先行する仮定：+5min）
        out[f"p7_corr_lead5_{label.lower()}"] = ret.rolling(
            60).corr(r.shift(5))

    # USDJPY（為替）
    for name in ["USDJPY", "USDJPY=X"]:
        fx = _fx(name)
        if fx is not None:
            r = fx.pct_change()
            out["p7_corr_usdjpy"] = ret.rolling(60).corr(r)
            out["p7_beta_usdjpy"] = _rolling_beta(ret, r, window=60)
            resid = ret - out["p7_beta_usdjpy"] * r
            out["p7_resid_usdjpy_z"] = _ewm_zscore(resid, span=60)
            break

    return out


# ---------- Pack8: 時間・季節性（拡張） ----------
def compute_pack8(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    idx = df.index

    # 分（UTC基準）：分解しやすい一般形
    minute_of_day = idx.hour * 60 + idx.minute
    out["p8_mins_since_open"] = minute_of_day  # そのまま保持

    # 時間の周期（1日周期）
    out["p8_tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["p8_tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

    # 曜日（週周期）
    out["p8_dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    out["p8_dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)

    # 月（年周期）
    out["p8_month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12.0)
    out["p8_month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12.0)

    # 週の中での位置（0〜1）
    week_pos = (idx.dayofweek + minute_of_day / 1440.0) / 7.0
    out["p8_week_pos01"] = week_pos

    return out


# ---------- Pack9: レジーム/イベント・ダミー（拡張） ----------
def compute_pack9(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["Close"]
    ret = c.pct_change()

    # 簡易ボラレジーム（20分の実現ボラが中央値より高い＝高ボラ）
    vol20 = ret.rolling(20).std()
    med = vol20.rolling(200, min_periods=20).median()
    out["p9_regime_vol_high"] = (vol20 > med).astype(float)

    # レンジブレイク（20）
    h, l = df["High"], df["Low"]
    out["p9_break20_up"] = (h > h.rolling(20).max().shift(1)).astype(float)
    out["p9_break20_dn"] = (l < l.rolling(20).min().shift(1)).astype(float)

    # ギャップ近似（1分前終値との乖離）
    out["p9_gap1m"] = (df["Open"] - df["Close"].shift(1)) / \
        (df["Close"].shift(1) + 1e-9)

    # ダミー（将来のイベント拡張用の空きボックス）
    out["p9_dummy"] = 0.0

    return out


# ---------- Pack10: 運用/学習と接続する箱（改良） ----------
def compute_pack10(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["Close"]

    # 直近PnL擬似（モメンタムの累積）
    out["p10_recent_pnl"] = c.pct_change().rolling(10).sum()

    # ローカル・ドローダウン
    out["p10_mdd60"] = _rolling_max_drawdown(c, window=60)

    # MLスコアは後工程（predict_scores.py）が signals_1m に書く想定なので、
    # ここは占位のみ（NaNのまま）。必要なら features_1m にも流せるように後で変更可。
    out["p10_ml_score"] = np.nan

    # 当日しきい値（thresholds_daily の参照を想定 → ここは空欄で占位）
    out["p10_thr_long"] = np.nan
    out["p10_thr_short"] = np.nan

    return out


def compute_pack11(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["Close"]

    # 価格アノマリー
    out["p11_gap_open"] = (
        df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    out["p11_streak_up"] = (c.pct_change() > 0).astype(
        int).groupby((c.pct_change() <= 0).cumsum()).cumsum()
    out["p11_streak_down"] = (c.pct_change() < 0).astype(
        int).groupby((c.pct_change() >= 0).cumsum()).cumsum()

    return out


def compute_pack12(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c, v = df["Close"], df["Volume"]

    # OBV
    ret = c.pct_change().fillna(0)
    direction = np.sign(ret)
    out["p12_obv"] = (direction * v).cumsum()

    # 出来高の自己相関
    out["p12_vol_autocorr20"] = v.rolling(60).corr(v.shift(20))

    return out


def compute_pack13(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    h, l, c = df["High"], df["Low"], df["Close"]

    # ADX
    adx = ta.trend.ADXIndicator(h, l, c, window=14).adx()
    out["p13_adx14"] = adx

    # Aroon Up/Down
    aroon = ta.trend.AroonIndicator(high=h, low=l, window=25)
    out["p13_aroon_up"] = aroon.aroon_up()
    out["p13_aroon_down"] = aroon.aroon_down()

    return out


def compute_pack14(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["Close"]
    ret = c.pct_change()

    # シャープ近似
    rolling_ret = ret.rolling(20).mean()
    rolling_vol = ret.rolling(20).std()
    out["p14_sharpe20"] = rolling_ret / (rolling_vol + 1e-9)

    # Sortino比率
    downside_vol = ret.clip(upper=0).pow(2).rolling(20).mean().pow(0.5)
    out["p14_sortino20"] = rolling_ret / (downside_vol + 1e-9)

    return out


def compute_pack15(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    idx = df.index

    # 月末/週末 dummy
    out["p15_eom_dummy"] = (idx.is_month_end).astype(int)
    out["p15_friday_dummy"] = (idx.dayofweek == 4).astype(int)

    return out

# ========= Pack16: 為替相関 =========


def compute_pack16(df, df_fx=None):
    out = pd.DataFrame(index=df.index)
    if df_fx is None:
        return out

    c = df["Close"]

    for pair in ["USDJPY", "EURJPY", "GBPJPY"]:
        if pair in df_fx.columns:
            out[f"p16_corr_{pair.lower()}"] = (
                c.pct_change().rolling(60).corr(df_fx[pair].pct_change())
            )
    return out


# ========= Pack17: 米国株相関 =========
def compute_pack17(df, df_mkt=None):
    out = pd.DataFrame(index=df.index)
    if df_mkt is None:
        return out

    c = df["Close"]

    mapping = {
        "SP500": "sp500",
        "DOW": "dowjones",
        "NDX": "nasdaq100",
    }
    for key, name in mapping.items():
        if key in df_mkt.columns:
            out[f"p17_corr_{name}"] = (
                c.pct_change().rolling(60).corr(df_mkt[key].pct_change())
            )
    return out


# ========= Pack18: コモディティ相関 =========
def compute_pack18(df, df_mkt=None):
    out = pd.DataFrame(index=df.index)
    if df_mkt is None:
        return out

    c = df["Close"]
    mapping = {
        "GOLD": "gold",
        "WTI": "oil",
        "COPPER": "copper",
    }
    for key, name in mapping.items():
        if key in df_mkt.columns:
            out[f"p18_corr_{name}"] = (
                c.pct_change().rolling(60).corr(df_mkt[key].pct_change())
            )
    return out


# ========= Pack19: 金利・債券相関 =========
def compute_pack19(df, df_mkt=None):
    out = pd.DataFrame(index=df.index)
    if df_mkt is None:
        return out

    c = df["Close"]
    mapping = {
        "US10Y": "us10y",
        "JP10Y": "jp10y",
        "DE10Y": "de10y",
    }
    for key, name in mapping.items():
        if key in df_mkt.columns:
            out[f"p19_corr_{name}"] = (
                c.pct_change().rolling(60).corr(df_mkt[key].pct_change())
            )
    return out


# ========= Pack20: VIX & リスク指数相関 =========
def compute_pack20(df, df_mkt=None):
    out = pd.DataFrame(index=df.index)
    if df_mkt is None:
        return out

    c = df["Close"]
    mapping = {
        "VIX": "vix",
        "MOVE": "move",
        "DXY": "dxy",
    }
    for key, name in mapping.items():
        if key in df_mkt.columns:
            out[f"p20_corr_{name}"] = (
                c.pct_change().rolling(60).corr(df_mkt[key].pct_change())
            )
    return out


def restore_compat_columns(out: pd.DataFrame) -> pd.DataFrame:
    """
    旧版で存在していたが新パックで抜けていたカラムを復活させる
    """
    # 1. buy/sell base カラム（ダミーで 0）
    for i in range(1, 6):
        if f"buy{i}_base" not in out:
            out[f"buy{i}_base"] = 0
        if f"sell{i}_base" not in out:
            out[f"sell{i}_base"] = 0

    # 2. p4_holiday_dummy（とりあえず全部0、必要なら jpholiday を後で導入）
    if "p4_holiday_dummy" not in out:
        out["p4_holiday_dummy"] = 0

    # 3. p4_vol_regime_mkt（高ボラダミー）
    if "p4_vol_regime_mkt" not in out:
        if "p4_vol5m_z" in out:
            out["p4_vol_regime_mkt"] = (
                out["p4_vol5m_z"].abs() > 1.0).astype(int)
        else:
            out["p4_vol_regime_mkt"] = 0

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
        compute_pack11(df),
        compute_pack12(df),
        compute_pack13(df),
        compute_pack14(df),
        compute_pack15(df),
        compute_pack16(df, df_fx),
        compute_pack17(df, df_mkt),
        compute_pack18(df, df_mkt),
        compute_pack19(df, df_mkt),
        compute_pack20(df, df_mkt),

    ]
    return pd.concat(parts, axis=1)


def compute_packs(df: pd.DataFrame,
                  df_fx: pd.DataFrame | None = None,
                  df_mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    new = compute_packs_native(df, df_fx, df_mkt)
    new = restore_compat_columns(new)

    if not COMPAT_BACKFILL:
        return new

    # 旧featuresからの完全互換バックフィル
    try:
        from tq.features import compute_packs as old_compute
        old = old_compute(df) if df_mkt is None else old_compute(
            df, df_fx, df_mkt)
    except Exception as e:
        # 旧が読み込めない場合はネイティブのみで返す
        print(
            f"[features_packs] compat backfill skipped (old import failed): {e}")
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

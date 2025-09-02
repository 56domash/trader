# tq/ingest.py
from __future__ import annotations
import time
import pandas as pd

JST = "Asia/Tokyo"

# =========================
# 基本ユーティリティ
# =========================
def _require_yf():
    try:
        import yfinance as yf  # noqa
        return True
    except Exception:
        return False

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinanceが返すMultiIndex列を単一レベルへ"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Index（DatetimeIndex）をUTCのtz-awareに統一"""
    if df is None or df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV列名を標準化して、open,high,low,close,volume のみに絞る"""
    cols = {"Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"}
    df = df.rename(columns=cols)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep]

def _insert_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Index(UTC)を 'datetime' 列に複製（DB書き込み等の利便性向上）"""
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.insert(0, "datetime", df.index)
    return df

def _yf_download(symbol: str, interval: str, period: str, tries: int = 3, backoff: float = 1.0) -> pd.DataFrame:
    """
    yfinanceの安定ダウンロードラッパ。
    - MultiIndex回避: group_by='column'
    - エラー時: 指数バックオフで再試行
    - 成功時: フラット化＋UTC化
    """
    if not _require_yf():
        raise RuntimeError("yfinance not installed")
    import yfinance as yf

    last_err = None
    for k in range(tries):
        try:
            df = yf.download(
                symbol,
                interval=interval,
                period=period,
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",   # ← 単一レベル列を要求
            )
            if df is not None and not df.empty:
                df = _flatten_columns(df)
                df = _to_utc_index(df)
                return df
        except Exception as e:
            last_err = e
        time.sleep(backoff * (2 ** k))
    if last_err:
        # 呼び出し側で空DFを安全に扱えるよう、ここでは例外にしない
        # print(f"[ingest] WARN download failed {symbol}: {repr(last_err)}")
        pass
    return pd.DataFrame()

# =========================
# 価格（対象銘柄） 1分足
# =========================
def fetch_price_1m(symbol: str, period: str) -> pd.DataFrame:
    """
    対象銘柄の1分足OHLCVを取得して整形して返す。
    返り値列: ['datetime','open','high','low','close','volume']
    """
    df = _yf_download(symbol, interval="1m", period=period)
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    df = _normalize_ohlcv(df)
    df = _insert_datetime_column(df)
    # 列順を固定
    cols = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols]

# =========================
# 為替（USDJPY/EURJPY/GBPJPY 等）1分足
# =========================
def fetch_fx_1m(pairs, period: str) -> pd.DataFrame:
    """
    複数ティッカー（例: ["USDJPY=X","EURJPY=X","GBPJPY=X"]）から1分足を取得して横結合。
    列名は主要通貨について標準化（USDJPY, EURJPY, GBPJPY, JPY）。
    返り値例: ['datetime','USDJPY','EURJPY','GBPJPY']（取得できたもののみ）
    """
    if isinstance(pairs, str):
        pairs = [pairs]

    frames = []
    for p in pairs:
        df = _yf_download(p, interval="1m", period=period)
        if df is None or df.empty:
            continue
        # Closeのみを採用（為替のOHLCが必要ならここで拡張）
        close_col = "Close" if "Close" in df.columns else "close" if "close" in df.columns else None
        if close_col is None:
            continue
        f = df[[close_col]].rename(columns={close_col: p})
        frames.append(f)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)

    # 列名を標準化
    rename_map = {
        "USDJPY=X": "USDJPY",
        "EURJPY=X": "EURJPY",
        "GBPJPY=X": "GBPJPY",
        "JPY=X": "JPY",
    }
    out = out.rename(columns=rename_map)

    out = _to_utc_index(out)        # 念のため
    out = _insert_datetime_column(out)

    # ['datetime', <available fx cols>] の順にする
    fx_cols = [c for c in ["USDJPY", "EURJPY", "GBPJPY", "JPY"] if c in out.columns]
    return out[["datetime"] + fx_cols] if fx_cols else out[["datetime"]]

# =========================
# 市場プロキシ（NK225_FUT / SECTOR_AUTO）1分足
# =========================
def fetch_market_1m(period: str) -> pd.DataFrame:
    """
    ・日経先物（代替フォールバック）：NK=F → NIY=F → JP225USD=X
    ・自動車セクターの代用：1622.T（東証ETF）→ 7201.T（日産）など
    Closeのみを1分足で取得し、列名を 'NK225_FUT', 'SECTOR_AUTO' に正規化して返す。
    返り値列（一例）: ['datetime','NK225_FUT','SECTOR_AUTO']
    """
    # Nikkei 225 proxy
    nk_candidates = ["NK=F", "NIY=F", "JP225USD=X"]
    nk_df = None
    for sym in nk_candidates:
        df = _yf_download(sym, interval="1m", period=period)
        if df is not None and not df.empty:
            c = "Close" if "Close" in df.columns else "close" if "close" in df.columns else None
            if c:
                nk_df = df[[c]].rename(columns={c: "NK225_FUT"})
                break

    # Auto sector proxy
    auto_candidates = ["1622.T", "7201.T"]  # 1622.T: 自動車・輸送機ETF, 7201.T: 日産
    auto_df = None
    for sym in auto_candidates:
        df = _yf_download(sym, interval="1m", period=period)
        if df is not None and not df.empty:
            c = "Close" if "Close" in df.columns else "close" if "close" in df.columns else None
            if c:
                auto_df = df[[c]].rename(columns={c: "SECTOR_AUTO"})
                break

    if nk_df is None and auto_df is None:
        return pd.DataFrame()

    # 結合（存在する方だけ）
    frames = []
    if nk_df is not None:
        frames.append(nk_df)
    if auto_df is not None:
        frames.append(auto_df)

    out = pd.concat(frames, axis=1)
    out = _to_utc_index(out)
    out = _insert_datetime_column(out)

    cols = ["datetime"]
    if "NK225_FUT" in out.columns:
        cols.append("NK225_FUT")
    if "SECTOR_AUTO" in out.columns:
        cols.append("SECTOR_AUTO")
    return out[cols]

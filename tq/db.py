
from __future__ import annotations
import sqlite3, json
from pathlib import Path

PRAGMAS = [
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA temp_store=MEMORY;",
]

SCHEMA = [
    '''CREATE TABLE IF NOT EXISTS bars_1m(
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY(symbol, ts)
    );''',
    '''CREATE TABLE IF NOT EXISTS fx_1m(
        pair TEXT NOT NULL,
        ts TEXT NOT NULL,
        close REAL,
        PRIMARY KEY(pair, ts)
    );''',
    '''CREATE TABLE IF NOT EXISTS market_1m(
        kind TEXT NOT NULL,
        ts TEXT NOT NULL,
        close REAL,
        PRIMARY KEY(kind, ts)
    );''',
    '''CREATE TABLE IF NOT EXISTS signals_1m(
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        buy1 REAL, buy2 REAL, buy3 REAL, buy4 REAL, buy5 REAL,
        sell1 REAL, sell2 REAL, sell3 REAL, sell4 REAL, sell5 REAL,
        S REAL,
        meta_json TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY(symbol, ts)
    );''',
    '''CREATE TABLE IF NOT EXISTS orders(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        qty INTEGER NOT NULL,
        px REAL NOT NULL,
        status TEXT NOT NULL,
        corr_id TEXT UNIQUE,
        reason TEXT
    );''',
    '''CREATE TABLE IF NOT EXISTS executions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        ts TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        qty INTEGER NOT NULL,
        px REAL NOT NULL,
        pnl REAL,
        FOREIGN KEY(order_id) REFERENCES orders(id)
    );''',
    '''CREATE TABLE IF NOT EXISTS positions(
        symbol TEXT PRIMARY KEY,
        side TEXT,
        qty INTEGER,
        avg_px REAL,
        entry_ts TEXT
    );''',
    '''CREATE TABLE IF NOT EXISTS meta(
        key TEXT PRIMARY KEY,
        value TEXT
    );''',
    '''CREATE TABLE IF NOT EXISTS risk_flags(
        key TEXT PRIMARY KEY,
        value TEXT
    );'''
]

def connect(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=30.0)
    conn.execute("PRAGMA foreign_keys=ON;")
    for p in PRAGMAS: conn.execute(p)
    for s in SCHEMA: conn.execute(s)
    return conn

def upsert_bar(conn, symbol: str, ts: str, o: float,h:float,l:float,c:float,v:float):
    conn.execute("""INSERT INTO bars_1m(symbol, ts, open, high, low, close, volume)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(symbol, ts) DO UPDATE SET
                      open=excluded.open, high=excluded.high, low=excluded.low,
                      close=excluded.close, volume=excluded.volume""",
                 (symbol, ts, o,h,l,c,v))

def upsert_fx(conn, pair: str, ts: str, close: float):
    conn.execute("""INSERT INTO fx_1m(pair, ts, close)
                    VALUES(?,?,?)
                    ON CONFLICT(pair, ts) DO UPDATE SET
                      close=excluded.close""", (pair, ts, close))

def upsert_market(conn, kind: str, ts: str, close: float):
    conn.execute("""INSERT INTO market_1m(kind, ts, close)
                    VALUES(?,?,?)
                    ON CONFLICT(kind, ts) DO UPDATE SET
                      close=excluded.close""", (kind, ts, close))

def put_meta(conn, key: str, value):
    conn.execute("""INSERT INTO meta(key, value) VALUES(?,?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value""", (key, json.dumps(value)))

def get_meta(conn, key: str, default=None):
    cur = conn.execute("SELECT value FROM meta WHERE key=?", (key,))
    row = cur.fetchone()
    return (json.loads(row[0]) if row and row[0] is not None else default)

def insert_signal(conn, symbol: str, ts: str, buy, sell, S: float, meta: dict):
    conn.execute("""INSERT OR REPLACE INTO signals_1m
        (symbol, ts, buy1,buy2,buy3,buy4,buy5, sell1,sell2,sell3,sell4,sell5, S, meta_json)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (symbol, ts, *buy, *sell, S, json.dumps(meta)))

def last_bar_ts(conn, symbol: str):
    cur = conn.execute("SELECT ts FROM bars_1m WHERE symbol=? ORDER BY ts DESC LIMIT 1", (symbol,))
    r = cur.fetchone()
    return r[0] if r else None

def day_bars_df(conn, symbol: str, date_yyyy_mm_dd: str):
    import pandas as pd
    cur = conn.execute("""SELECT ts, open,high,low,close,volume FROM bars_1m
                          WHERE symbol=? AND substr(ts,1,10)=?
                          ORDER BY ts ASC""", (symbol, date_yyyy_mm_dd))
    rows = cur.fetchall()
    if not rows: return None
    df = pd.DataFrame(rows, columns=["datetime","open","high","low","close","volume"])
    ts = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_convert("Asia/Tokyo")
    df = df.drop(columns=["datetime"]); df.index = ts
    return df

def day_series_df(conn, table: str, keycol: str, key: str, date_yyyy_mm_dd: str):
    import pandas as pd
    cur = conn.execute(f"""SELECT ts, close FROM {table}
                           WHERE {keycol}=? AND substr(ts,1,10)=?
                           ORDER BY ts ASC""", (key, date_yyyy_mm_dd))
    rows = cur.fetchall()
    if not rows: return None
    df = pd.DataFrame(rows, columns=["datetime","close"])
    ts = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_convert("Asia/Tokyo")
    df = df.drop(columns=["datetime"]); df.index = ts
    return df["close"]

def upsert_position(conn, symbol: str, side: str|None, qty: int, avg_px: float|None, entry_ts: str|None):
    conn.execute("""INSERT INTO positions(symbol, side, qty, avg_px, entry_ts)
                    VALUES(?,?,?,?,?)
                    ON CONFLICT(symbol) DO UPDATE SET
                      side=excluded.side, qty=excluded.qty, avg_px=excluded.avg_px, entry_ts=excluded.entry_ts""",
                 (symbol, side, qty, avg_px, entry_ts))

def get_position(conn, symbol: str):
    cur = conn.execute("SELECT side, qty, avg_px, entry_ts FROM positions WHERE symbol=?", (symbol,))
    r = cur.fetchone()
    if not r: return None
    return dict(side=r[0], qty=r[1], avg_px=r[2], entry_ts=r[3])

def insert_order(conn, ts: str, symbol: str, side: str, qty: int, px: float, corr_id: str, reason: str):
    conn.execute("""INSERT OR IGNORE INTO orders(ts, symbol, side, qty, px, status, corr_id, reason)
                    VALUES(?,?,?,?,?,'NEW',?,?)""", (ts, symbol, side, qty, px, corr_id, reason))

def fill_order(conn, corr_id: str, fill_ts: str, fill_px: float, pnl: float|None, symbol: str, side: str, qty: int):
    cur = conn.execute("SELECT id FROM orders WHERE corr_id=?", (corr_id,))
    r = cur.fetchone()
    if not r: return
    oid = r[0]
    conn.execute("UPDATE orders SET status='FILLED' WHERE id=?", (oid,))
    conn.execute("""INSERT INTO executions(order_id, ts, symbol, side, qty, px, pnl)
                    VALUES(?,?,?,?,?,?,?)""", (oid, fill_ts, symbol, side, qty, fill_px, pnl))

def get_risk_flag(conn, key: str, default="0"):
    cur = conn.execute("SELECT value FROM risk_flags WHERE key=?", (key,))
    r = cur.fetchone()
    return (r[0] if r else default)

def set_risk_flag(conn, key: str, value: str):
    conn.execute("""INSERT INTO risk_flags(key, value) VALUES(?,?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value""", (key, value))

def get_meta_ts(conn, key: str):
    return get_meta(conn, key, None)

def _as5floats(x):
    """x がスカラ/シーケンス/ndarray いずれでも、長さ5の float タプルへ"""
    if x is None:
        return (0.5,)*5
    if isinstance(x, (float, int, np.floating, np.integer)):
        return (float(x),)*5
    try:
        lst = [float(v) for v in list(x)]
    except Exception:
        return (0.5,)*5
    if len(lst) < 5:
        lst = lst + [0.5]*(5-len(lst))
    elif len(lst) > 5:
        lst = lst[:5]
    return tuple(lst)

def insert_signal(conn, symbol: str, ts_iso_utc: str, Buy, Sell, S: float, meta: dict):
    """
    signals_1m へ保存。スキーマ例：
      symbol TEXT, ts TEXT PRIMARY KEY,
      buy1 REAL, buy2 REAL, buy3 REAL, buy4 REAL, buy5 REAL,
      sell1 REAL, sell2 REAL, sell3 REAL, sell4 REAL, sell5 REAL,
      S REAL, meta_json TEXT
    """
    b1,b2,b3,b4,b5 = _as5floats(Buy)
    s1,s2,s3,s4,s5 = _as5floats(Sell)
    S = float(S)
    meta_json = json.dumps(meta, ensure_ascii=False, default=float)

    conn.execute("""
        INSERT INTO signals_1m
        (symbol, ts, buy1,buy2,buy3,buy4,buy5, sell1,sell2,sell3,sell4,sell5, S, meta_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(symbol, ts) DO UPDATE SET
          buy1=excluded.buy1, buy2=excluded.buy2, buy3=excluded.buy3, buy4=excluded.buy4, buy5=excluded.buy5,
          sell1=excluded.sell1, sell2=excluded.sell2, sell3=excluded.sell3, sell4=excluded.sell4, sell5=excluded.sell5,
          S=excluded.S, meta_json=excluded.meta_json
    """, (symbol, ts_iso_utc,
          b1,b2,b3,b4,b5, s1,s2,s3,s4,s5, S, meta_json))
    conn.commit()
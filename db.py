import sqlite3

DB_PATH = "runtime.db"


def check_recent_signals(limit=20):
    conn = sqlite3.connect("runtime.db")
    cur = conn.cursor()
    query = f"""
    SELECT ts, S, S_buy, S_sell
    FROM signals_1m
    ORDER BY ts DESC
    LIMIT {limit}
    """
    rows = cur.execute(query).fetchall()
    conn.close()

    print("\n[signals_1m] recent signals:")
    for r in rows:
        print(r)


# def check_recent_signals(limit=20):
#     conn = sqlite3.connect("runtime.db")
#     query = f"""
#     SELECT ts, S, S_buy, S_sell
#     FROM signals_1m
#     ORDER BY ts DESC
#     LIMIT {limit}
#     """
#     df = pd.read_sql(query, conn)
#     conn.close()
#     print("\n[signals_1m] recent signals:")
#     print(df)


check_recent_signals()


def check_trades(limit: int = 20):
    print("\n[trades_1m] Recent trades:")
    query = f"""
        SELECT *
        FROM trades_1m
        ORDER BY ts DESC
        LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    print(df)


def check_trades_by_date(target_date: str):
    print(f"\n[trades_1m] Trades on {target_date}:")
    query = f"""
        SELECT *
        FROM trades_1m
        WHERE date(ts) = '{target_date}'
        ORDER BY ts
    """
    df = pd.read_sql(query, conn)
    print(df)


check_trades(20)                     # 直近20件
check_trades_by_date("2025-09-12")   # 特定日分

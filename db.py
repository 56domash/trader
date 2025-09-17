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

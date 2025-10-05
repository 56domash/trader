"""
データベースの状態をクイックチェック
"""
import sqlite3

db_path = "runtime.db"
conn = sqlite3.connect(db_path, timeout=10_000)

print("\n" + "="*70)
print("Database Quick Check")
print("="*70 + "\n")

# 変数の初期化
bar_count = 0
legacy_count = 0
v3_count = 0
feature_count = 0

# 1. テーブル一覧
print("📋 Tables:")
tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for (table,) in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  - {table:30s}: {count:,} rows")

print("\n" + "-"*70 + "\n")

# 2. bars_1m の詳細
bar_count = conn.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
if bar_count > 0:
    latest_bar = conn.execute("SELECT MAX(ts) FROM bars_1m").fetchone()[0]
    earliest_bar = conn.execute("SELECT MIN(ts) FROM bars_1m").fetchone()[0]
    trading_days = conn.execute(
        "SELECT COUNT(DISTINCT date(ts)) FROM bars_1m").fetchone()[0]

    print(f"📊 Market Data (bars_1m):")
    print(f"  Total bars: {bar_count:,}")
    print(f"  Date range: {earliest_bar} ~ {latest_bar}")
    print(f"  Trading days: {trading_days}")
else:
    print(f"📊 Market Data (bars_1m):")
    print(f"  ⚠️  No market data found")

print("\n" + "-"*70 + "\n")

# 3. signals_1m の詳細
try:
    # Legacy シグナル
    legacy_count = conn.execute(
        "SELECT COUNT(*) FROM signals_1m WHERE S IS NOT NULL").fetchone()[0]
    print(f"🔵 Legacy Signals:")
    print(f"  Count: {legacy_count:,}")

    if legacy_count > 0:
        latest_legacy = conn.execute(
            "SELECT MAX(ts) FROM signals_1m WHERE S IS NOT NULL").fetchone()[0]
        print(f"  Latest: {latest_legacy}")

    print()

    # V3 シグナル確認
    try:
        v3_count = conn.execute(
            "SELECT COUNT(*) FROM signals_1m WHERE V3_S IS NOT NULL").fetchone()[0]
        print(f"🟢 V3 Signals:")
        print(f"  Count: {v3_count:,}")

        if v3_count > 0:
            latest_v3 = conn.execute(
                "SELECT MAX(ts) FROM signals_1m WHERE V3_S IS NOT NULL").fetchone()[0]
            print(f"  Latest: {latest_v3}")

            # アクション分布
            actions = conn.execute("""
                SELECT V3_action, COUNT(*) 
                FROM signals_1m 
                WHERE V3_action IS NOT NULL 
                GROUP BY V3_action
                ORDER BY COUNT(*) DESC
            """).fetchall()

            if actions:
                print(f"  Actions:")
                for action, count in actions:
                    print(f"    - {action}: {count}")
        else:
            print(f"  ⚠️  No V3 signals yet")
    except Exception as e:
        print(f"🟢 V3 Signals:")
        print(f"  ⚠️  V3 columns not found (need to run migrate_db_for_v3.py)")
        v3_count = 0

except Exception as e:
    print(f"❌ Error checking signals: {e}")
    legacy_count = 0
    v3_count = 0

print("\n" + "-"*70 + "\n")

# 4. feature_values の詳細
try:
    feature_count = conn.execute(
        "SELECT COUNT(*) FROM feature_values").fetchone()[0]
    print(f"🔬 Feature Values:")
    print(f"  Total records: {feature_count:,}")

    if feature_count > 0:
        latest_feature = conn.execute(
            "SELECT MAX(ts) FROM feature_values").fetchone()[0]
        unique_features = conn.execute(
            "SELECT COUNT(DISTINCT feature_name) FROM feature_values").fetchone()[0]
        print(f"  Latest: {latest_feature}")
        print(f"  Unique features: {unique_features}")

        # 日付ごとの集計
        daily = conn.execute("""
            SELECT date(ts), COUNT(DISTINCT feature_name), COUNT(*)
            FROM feature_values
            GROUP BY date(ts)
            ORDER BY date(ts) DESC
            LIMIT 3
        """).fetchall()

        if daily:
            print(f"  Recent days:")
            for date, features, records in daily:
                print(f"    - {date}: {features} features, {records} records")
    else:
        print(f"  ⚠️  No feature data yet")

except Exception as e:
    print(f"🔬 Feature Values:")
    print(f"  ⚠️  feature_values table not found: {e}")
    feature_count = 0

conn.close()

print("\n" + "="*70 + "\n")

# 推奨アクション
print("📝 Next Steps:")
print("-"*70)

if bar_count == 0:
    print("⚠️  No market data found!")
    print("👉 Run: python scripts/ingest_loop.py --config config.toyota.yaml")
elif feature_count == 0:
    print("✅ Market data exists")
    print("⚠️  No V3 feature data yet")
    print("👉 Generate V3 signals for a specific date:")
    print("   python scripts/strategy_loop_v3.py --db runtime.db --date 2024-10-01")
elif v3_count == 0:
    print("✅ Market data exists")
    print("✅ Feature data exists")
    print("⚠️  No V3 signals yet (V3 columns may not exist)")
    print("👉 First, ensure V3 columns exist:")
    print("   python migrate_db_for_v3.py")
    print("👉 Then generate V3 signals:")
    print("   python scripts/strategy_loop_v3.py --db runtime.db --date 2024-10-01")
else:
    print("✅ Everything is ready!")
    print(f"   Market data: {bar_count:,} bars")
    print(f"   Legacy signals: {legacy_count:,}")
    print(f"   V3 signals: {v3_count:,}")
    print(f"   Feature data: {feature_count:,}")
    print()
    print("👉 Recommended next steps:")
    print("   1. Run backtest:")
    print("      python scripts/backtest_v3.py --days 7")
    print("   2. Analyze feature contributions:")
    print("      python scripts/analyze_contribution.py")
    print("   3. Optimize weights:")
    print("      python scripts/optimize_weights_v3.py --trials 50")

print()

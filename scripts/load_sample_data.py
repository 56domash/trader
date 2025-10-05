"""
ingest_loop.pyのデバッグ
なぜデータが保存されないかを確認
"""
from zoneinfo import ZoneInfo
from datetime import datetime
import sys
sys.path.append('.')
sys.path.append('scripts')


# tq.ingestをインポート
try:
    from tq import ingest
    print("✅ tq.ingest imported successfully")
except ImportError as e:
    print(f"❌ Failed to import tq.ingest: {e}")
    sys.exit(1)

# データ取得をテスト
print("\nTesting data fetch...")
print("="*70)

# Toyotaのデータ取得
symbol = "7203.T"
print(f"\nFetching data for {symbol}...")

try:
    df = ingest.fetch_bars(symbol, period="5d", interval="1m")

    if df is not None and not df.empty:
        print(f"✅ Fetched {len(df)} rows")
        print(f"\nDate range: {df.index.min()} ~ {df.index.max()}")
        print(f"\nFirst few rows:")
        print(df.head())

        # JST時間に変換して確認
        print("\n" + "="*70)
        print("Checking JST times...")

        JST = ZoneInfo("Asia/Tokyo")
        df_jst = df.copy()
        df_jst.index = df_jst.index.tz_convert(JST)

        print(f"JST range: {df_jst.index.min()} ~ {df_jst.index.max()}")

        # 09:00-10:00 JST のデータを抽出
        df_trading = df_jst.between_time("09:00", "10:00")

        print(f"\nRows in JST 09:00-10:00: {len(df_trading)}")

        if len(df_trading) == 0:
            print("\n⚠️  No data in trading hours (JST 09:00-10:00)")
            print("This is why ingest_loop.py skipped the data!")
            print("\nSolutions:")
            print("  1. Use sample data: python load_sample_data.py --generate")
            print("  2. Fetch data for a valid trading day")
            print("  3. Disable time filtering in ingest_loop.py for testing")
        else:
            print(f"\n✅ Found {len(df_trading)} bars in trading hours")
            print(df_trading.head(10))
    else:
        print("❌ No data fetched")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70 + "\n")

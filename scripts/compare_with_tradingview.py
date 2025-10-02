# compare_with_tradingview.py
from tq.features_packs import compute_pack1
from core.features.implementations.momentum import RSI14Feature
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


print("="*60)
print("実データでRSI比較")
print("="*60)

# 実データ読み込み
data = pd.read_parquet("toyota_real_data.parquet")
print(f"\n✓ データ読み込み: {len(data)} bars")

# 既存システム
old_features = compute_pack1(data)
old_rsi = old_features["p1_rsi14_01"]

# 新システム
new_rsi_feature = RSI14Feature()
new_rsi = new_rsi_feature.compute(data)

# 最新10本を表示
print("\n最新10本のRSI比較:")
print("-" * 70)
print(f"{'Datetime':<20} {'既存(0-1)':<12} {'新(0-100)':<12} {'新/100':<12}")
print("-" * 70)

for i in range(-10, 0):
    dt = data.index[i]
    old_val = old_rsi.iloc[i]
    new_val = new_rsi.iloc[i]
    new_scaled = new_val / 100.0

    print(
        f"{str(dt)[:19]:<20} {old_val:>10.4f}  {new_val:>10.2f}  {new_scaled:>10.4f}")

print("\n" + "="*70)
print("📝 次のステップ:")
print("  1. TradingView で 7203.T を開く")
print("  2. RSI(14) インジケーターを追加")
print("  3. 上記の時刻のRSI値を確認")
print("  4. どちらが近いか報告してください")
print("="*70)

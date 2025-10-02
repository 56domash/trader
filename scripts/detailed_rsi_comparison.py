# detailed_rsi_comparison.py
from tq.features_packs import compute_pack1
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# 実データ読み込み
data = pd.read_parquet("toyota_real_data.parquet")
close = data['close'].values

print("="*60)
print("RSI詳細比較")
print("="*60)
print(f"データ数: {len(close)}")
print(f"価格範囲: {close.min():.2f} - {close.max():.2f}")

# 新システムで段階的計算
delta = np.diff(close, prepend=close[0])
print(f"\n1. Delta計算: {delta[-5:]}")

gains = np.where(delta > 0, delta, 0)
losses = np.where(delta < 0, -delta, 0)
print(f"2. Gains: {gains[-5:]}")
print(f"3. Losses: {losses[-5:]}")

# 14期間の平均
avg_gain = pd.Series(gains).rolling(14).mean()
avg_loss = pd.Series(losses).rolling(14).mean()
print(f"\n4. Avg Gain (最後): {avg_gain.iloc[-1]:.4f}")
print(f"5. Avg Loss (最後): {avg_loss.iloc[-1]:.4f}")

# RS
rs = avg_gain / (avg_loss + 1e-10)
print(f"6. RS (最後): {rs.iloc[-1]:.4f}")

# RSI
rsi = 100 - (100 / (1 + rs))
print(f"7. RSI (最後): {rsi.iloc[-1]:.2f}")

# 既存システム
old_features = compute_pack1(data)
old_rsi_01 = old_features["p1_rsi14_01"].iloc[-1]
print(f"\n既存システム RSI:")
print(f"  0-1値: {old_rsi_01:.4f}")
print(f"  100倍: {old_rsi_01 * 100:.2f}")

print(f"\n差分: {abs(rsi.iloc[-1] - old_rsi_01 * 100):.2f}")

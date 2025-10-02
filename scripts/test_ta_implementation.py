# test_ta_implementation.py
from core.features.implementations.momentum import RSI14Feature
import ta
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# 実データ
data = pd.read_parquet("toyota_real_data.parquet")

# ta-lib直接
rsi_ta = ta.momentum.RSIIndicator(data['close'], window=14).rsi()

# 新システム（修正版）
rsi_new = RSI14Feature().compute(data)

# 比較
print("="*60)
print("修正版の検証")
print("="*60)
print("\n最新10本の比較:")
comparison = pd.DataFrame({
    'ta-lib': rsi_ta.tail(10),
    '新システム': rsi_new.tail(10),
    '差分': (rsi_ta - rsi_new).abs().tail(10)
})
print(comparison)

max_diff = (rsi_ta - rsi_new).abs().max()
print(f"\n最大差分: {max_diff:.6f}")

if max_diff < 0.01:
    print("✅ 完全一致！標準実装になりました")
else:
    print(f"⚠️ まだ差分があります: {max_diff}")

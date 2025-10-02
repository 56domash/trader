# compare_with_ta.py
from tq.features_packs import compute_pack1
from core.features.implementations.momentum import RSI14Feature
import pandas as pd
import ta

data = pd.read_parquet("toyota_real_data.parquet")

# ta-lib標準実装
rsi_ta = ta.momentum.RSIIndicator(data['close'], window=14).rsi()

# 新システム
rsi_new = RSI14Feature().compute(data)

# 既存システム
rsi_old = compute_pack1(data)["p1_rsi14_01"] * 100

print("最新10本の比較:")
print(pd.DataFrame({
    'ta-lib': rsi_ta.tail(10),
    '新システム': rsi_new.tail(10),
    '既存システム': rsi_old.tail(10)
}))

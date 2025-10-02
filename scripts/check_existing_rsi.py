# check_existing_rsi.py
import inspect
from tq.features_packs import compute_pack1

# ソースコードを表示
print("="*60)
print("既存システムのRSI計算コード")
print("="*60)
source = inspect.getsource(compute_pack1)

# RSI関連の行だけ抽出
lines = source.split('\n')
in_rsi = False
rsi_lines = []

for i, line in enumerate(lines):
    if 'rsi' in line.lower():
        in_rsi = True
        # 前後5行も表示
        start = max(0, i-5)
        end = min(len(lines), i+10)
        rsi_lines.extend(lines[start:end])
        break

print('\n'.join(set(rsi_lines)))

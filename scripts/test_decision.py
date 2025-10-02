# test_decision.py
from core.signals.decision import DecisionEngine, ThresholdConfig
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


print("="*60)
print("DecisionEngine テスト")
print("="*60)

# シグナルデータ生成（S信号のみ）
np.random.seed(42)
dates = pd.date_range("2025-09-29", periods=100, freq="1min", tz="UTC")

# トレンドのあるシグナル
trend = np.linspace(-0.3, 0.3, 100)
noise = np.random.randn(100) * 0.05
S = trend + noise

signals = pd.DataFrame({
    "S": S,
    "S_buy": (S + 1) / 2,
    "S_sell": (-S + 1) / 2
}, index=dates)

print(f"\nシグナル範囲: S=[{S.min():.3f}, {S.max():.3f}]")

# DecisionEngine作成
config = ThresholdConfig(
    thr_long=0.15,
    thr_short=-0.15,
    exit_long=0.05,
    exit_short=-0.05,
    confirm_bars=2,
    ema_span=3
)

engine = DecisionEngine(config)

# 判定実行
result = engine.decide(signals, current_position=0)

print(f"\n判定結果:")
print(f"  ENTRY_LONG: {(result['action'] == 'ENTRY_LONG').sum()} 回")
print(f"  ENTRY_SHORT: {(result['action'] == 'ENTRY_SHORT').sum()} 回")
print(f"  HOLD: {(result['action'] == 'HOLD').sum()} 回")

# 最新アクション
latest = engine.get_latest_action(signals)
print(f"\n最新のアクション:")
print(f"  Action: {latest['action']}")
print(f"  S_ema: {latest['S_ema']:.3f}")
print(f"  can_long: {latest['can_long']}")
print(f"  can_short: {latest['can_short']}")

# サンプル表示
print("\n判定結果サンプル（最後の10本）:")
print(result[['S', 'S_ema', 'can_long', 'can_short', 'action']].tail(10))

print("\n✅ DecisionEngine テスト完了")

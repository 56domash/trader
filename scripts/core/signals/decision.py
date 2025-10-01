# core/signals/decision.py
"""
エントリー・エグジット判定
Toyota Trading System V3 - Phase 3A
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class ThresholdConfig:
    """判定閾値の設定"""
    # エントリー閾値
    thr_long: float = 0.15      # ロングエントリー
    thr_short: float = -0.15    # ショートエントリー

    # エグジット閾値
    exit_long: float = 0.05     # ロングクローズ
    exit_short: float = -0.05   # ショートクローズ

    # 判定条件
    confirm_bars: int = 1       # 連続確認本数
    ema_span: int = 3           # EMA平滑化期間

    # 追加ゲート（オプション）
    min_edge: float = 0.0       # 閾値からの余裕
    min_slope: float = 0.0      # 傾き条件
    vwap_gate: bool = False     # VWAPゲート使用


class DecisionEngine:
    """エントリー・エグジット判定エンジン"""

    def __init__(self, config: ThresholdConfig):
        self.config = config

    def decide(self, signals: pd.DataFrame, current_position: int = 0) -> pd.DataFrame:
        """
        エントリー・エグジット判定

        Args:
            signals: SignalAggregatorの出力（S, S_buy, S_sell含む）
            current_position: 現在のポジション（0=フラット, 100=ロング, -100=ショート）

        Returns:
            判定結果を含むDataFrame
            - action: "ENTRY_LONG", "ENTRY_SHORT", "EXIT", "HOLD"
            - S_ema: EMA平滑化されたシグナル
            - can_long: ロング条件満たしているか
            - can_short: ショート条件満たしているか
        """
        df = signals.copy()

        # 1. S信号のEMA平滑化
        df["S_ema"] = df["S"].ewm(
            span=self.config.ema_span, adjust=False).mean()

        # 2. 連続カウント用の状態計算
        above_long = df["S_ema"] >= self.config.thr_long
        below_short = df["S_ema"] <= self.config.thr_short

        # 連続カウント（シンプル実装）
        df["above_long_count"] = (
            above_long.groupby((~above_long).cumsum()).cumcount() + 1
        ) * above_long.astype(int)

        df["below_short_count"] = (
            below_short.groupby((~below_short).cumsum()).cumcount() + 1
        ) * below_short.astype(int)

        # 3. エントリー条件判定
        df["can_long"] = (
            (df["above_long_count"] >= self.config.confirm_bars) &
            (df["S_ema"] >= self.config.thr_long + self.config.min_edge)
        )

        df["can_short"] = (
            (df["below_short_count"] >= self.config.confirm_bars) &
            (df["S_ema"] <= self.config.thr_short - self.config.min_edge)
        )

        # 4. エグジット条件判定（ポジションに応じて）
        if current_position > 0:  # ロング保有中
            df["should_exit"] = df["S_ema"] <= self.config.exit_long
        elif current_position < 0:  # ショート保有中
            df["should_exit"] = df["S_ema"] >= self.config.exit_short
        else:
            df["should_exit"] = False

        # 5. アクション決定
        df["action"] = "HOLD"

        if current_position == 0:
            # フラット時：エントリー判定
            df.loc[df["can_long"], "action"] = "ENTRY_LONG"
            df.loc[df["can_short"], "action"] = "ENTRY_SHORT"
        else:
            # ポジション保有中：エグジット判定
            df.loc[df["should_exit"], "action"] = "EXIT"

        return df

    def get_latest_action(self, signals: pd.DataFrame, current_position: int = 0) -> dict:
        """
        最新のアクションを取得（リアルタイム用）

        Returns:
            {"action": "ENTRY_LONG"|"EXIT"|"HOLD", "S_ema": float, ...}
        """
        result = self.decide(signals, current_position)
        latest = result.iloc[-1]

        return {
            "action": latest["action"],
            "S_ema": float(latest["S_ema"]),
            "S": float(latest["S"]),
            "can_long": bool(latest.get("can_long", False)),
            "can_short": bool(latest.get("can_short", False)),
            "timestamp": result.index[-1]
        }

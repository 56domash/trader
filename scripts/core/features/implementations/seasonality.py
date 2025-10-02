# core/features/implementations/seasonality.py
"""
Seasonality Features
時間帯・曜日・カレンダー効果を捉える特徴量
"""

from ..base import Feature, FeatureMetadata
import pandas as pd
import numpy as np
from datetime import datetime


class TimeOfDayFeature(Feature):
    """時間帯特徴量（09:00-10:00 内での進行度）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="time_of_day",
            category="seasonality",
            version="1.0",
            lookback_bars=0,
            expected_range=(0, 1),
            description="取引時間内での進行度（09:00=0, 10:00=1）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """時間帯スコア計算"""
        result = pd.Series(0.0, index=data.index)

        for idx in data.index:
            jst_time = idx.tz_convert('Asia/Tokyo')

            # 09:00-10:00の範囲での進行度
            if jst_time.hour == 9:
                progress = jst_time.minute / 60.0  # 0.0 - 1.0
                result.loc[idx] = progress
            elif jst_time.hour >= 10:
                result.loc[idx] = 1.0

        return result


class EarlySessionBiasFeature(Feature):
    """セッション前半バイアス（09:00-09:30）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="early_session_bias",
            category="seasonality",
            version="1.0",
            lookback_bars=0,
            expected_range=(0, 1),
            description="セッション前半（09:00-09:30）で1、後半で0"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """前半/後半フラグ"""
        result = pd.Series(0.0, index=data.index)

        for idx in data.index:
            jst_time = idx.tz_convert('Asia/Tokyo')

            if jst_time.hour == 9 and jst_time.minute < 30:
                result.loc[idx] = 1.0
            else:
                result.loc[idx] = 0.0

        return result


class DayOfWeekFeature(Feature):
    """曜日効果（月曜効果など）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="day_of_week",
            category="seasonality",
            version="1.0",
            lookback_bars=0,
            expected_range=(0, 1),
            description="曜日（月=0.0, 金=1.0）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """曜日スコア計算"""
        result = pd.Series(0.0, index=data.index)

        for idx in data.index:
            jst_time = idx.tz_convert('Asia/Tokyo')
            weekday = jst_time.weekday()  # 0=Monday, 4=Friday

            # 0-4 → 0.0-1.0
            result.loc[idx] = weekday / 4.0

        return result


class MonthEndEffectFeature(Feature):
    """月末効果（リバランス需要）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="month_end_effect",
            category="seasonality",
            version="1.0",
            lookback_bars=0,
            expected_range=(0, 1),
            description="月末5営業日で1、それ以外で0"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """月末効果フラグ"""
        result = pd.Series(0.0, index=data.index)

        for idx in data.index:
            jst_time = idx.tz_convert('Asia/Tokyo')

            # 簡易実装: 月の最後5日間
            day_of_month = jst_time.day

            # 月の日数取得
            if jst_time.month == 12:
                next_month = jst_time.replace(
                    year=jst_time.year + 1, month=1, day=1)
            else:
                next_month = jst_time.replace(month=jst_time.month + 1, day=1)

            last_day = (next_month - pd.Timedelta(days=1)).day

            if day_of_month >= last_day - 5:
                result.loc[idx] = 1.0

        return result

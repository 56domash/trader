# core/features/implementations/microstructure.py (修正版)
"""
マイクロストラクチャ系特徴量: VWAP（修正版）
Toyota Trading System V3 - Phase 1 (バグフィックス)
"""

import pandas as pd
import numpy as np
from ..base import Feature, FeatureMetadata


class VWAPDeviationFeature(Feature):
    """VWAP からの乖離（修正版）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="vwap_deviation",
            category="microstructure",
            version="1.1",  # バージョンアップ
            lookback_bars=1,
            expected_range=(-0.05, 0.05),  # ±5%以内が正常
            description="VWAP からの乖離率（修正版）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """VWAP乖離計算（バグ修正版）"""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Typical Price
        tp = (high + low + close) / 3

        # 🔧 修正1: volume * tp の計算
        tpv = tp * volume

        # 🔧 修正2: ゼロ除算を防ぐ
        cumulative_vol = volume.cumsum()
        # volumeがゼロの行をスキップ
        valid_mask = cumulative_vol > 0

        # VWAP計算
        vwap = pd.Series(np.nan, index=data.index)
        vwap[valid_mask] = tpv.cumsum()[valid_mask] / \
            cumulative_vol[valid_mask]

        # 前方補完（VWAPが計算できない初期値）
        vwap = vwap = vwap.bfill().fillna(close)

        # 🔧 修正3: 異常値チェック
        # VWAPがcloseから50%以上乖離していたら異常値として除外
        deviation_raw = (close - vwap) / vwap
        deviation_raw = deviation_raw.replace([np.inf, -np.inf], np.nan)

        # 異常値（±0.5超）をclipして補正
        deviation = deviation_raw.clip(-0.5, 0.5)

        return deviation.fillna(0.0)


class VWAPDeviationFeatureRolling(Feature):
    """VWAP からの乖離（ローリング版・代替案）"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="vwap_deviation_rolling",
            category="microstructure",
            version="1.0",
            lookback_bars=20,  # 20分のローリングVWAP
            expected_range=(-0.05, 0.05),
            description="20期間ローリングVWAPからの乖離率"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """ローリングVWAP計算（累積VWAPより安定）"""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Typical Price
        tp = (high + low + close) / 3
        tpv = tp * volume

        # ローリング窓でVWAP計算（累積より安定）
        window = 20
        rolling_tpv = tpv.rolling(window, min_periods=1).sum()
        rolling_vol = volume.rolling(window, min_periods=1).sum()

        # ゼロ除算回避
        vwap = rolling_tpv / (rolling_vol + 1e-10)

        # 乖離率
        deviation = (close - vwap) / (vwap + 1e-10)
        deviation = deviation.replace([np.inf, -np.inf], np.nan)
        deviation = deviation.clip(-0.5, 0.5)

        return deviation.fillna(0.0)


class VolumeSpikeFeature(Feature):
    """Volume Spike - 出来高Z-score"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="volume_spike",
            category="microstructure",
            version="1.0",
            lookback_bars=20,
            expected_range=(-3, 5),
            description="出来高急増検出（Z-score）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Volume Z-score = (volume - mean) / std
        Z > 2.0 → 出来高急増
        """
        volume = data["volume"]

        # 20期間の平均と標準偏差
        mean_vol = volume.rolling(20, min_periods=10).mean()
        std_vol = volume.rolling(20, min_periods=10).std()

        # ゼロ除算回避
        std_vol = std_vol.replace(0, np.nan)
        std_vol = std_vol.fillna(volume.std() if volume.std() > 0 else 1.0)

        # Z-score計算
        z_score = (volume - mean_vol) / std_vol

        # 範囲制限 (-3σ ~ 5σ)
        z_score = z_score.clip(-3, 5).fillna(0.0)

        return z_score


class VolumeImbalanceFeature(Feature):
    """Volume Imbalance - 買い/売り出来高バランス"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="volume_imbalance",
            category="microstructure",
            version="1.0",
            lookback_bars=1,
            expected_range=(-1, 1),
            description="買い/売り出来高のバランス"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Volume Imbalance = (up_volume - down_volume) / total_volume
        1.0 = 完全に買い優勢
        -1.0 = 完全に売り優勢
        """
        close = data["close"]
        volume = data["volume"]

        # リターン計算
        returns = close.pct_change()

        # 上昇時の出来高
        up_volume = volume.where(returns > 0, 0)

        # 下落時の出来高
        down_volume = volume.where(returns < 0, 0)

        # 総出来高
        total_volume = up_volume + down_volume

        # インバランス計算
        imbalance = (up_volume - down_volume) / (total_volume + 1e-10)

        # 範囲制限
        imbalance = imbalance.clip(-1, 1).fillna(0.0)

        return imbalance


class VolumeRatioFeature(Feature):
    """Volume Ratio - 現在出来高 / 平均出来高"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="volume_ratio",
            category="microstructure",
            version="1.0",
            lookback_bars=20,
            expected_range=(0, 5),
            description="現在出来高 / 20期間平均"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Volume Ratio = volume / rolling_average(20)
        1.0 = 平均的
        2.0 = 平均の2倍
        0.5 = 平均の半分
        """
        volume = data["volume"]

        # 20期間平均出来高
        avg_volume = volume.rolling(20, min_periods=10).mean()

        # 比率計算
        ratio = volume / (avg_volume + 1e-10)

        # 範囲制限 (0～5倍まで)
        ratio = ratio.clip(0, 5).fillna(1.0)

        return ratio

# core/features/implementations/microstructure.py に追加


class OpeningRangePositionFeature(Feature):
    """Opening Range Position (Toyota特化)"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="opening_range_position",
            category="microstructure",
            version="1.0",
            lookback_bars=5,  # 09:00-09:05
            expected_range=(0, 1),
            description="09:00-09:05のレンジ内での現在価格位置"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """OR Position計算"""
        # 注意: dataのindexはUTC時刻
        # 09:00-09:05 JST = 00:00-00:05 UTC

        result = pd.Series(0.5, index=data.index)

        for idx in data.index:
            # その日の00:00-00:05 UTCのデータを取得
            jst_time = idx.tz_convert('Asia/Tokyo')

            if jst_time.hour >= 9:  # 9時以降のみ計算
                # その日の9:00-9:05のデータを抽出
                day_start = jst_time.replace(hour=0, minute=0, second=0)
                or_start = day_start.replace(
                    hour=9, minute=0).tz_convert('UTC')
                or_end = day_start.replace(hour=9, minute=5).tz_convert('UTC')

                or_data = data[(data.index >= or_start)
                               & (data.index < or_end)]

                if not or_data.empty:
                    or_high = or_data["high"].max()
                    or_low = or_data["low"].min()
                    current_price = data.loc[idx, "close"]

                    if or_high > or_low:
                        position = (current_price - or_low) / \
                            (or_high - or_low)
                        result.loc[idx] = min(max(position, 0), 1)

        return result


class OpeningRangeBreakoutFeature(Feature):
    """Opening Range Breakout検出"""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="opening_range_breakout",
            category="microstructure",
            version="1.0",
            lookback_bars=5,
            expected_range=(0, 1),
            description="ORブレイクアウト検出（上抜け=1, 下抜け=0）"
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """ORブレイクアウト検出"""
        result = pd.Series(0.5, index=data.index)

        for idx in data.index:
            jst_time = idx.tz_convert('Asia/Tokyo')

            if jst_time.hour >= 9:
                day_start = jst_time.replace(hour=0, minute=0, second=0)
                or_start = day_start.replace(
                    hour=9, minute=0).tz_convert('UTC')
                or_end = day_start.replace(hour=9, minute=5).tz_convert('UTC')

                or_data = data[(data.index >= or_start)
                               & (data.index < or_end)]

                if not or_data.empty:
                    or_high = or_data["high"].max()
                    or_low = or_data["low"].min()
                    current_price = data.loc[idx, "close"]

                    # ブレイクアウト判定
                    if current_price > or_high * 1.001:  # 0.1%以上の上抜け
                        result.loc[idx] = 1.0
                    elif current_price < or_low * 0.999:  # 0.1%以上の下抜け
                        result.loc[idx] = 0.0

        return result

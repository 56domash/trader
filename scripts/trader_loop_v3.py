# scripts/trader_loop_v3.py
"""
V3対応トレーディングループ
既存システムとV3を並行稼働
"""

import argparse
import sqlite3
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Optional
import pandas as pd

JST = ZoneInfo("Asia/Tokyo")


@dataclass
class Position:
    """ポジション"""
    symbol: str
    direction: str  # LONG or SHORT
    entry_time: datetime
    entry_price: float
    size: int
    system: str  # V3 or LEGACY


class TradingLoopV3:
    """V3対応トレーディングループ"""

    def __init__(self, db_path: str, symbol: str = "7203.T",
                 dry_run: bool = True, base_size: int = 100):
        self.db_path = db_path
        self.symbol = symbol
        self.dry_run = dry_run
        self.base_size = base_size

        # ポジション管理
        self.legacy_position: Optional[Position] = None
        self.v3_position: Optional[Position] = None

        # 日次P&L
        self.legacy_pnl_today = 0.0
        self.v3_pnl_today = 0.0

        # DB接続
        self.conn = None

    def initialize(self):
        """初期化"""
        self.conn = sqlite3.connect(self.db_path, timeout=10_000)
        self._create_v3_tables()

        print(f"\n{'='*60}")
        print(f"V3 Trading Loop Initialized")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Dry Run: {self.dry_run}")
        print(f"Base Size: {self.base_size}")
        print(f"{'='*60}\n")

    def _create_v3_tables(self):
        """V3用テーブル作成"""
        # V3ポジションテーブル
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS v3_positions (
            symbol TEXT,
            entry_time TEXT,
            exit_time TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            size INTEGER,
            pnl REAL,
            status TEXT,
            PRIMARY KEY (symbol, entry_time)
        )
        """)

        # V3約定テーブル
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS v3_fills (
            symbol TEXT,
            ts TEXT,
            side TEXT,
            price REAL,
            size INTEGER,
            commission REAL,
            v3_signal REAL,
            PRIMARY KEY (symbol, ts)
        )
        """)

        self.conn.commit()

    def run_cycle(self):
        """1サイクル実行"""
        now = datetime.now(JST)

        # 取引時間チェック（09:00-10:00）
        if not self._is_trading_time(now):
            return

        # 最新価格取得
        current_price = self._get_current_price()
        if current_price is None:
            return

        # Legacyシグナルチェック
        legacy_action = self._check_legacy_signal()
        self._process_legacy_action(legacy_action, current_price)

        # V3シグナルチェック
        v3_action = self._check_v3_signal()
        self._process_v3_action(v3_action, current_price)

        # ステータス表示
        self._print_status(current_price)

    def _is_trading_time(self, now: datetime) -> bool:
        """取引時間判定"""
        hour = now.hour
        minute = now.minute

        # JST 09:00-10:00
        if hour == 9:
            return True
        elif hour == 10 and minute == 0:
            return True
        else:
            return False

    def _get_current_price(self) -> Optional[float]:
        """最新価格取得"""
        query = """
        SELECT close FROM bars_1m
        WHERE symbol = ?
        ORDER BY ts DESC
        LIMIT 1
        """

        row = self.conn.execute(query, (self.symbol,)).fetchone()
        return row[0] if row else None

    def _check_legacy_signal(self) -> str:
        """Legacyシグナル取得"""
        query = """
        SELECT S FROM signals_1m
        WHERE symbol = ?
        ORDER BY ts DESC
        LIMIT 1
        """

        row = self.conn.execute(query, (self.symbol,)).fetchone()
        if not row:
            return 'HOLD'

        S = row[0]

        # エントリー判定
        if self.legacy_position is None:
            if S >= 0.15:
                return 'ENTRY_LONG'
            elif S <= -0.15:
                return 'ENTRY_SHORT'

        # エグジット判定
        elif self.legacy_position is not None:
            if self.legacy_position.direction == 'LONG' and S <= 0.05:
                return 'EXIT'
            elif self.legacy_position.direction == 'SHORT' and S >= -0.05:
                return 'EXIT'

        return 'HOLD'

    def _check_v3_signal(self) -> str:
        """V3シグナル取得"""
        query = """
        SELECT V3_action, V3_S_ema
        FROM signals_1m
        WHERE symbol = ?
        ORDER BY ts DESC
        LIMIT 1
        """

        row = self.conn.execute(query, (self.symbol,)).fetchone()
        if not row or row[0] is None:
            return 'HOLD'

        return row[0]  # V3_action

    def _process_legacy_action(self, action: str, price: float):
        """Legacyアクション処理"""
        if action == 'ENTRY_LONG' and self.legacy_position is None:
            self._legacy_enter_long(price)

        elif action == 'ENTRY_SHORT' and self.legacy_position is None:
            self._legacy_enter_short(price)

        elif action == 'EXIT' and self.legacy_position is not None:
            self._legacy_exit(price)

    def _process_v3_action(self, action: str, price: float):
        """V3アクション処理"""
        if action == 'ENTRY_LONG' and self.v3_position is None:
            self._v3_enter_long(price)

        elif action == 'ENTRY_SHORT' and self.v3_position is None:
            self._v3_enter_short(price)

        elif action == 'EXIT' and self.v3_position is not None:
            self._v3_exit(price)

    def _legacy_enter_long(self, price: float):
        """Legacy LONG エントリー"""
        self.legacy_position = Position(
            symbol=self.symbol,
            direction='LONG',
            entry_time=datetime.now(JST),
            entry_price=price,
            size=self.base_size,
            system='LEGACY'
        )
        print(f"[LEGACY] ENTRY LONG @ {price:.1f} x {self.base_size}")

    def _legacy_enter_short(self, price: float):
        """Legacy SHORT エントリー"""
        self.legacy_position = Position(
            symbol=self.symbol,
            direction='SHORT',
            entry_time=datetime.now(JST),
            entry_price=price,
            size=self.base_size,
            system='LEGACY'
        )
        print(f"[LEGACY] ENTRY SHORT @ {price:.1f} x {self.base_size}")

    def _legacy_exit(self, price: float):
        """Legacy エグジット"""
        if self.legacy_position is None:
            return

        # P&L計算
        if self.legacy_position.direction == 'LONG':
            pnl = (price - self.legacy_position.entry_price) * \
                self.legacy_position.size
        else:
            pnl = (self.legacy_position.entry_price - price) * \
                self.legacy_position.size

        # 手数料
        commission = (self.legacy_position.entry_price + price) * \
            self.legacy_position.size * 0.0002
        pnl -= commission

        self.legacy_pnl_today += pnl

        print(
            f"[LEGACY] EXIT {self.legacy_position.direction} @ {price:.1f}, P&L: {pnl:+.0f}")

        self.legacy_position = None

    def _v3_enter_long(self, price: float):
        """V3 LONG エントリー"""
        self.v3_position = Position(
            symbol=self.symbol,
            direction='LONG',
            entry_time=datetime.now(JST),
            entry_price=price,
            size=self.base_size,
            system='V3'
        )

        # DB記録
        self.conn.execute("""
        INSERT INTO v3_positions (symbol, entry_time, direction, entry_price, size, status)
        VALUES (?, ?, 'LONG', ?, ?, 'OPEN')
        """, (self.symbol, self.v3_position.entry_time.isoformat(), price, self.base_size))
        self.conn.commit()

        print(f"[V3] ENTRY LONG @ {price:.1f} x {self.base_size}")

    def _v3_enter_short(self, price: float):
        """V3 SHORT エントリー"""
        self.v3_position = Position(
            symbol=self.symbol,
            direction='SHORT',
            entry_time=datetime.now(JST),
            entry_price=price,
            size=self.base_size,
            system='V3'
        )

        # DB記録
        self.conn.execute("""
        INSERT INTO v3_positions (symbol, entry_time, direction, entry_price, size, status)
        VALUES (?, ?, 'SHORT', ?, ?, 'OPEN')
        """, (self.symbol, self.v3_position.entry_time.isoformat(), price, self.base_size))
        self.conn.commit()

        print(f"[V3] ENTRY SHORT @ {price:.1f} x {self.base_size}")

    def _v3_exit(self, price: float):
        """V3 エグジット"""
        if self.v3_position is None:
            return

        # P&L計算
        if self.v3_position.direction == 'LONG':
            pnl = (price - self.v3_position.entry_price) * \
                self.v3_position.size
        else:
            pnl = (self.v3_position.entry_price - price) * \
                self.v3_position.size

        # 手数料
        commission = (self.v3_position.entry_price + price) * \
            self.v3_position.size * 0.0002
        pnl -= commission

        self.v3_pnl_today += pnl

        # DB更新
        self.conn.execute("""
        UPDATE v3_positions
        SET exit_time = ?, exit_price = ?, pnl = ?, status = 'CLOSED'
        WHERE symbol = ? AND entry_time = ?
        """, (datetime.now(JST).isoformat(), price, pnl,
              self.symbol, self.v3_position.entry_time.isoformat()))
        self.conn.commit()

        print(
            f"[V3] EXIT {self.v3_position.direction} @ {price:.1f}, P&L: {pnl:+.0f}")

        self.v3_position = None

    def _print_status(self, price: float):
        """ステータス表示"""
        now = datetime.now(JST)
        total_pnl = self.legacy_pnl_today + self.v3_pnl_today

        legacy_status = f"{self.legacy_position.direction}" if self.legacy_position else "FLAT"
        v3_status = f"{self.v3_position.direction}" if self.v3_position else "FLAT"

        print(f"[{now.strftime('%H:%M:%S')}] Price: {price:.1f} | "
              f"Legacy: {legacy_status} | V3: {v3_status} | "
              f"P&L: Legacy={self.legacy_pnl_today:+.0f} V3={self.v3_pnl_today:+.0f} Total={total_pnl:+.0f}")

    def close(self):
        """クリーンアップ"""
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="V3 Trading Loop")
    parser.add_argument("--db", default="runtime.db", help="Database path")
    parser.add_argument("--symbol", default="7203.T", help="Symbol")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--interval", type=int, default=10,
                        help="Check interval (seconds)")

    args = parser.parse_args()

    loop = TradingLoopV3(args.db, args.symbol, dry_run=args.dry_run)
    loop.initialize()

    try:
        while True:
            loop.run_cycle()
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        loop.close()


if __name__ == "__main__":
    main()

# scripts/risk_loop_v3.py
"""
V3対応リスク管理ループ
Legacy + V3の合算でリスク管理
"""

import argparse
import sqlite3
import time
from datetime import datetime, date
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")


class RiskManagerV3:
    """V3対応リスク管理"""

    def __init__(self, db_path: str, symbol: str = "7203.T",
                 daily_loss_limit: float = 50000,
                 max_drawdown_limit: float = 100000):
        self.db_path = db_path
        self.symbol = symbol
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.conn = None

        self.is_halted = False

    def initialize(self):
        """初期化"""
        self.conn = sqlite3.connect(self.db_path, timeout=10_000)

        print(f"\n{'='*60}")
        print(f"V3 Risk Manager Initialized")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Daily Loss Limit: {self.daily_loss_limit:,.0f} JPY")
        print(f"Max Drawdown Limit: {self.max_drawdown_limit:,.0f} JPY")
        print(f"{'='*60}\n")

    def run_cycle(self):
        """1サイクル実行"""
        now = datetime.now(JST)
        today = now.date()

        # P&L取得
        legacy_pnl = self._get_legacy_pnl_today(today)
        v3_pnl = self._get_v3_pnl_today(today)
        total_pnl = legacy_pnl + v3_pnl

        # リスクチェック
        self._check_daily_loss(total_pnl, legacy_pnl, v3_pnl)
        self._check_max_drawdown()

        # ステータス表示
        self._print_status(total_pnl, legacy_pnl, v3_pnl)

    def _get_legacy_pnl_today(self, today: date) -> float:
        """Legacy 当日P&L取得"""
        query = """
        SELECT COALESCE(SUM(pnl), 0.0)
        FROM fills
        WHERE symbol = ? AND date(ts) = ?
        """

        row = self.conn.execute(
            query, (self.symbol, today.isoformat())).fetchone()
        return row[0] if row else 0.0

    def _get_v3_pnl_today(self, today: date) -> float:
        """V3 当日P&L取得"""
        query = """
        SELECT COALESCE(SUM(pnl), 0.0)
        FROM v3_positions
        WHERE symbol = ? AND date(entry_time) = ? AND status = 'CLOSED'
        """

        row = self.conn.execute(
            query, (self.symbol, today.isoformat())).fetchone()
        return row[0] if row else 0.0

    def _check_daily_loss(self, total_pnl: float,
                          legacy_pnl: float, v3_pnl: float):
        """日次損失上限チェック"""
        if total_pnl < -self.daily_loss_limit and not self.is_halted:
            print(f"\n{'='*60}")
            print(f"⚠️  DAILY LOSS LIMIT EXCEEDED")
            print(f"{'='*60}")
            print(f"Total P&L: {total_pnl:,.0f} JPY")
            print(f"  Legacy: {legacy_pnl:,.0f} JPY")
            print(f"  V3: {v3_pnl:,.0f} JPY")
            print(f"Limit: {-self.daily_loss_limit:,.0f} JPY")
            print(f"{'='*60}\n")

            self._halt_trading()
            self.is_halted = True

    def _check_max_drawdown(self):
        """最大ドローダウンチェック"""
        # 簡易実装: 当日の累積P&Lから計算
        today = datetime.now(JST).date()
        legacy_pnl = self._get_legacy_pnl_today(today)
        v3_pnl = self._get_v3_pnl_today(today)
        total_pnl = legacy_pnl + v3_pnl

        if total_pnl < -self.max_drawdown_limit and not self.is_halted:
            print(f"\n{'='*60}")
            print(f"⚠️  MAX DRAWDOWN LIMIT EXCEEDED")
            print(f"{'='*60}")
            print(f"Total P&L: {total_pnl:,.0f} JPY")
            print(f"Limit: {-self.max_drawdown_limit:,.0f} JPY")
            print(f"{'='*60}\n")

            self._halt_trading()
            self.is_halted = True

    def _halt_trading(self):
        """取引停止処理"""
        # 実装: 全ポジションクローズフラグを立てる
        # trader_loop_v3.py が参照するフラグをDBに保存

        self.conn.execute("""
        INSERT OR REPLACE INTO trading_status (symbol, status, updated_at)
        VALUES (?, 'HALTED', ?)
        """, (self.symbol, datetime.now(JST).isoformat()))

        # テーブル作成（初回のみ）
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS trading_status (
            symbol TEXT PRIMARY KEY,
            status TEXT,
            updated_at TEXT
        )
        """)

        self.conn.commit()

        print("🛑 Trading HALTED. All positions should be closed.")

    def _print_status(self, total_pnl: float,
                      legacy_pnl: float, v3_pnl: float):
        """ステータス表示"""
        now = datetime.now(JST)

        status = "🛑 HALTED" if self.is_halted else "✅ OK"

        # リスク余地計算
        loss_headroom = self.daily_loss_limit + total_pnl
        loss_pct = (abs(total_pnl) / self.daily_loss_limit) * \
            100 if total_pnl < 0 else 0

        print(f"[{now.strftime('%H:%M:%S')}] {status} | "
              f"P&L: Legacy={legacy_pnl:+,.0f} V3={v3_pnl:+,.0f} Total={total_pnl:+,.0f} | "
              f"Loss Limit: {loss_pct:.1f}% ({loss_headroom:+,.0f} remaining)")

    def close(self):
        """クリーンアップ"""
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="V3 Risk Manager")
    parser.add_argument("--db", default="runtime.db", help="Database path")
    parser.add_argument("--symbol", default="7203.T", help="Symbol")
    parser.add_argument("--daily-loss-limit", type=float, default=50000,
                        help="Daily loss limit (JPY)")
    parser.add_argument("--max-drawdown-limit", type=float, default=100000,
                        help="Max drawdown limit (JPY)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Check interval (seconds)")

    args = parser.parse_args()

    manager = RiskManagerV3(
        args.db,
        args.symbol,
        args.daily_loss_limit,
        args.max_drawdown_limit
    )
    manager.initialize()

    try:
        while True:
            manager.run_cycle()
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        manager.close()


if __name__ == "__main__":
    main()

# scripts/trader_loop_v3_advanced.py
"""
V3対応トレーディングループ（約定管理版）
- 約定遅延を考慮
- 実際のポジションをリアルタイム同期
- 利益最大化ロジック
"""

from core.execution.order_manager import OrderManager, Order, Position
from tq.io_kabu import create_kabu_client
import argparse
import sqlite3
import time
import sys
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

# scripts/ フォルダをパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


JST = ZoneInfo("Asia/Tokyo")


class AdvancedTradingLoop:
    """約定管理対応トレーディングループ"""

    def __init__(self, db_path: str, symbol: str = "7203.T",
                 use_mock: bool = True, base_size: int = 100,
                 fee_rate: float = 0.0002):
        self.db_path = db_path
        self.symbol = symbol
        self.use_mock = use_mock
        self.base_size = base_size
        self.fee_rate = fee_rate

        # kabuクライアント
        self.kabu = create_kabu_client(use_mock=use_mock, db_path=db_path)

        # 注文管理
        self.order_mgr = OrderManager(self.kabu, db_path, timeout_seconds=30)

        # 日次P&L
        self.legacy_pnl_today = 0.0
        self.v3_pnl_today = 0.0

        # DB接続
        self.conn = None

        # 戦略状態
        self.last_signal_check = datetime.now(JST)
        self.signal_check_interval = timedelta(seconds=5)  # 5秒ごとにシグナル再評価

    def initialize(self):
        """初期化"""
        self.conn = sqlite3.connect(self.db_path, timeout=10_000)
        self.order_mgr.initialize()

        mode = "MOCK" if self.use_mock else "REAL"

        print(f"\n{'='*60}")
        print(f"Advanced Trading Loop ({mode} Mode)")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Base Size: {self.base_size}")
        print(f"Fee Rate: {self.fee_rate*100:.2f}%")
        print(f"Order Timeout: {self.order_mgr.timeout_seconds}秒")
        print(f"{'='*60}\n")

    def run_cycle(self):
        """1サイクル実行"""
        now = datetime.now(JST)

        # 取引時間チェック（09:00-10:00）
        if not self._is_trading_time(now):
            return

        # 取引停止チェック
        if self._is_trading_halted():
            print("[HALTED] Trading is halted by risk manager")
            return

        # === ステップ1: 約定確認 ===
        self.order_mgr.check_all_pending_orders()

        # === ステップ2: 実際のポジション同期 ===
        self.order_mgr.sync_positions()
        actual_position = self.order_mgr.get_actual_position(self.symbol)

        # === ステップ3: 現在価格取得 ===
        try:
            current_price = self.kabu.get_current_price(self.symbol)
        except Exception as e:
            print(f"[ERROR] Failed to get price: {e}")
            return

        # === ステップ4: シグナル評価（定期的） ===
        should_check_signal = (
            now - self.last_signal_check) >= self.signal_check_interval

        if should_check_signal:
            self.last_signal_check = now

            # Legacyシグナル
            legacy_signal = self._get_latest_signal('LEGACY')

            # V3シグナル
            v3_signal = self._get_latest_signal('V3')

            # === ステップ5: 戦略実行 ===
            self._execute_strategy(
                'LEGACY', legacy_signal, actual_position, current_price, now)
            self._execute_strategy(
                'V3', v3_signal, actual_position, current_price, now)

        # === ステップ6: ステータス表示 ===
        self._print_status(current_price, actual_position, now)

    def _execute_strategy(self, system: str, signal: dict,
                          actual_position: Optional[Position],
                          current_price: float, timestamp: datetime):
        """
        戦略実行（利益最大化ロジック）

        ロジック:
        1. ポジションなし → エントリー条件チェック
        2. ポジションあり → エグジット条件 or 更に良い機会を評価
        3. 約定待ち注文あり → シグナル変化をチェック、必要ならキャンセル
        """

        # 約定待ち注文があるか確認
        has_pending = self.order_mgr.has_pending_orders()

        # 実際のポジション状態
        has_position = actual_position is not None and actual_position.size > 0

        if signal is None:
            return

        action = signal.get('action', 'HOLD')
        signal_strength = signal.get('strength', 0.0)

        # === ケース1: ポジションなし、約定待ちなし → 新規エントリー検討 ===
        if not has_position and not has_pending:
            if action == 'ENTRY_LONG' and signal_strength >= 0.15:
                self._enter_position(system, 'LONG', current_price, timestamp)

            elif action == 'ENTRY_SHORT' and signal_strength <= -0.15:
                self._enter_position(system, 'SHORT', current_price, timestamp)

        # === ケース2: 約定待ち注文あり → シグナル変化チェック ===
        elif has_pending and not has_position:
            # シグナルが弱くなった → 注文キャンセル
            if abs(signal_strength) < 0.10:
                print(f"[{system}] Signal weakened, cancelling pending orders...")
                for order_id in list(self.order_mgr.pending_orders.keys()):
                    self.order_mgr.cancel_order(order_id)

        # === ケース3: ポジションあり → エグジット or ホールド ===
        elif has_position:
            position_side = actual_position.side

            # エグジット条件
            should_exit = False

            if position_side == 'LONG':
                # LONGエグジット: シグナルが弱い or 逆転
                if signal_strength <= 0.05 or action == 'ENTRY_SHORT':
                    should_exit = True
                    exit_reason = "weak signal" if signal_strength <= 0.05 else "reversal"

            elif position_side == 'SHORT':
                # SHORTエグジット: シグナルが弱い or 逆転
                if signal_strength >= -0.05 or action == 'ENTRY_LONG':
                    should_exit = True
                    exit_reason = "weak signal" if signal_strength >= -0.05 else "reversal"

            if should_exit:
                print(
                    f"[{system}] Exit signal ({exit_reason}): strength={signal_strength:.3f}")
                self._exit_position(system, actual_position,
                                    current_price, timestamp)

            # ホールド中でも更に良い機会を評価（利益最大化）
            else:
                unrealized_pnl = self._calculate_unrealized_pnl(
                    actual_position, current_price
                )

                # 含み益が十分あり、シグナルが更に強い → 追加エントリー検討
                if unrealized_pnl > 5000 and abs(signal_strength) > 0.20:
                    print(
                        f"[{system}] Strong signal with profit, considering add-on...")
                    # 実装: 追加ポジション取得（オプション）

    def _enter_position(self, system: str, direction: str,
                        price: float, timestamp: datetime):
        """ポジションエントリー"""
        side = "BUY" if direction == "LONG" else "SELL"

        order = self.order_mgr.place_order(
            self.symbol, side, self.base_size, price, "LIMIT"
        )

        if order:
            print(f"[{system}] ENTRY {direction} @ {price:.1f}, waiting for fill...")

    def _exit_position(self, system: str, position: Position,
                       price: float, timestamp: datetime):
        """ポジションエグジット"""
        # 決済方向
        side = "SELL" if position.side == "LONG" else "BUY"

        order = self.order_mgr.place_order(
            self.symbol, side, position.size, price, "LIMIT"
        )

        if order:
            # P&L計算（約定後に確定）
            unrealized_pnl = self._calculate_unrealized_pnl(position, price)

            print(f"[{system}] EXIT {position.side} @ {price:.1f}, "
                  f"Est. P&L: {unrealized_pnl:+,.0f}, waiting for fill...")

    def _calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """含み益計算"""
        if position.side == "LONG":
            gross_pnl = (current_price - position.avg_price) * position.size
        else:
            gross_pnl = (position.avg_price - current_price) * position.size

        # 手数料（往復）
        commission = (position.avg_price + current_price) * \
            position.size * self.fee_rate

        return gross_pnl - commission

    def _get_latest_signal(self, system: str) -> Optional[dict]:
        """最新シグナル取得"""
        if system == 'LEGACY':
            query = """
            SELECT S FROM signals_1m
            WHERE symbol = ?
            ORDER BY ts DESC
            LIMIT 1
            """

            row = self.conn.execute(query, (self.symbol,)).fetchone()
            if not row:
                return None

            S = row[0]

            # アクション判定
            if S >= 0.15:
                action = 'ENTRY_LONG'
            elif S <= -0.15:
                action = 'ENTRY_SHORT'
            elif abs(S) <= 0.05:
                action = 'EXIT'
            else:
                action = 'HOLD'

            return {'action': action, 'strength': S}

        elif system == 'V3':
            query = """
            SELECT V3_action, V3_S FROM signals_1m
            WHERE symbol = ?
            ORDER BY ts DESC
            LIMIT 1
            """

            row = self.conn.execute(query, (self.symbol,)).fetchone()
            if not row or row[0] is None:
                return None

            return {'action': row[0], 'strength': row[1]}

        return None

    def _is_trading_time(self, now: datetime) -> bool:
        """取引時間判定（JST 09:00-10:00）"""
        hour = now.hour
        minute = now.minute

        if hour == 9:
            return True
        elif hour == 10 and minute == 0:
            return True
        else:
            return False

    def _is_trading_halted(self) -> bool:
        """取引停止チェック"""
        query = """
        SELECT status FROM trading_status
        WHERE symbol = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """

        row = self.conn.execute(query, (self.symbol,)).fetchone()

        if row and row[0] == 'HALTED':
            return True

        return False

    def _print_status(self, price: float, position: Optional[Position], timestamp: datetime):
        """ステータス表示"""
        stats = self.order_mgr.get_stats()

        # ポジション状態
        if position and position.size > 0:
            pos_str = f"{position.side} {position.size}株@{position.avg_price:.1f}"
            unrealized = self._calculate_unrealized_pnl(position, price)
            pos_str += f" (含み: {unrealized:+,.0f})"
        else:
            pos_str = "FLAT"

        # 約定待ち
        pending_str = f"{stats['pending_orders']} pending" if stats['pending_orders'] > 0 else "no pending"

        # 日次統計
        total_pnl = self.legacy_pnl_today + self.v3_pnl_today

        print(f"[{timestamp.strftime('%H:%M:%S')}] "
              f"Price: {price:.1f} | "
              f"Position: {pos_str} | "
              f"Orders: {pending_str} | "
              f"P&L Today: {total_pnl:+,.0f}")

    def close(self):
        """クリーンアップ"""
        if self.conn:
            self.conn.close()
        self.order_mgr.close()


def main():
    parser = argparse.ArgumentParser(description="Advanced Trading Loop")
    parser.add_argument("--db", default="runtime.db", help="Database path")
    parser.add_argument("--symbol", default="7203.T", help="Symbol")
    parser.add_argument("--mock", action="store_true",
                        help="Use MOCK mode (default: True)")
    parser.add_argument("--real", action="store_true", help="Use REAL mode")
    parser.add_argument("--interval", type=int, default=5,
                        help="Check interval (seconds)")
    parser.add_argument("--base-size", type=int,
                        default=100, help="Base position size")

    args = parser.parse_args()

    # モード判定（デフォルトはMOCK）
    use_mock = not args.real

    loop = AdvancedTradingLoop(
        args.db,
        args.symbol,
        use_mock=use_mock,
        base_size=args.base_size
    )
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

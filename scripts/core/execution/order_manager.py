# scripts/core/execution/order_manager.py
"""
注文状態管理システム
約定遅延を考慮した注文追跡
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo
import sqlite3

JST = ZoneInfo("Asia/Tokyo")


@dataclass
class Order:
    """注文"""
    order_id: str
    symbol: str
    side: str  # BUY or SELL
    size: int
    price: float
    order_type: str  # LIMIT or MARKET
    status: str = "PENDING"  # PENDING, FILLED, CANCELLED, REJECTED
    created_at: datetime = field(default_factory=lambda: datetime.now(JST))
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_size: int = 0

    def is_filled(self) -> bool:
        """約定済みか"""
        return self.status == "FILLED"

    def is_pending(self) -> bool:
        """約定待ちか"""
        return self.status == "PENDING"

    def time_since_created(self) -> float:
        """注文からの経過時間（秒）"""
        return (datetime.now(JST) - self.created_at).total_seconds()


@dataclass
class Position:
    """実際の保有ポジション"""
    symbol: str
    side: str  # LONG or SHORT
    size: int
    avg_price: float
    unrealized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(JST))


class OrderManager:
    """注文状態管理"""

    def __init__(self, kabu_client, db_path: str, timeout_seconds: int = 30):
        """
        Args:
            kabu_client: kabuステーションクライアント
            db_path: データベースパス
            timeout_seconds: 注文タイムアウト時間
        """
        self.kabu = kabu_client
        self.db_path = db_path
        self.timeout_seconds = timeout_seconds

        # 注文管理
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []

        # 実際のポジション（kabu APIから取得）
        self.actual_positions: Dict[str, Position] = {}

        self.conn = None

    def initialize(self):
        """初期化"""
        self.conn = sqlite3.connect(self.db_path, timeout=10_000)
        self._create_tables()

    def _create_tables(self):
        """テーブル作成"""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS order_log (
            order_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            size INTEGER,
            price REAL,
            order_type TEXT,
            status TEXT,
            created_at TEXT,
            filled_at TEXT,
            filled_price REAL,
            filled_size INTEGER
        )
        """)
        self.conn.commit()

    def place_order(self, symbol: str, side: str, size: int,
                    price: float, order_type: str = "LIMIT") -> Optional[Order]:
        """
        注文発行（非同期前提）

        Returns:
            Order or None（発注失敗時）
        """
        try:
            # kabu APIに注文
            order_id = self.kabu.place_order(
                symbol, side, size, price, order_type)

            # 注文オブジェクト作成
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                size=size,
                price=price,
                order_type=order_type,
                status="PENDING"
            )

            # 管理リストに追加
            self.pending_orders[order_id] = order

            # DB記録
            self._save_order(order)

            print(f"[ORDER] {side} {size}@{price:.1f} placed, ID={order_id}")

            return order

        except Exception as e:
            print(f"[ERROR] Order placement failed: {e}")
            return None

    def check_order_status(self, order_id: str) -> Optional[Order]:
        """
        注文状態確認

        実際のAPIでは kabu.get_order_status(order_id) を使う
        モックでは即座に約定扱い
        """
        if order_id not in self.pending_orders:
            return None

        order = self.pending_orders[order_id]

        # タイムアウトチェック
        if order.time_since_created() > self.timeout_seconds:
            print(f"[TIMEOUT] Order {order_id} timed out, cancelling...")
            self.cancel_order(order_id)
            return order

        # 約定確認（実際のAPIでは get_orders() で確認）
        # モック実装では即座に約定とみなす
        if hasattr(self.kabu, 'get_fills'):
            fills = self.kabu.get_fills()
            for fill in fills:
                if fill['order_id'] == order_id and order.status == "PENDING":
                    order.status = "FILLED"
                    order.filled_at = datetime.now(JST)
                    order.filled_price = fill['fill_price']
                    order.filled_size = fill['size']

                    # 約定済みリストに移動
                    self.filled_orders.append(order)
                    del self.pending_orders[order_id]

                    # DB更新
                    self._update_order(order)

                    print(
                        f"[FILLED] Order {order_id} filled @ {order.filled_price:.1f}")

                    return order

        return order

    def cancel_order(self, order_id: str) -> bool:
        """注文キャンセル"""
        if order_id not in self.pending_orders:
            return False

        try:
            self.kabu.cancel_order(order_id)

            order = self.pending_orders[order_id]
            order.status = "CANCELLED"

            del self.pending_orders[order_id]

            # DB更新
            self._update_order(order)

            print(f"[CANCELLED] Order {order_id}")

            return True

        except Exception as e:
            print(f"[ERROR] Cancel failed: {e}")
            return False

    def sync_positions(self):
        """
        実際のポジションを同期
        kabu APIからポジション取得して更新
        """
        try:
            api_positions = self.kabu.get_positions()

            self.actual_positions.clear()

            for pos_data in api_positions:
                position = Position(
                    symbol=pos_data['symbol'],
                    side=pos_data['side'],
                    size=pos_data['size'],
                    avg_price=pos_data['avg_price'],
                    last_updated=datetime.now(JST)
                )

                self.actual_positions[pos_data['symbol']] = position

            return self.actual_positions

        except Exception as e:
            print(f"[ERROR] Position sync failed: {e}")
            return {}

    def get_actual_position(self, symbol: str) -> Optional[Position]:
        """実際の保有ポジション取得"""
        return self.actual_positions.get(symbol)

    def has_pending_orders(self) -> bool:
        """約定待ち注文があるか"""
        return len(self.pending_orders) > 0

    def check_all_pending_orders(self):
        """すべての約定待ち注文をチェック"""
        order_ids = list(self.pending_orders.keys())

        for order_id in order_ids:
            self.check_order_status(order_id)

    def _save_order(self, order: Order):
        """注文をDBに保存"""
        self.conn.execute("""
        INSERT OR REPLACE INTO order_log 
        (order_id, symbol, side, size, price, order_type, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.order_id, order.symbol, order.side, order.size,
            order.price, order.order_type, order.status,
            order.created_at.isoformat()
        ))
        self.conn.commit()

    def _update_order(self, order: Order):
        """注文状態をDBで更新"""
        self.conn.execute("""
        UPDATE order_log
        SET status = ?, filled_at = ?, filled_price = ?, filled_size = ?
        WHERE order_id = ?
        """, (
            order.status,
            order.filled_at.isoformat() if order.filled_at else None,
            order.filled_price,
            order.filled_size,
            order.order_id
        ))
        self.conn.commit()

    def get_stats(self) -> Dict:
        """統計情報"""
        return {
            'pending_orders': len(self.pending_orders),
            'filled_orders': len(self.filled_orders),
            'actual_positions': len(self.actual_positions),
            'total_filled_today': len([o for o in self.filled_orders
                                       if o.filled_at and o.filled_at.date() == datetime.now(JST).date()])
        }

    def close(self):
        """クリーンアップ"""
        if self.conn:
            self.conn.close()

# tq/io_kabu.py
"""
kabuステーション API実装
本番環境用データ取得・発注
"""

import os
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from typing import Optional, List, Dict

JST = ZoneInfo("Asia/Tokyo")


class KabuAPIError(Exception):
    """kabu API エラー"""
    pass


class KabuSteIO:
    """kabuステーション API クライアント"""

    def __init__(self, api_token: Optional[str] = None,
                 endpoint: str = "http://localhost:18080/kabusapi"):
        """
        Args:
            api_token: APIトークン（環境変数KABU_API_TOKENから取得可能）
            endpoint: kabuステーションのエンドポイント
        """
        self.api_token = api_token or os.environ.get('KABU_API_TOKEN')
        self.endpoint = endpoint
        self.session = requests.Session()

        if not self.api_token:
            raise ValueError("API token not provided")

        # 認証ヘッダー
        self.session.headers.update({
            'X-API-KEY': self.api_token,
            'Content-Type': 'application/json'
        })

    def fetch_bars(self, symbol: str, start: datetime, end: datetime,
                   interval: str = '1min') -> pd.DataFrame:
        """
        1分足データ取得

        Args:
            symbol: 銘柄コード（例: "7203"）
            start: 開始時刻（UTC）
            end: 終了時刻（UTC）
            interval: 足の種類（'1min', '5min', '1day'等）

        Returns:
            pd.DataFrame: OHLCV データ
        """
        # kabuステーションは日本時間で動作
        start_jst = start.astimezone(JST)
        end_jst = end.astimezone(JST)

        # APIエンドポイント（仮想的な実装例）
        url = f"{self.endpoint}/board/{symbol}"

        try:
            # リアルタイム価格取得
            response = self.session.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()

            # 1分足データは別エンドポイントから取得する必要がある
            # ここでは簡易実装として現在価格を返す

            current_price = data.get('CurrentPrice', 0)

            # 仮のOHLCVデータ構築（本番では履歴データAPIを使用）
            bars = pd.DataFrame({
                'open': [current_price],
                'high': [current_price],
                'low': [current_price],
                'close': [current_price],
                'volume': [data.get('TradingVolume', 0)]
            }, index=[datetime.now(JST).astimezone(ZoneInfo('UTC'))])

            return bars

        except requests.exceptions.RequestException as e:
            raise KabuAPIError(f"API request failed: {e}")

    def get_current_price(self, symbol: str) -> float:
        """
        現在価格取得

        Args:
            symbol: 銘柄コード

        Returns:
            float: 現在価格
        """
        url = f"{self.endpoint}/board/{symbol}"

        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()
            return data.get('CurrentPrice', 0.0)

        except requests.exceptions.RequestException as e:
            raise KabuAPIError(f"Failed to get current price: {e}")

    def place_order(self, symbol: str, side: str, size: int,
                    price: float, order_type: str = 'LIMIT') -> str:
        """
        発注

        Args:
            symbol: 銘柄コード
            side: 'BUY' or 'SELL'
            size: 株数
            price: 価格（成行の場合は0）
            order_type: 'LIMIT' or 'MARKET'

        Returns:
            str: 注文ID
        """
        url = f"{self.endpoint}/sendorder"

        # kabuステーション形式の注文データ
        order_data = {
            "Password": self.api_token,  # 実際はパスワード
            "Symbol": symbol,
            "Exchange": 1,  # 1=東証
            "SecurityType": 1,  # 1=株式
            "Side": 1 if side == 'BUY' else 2,
            "CashMargin": 1,  # 1=現物
            "DelivType": 0,  # 0=指定なし
            "AccountType": 2,  # 2=特定
            "Qty": size,
            "FrontOrderType": 10 if order_type == 'LIMIT' else 13,
            "Price": price if order_type == 'LIMIT' else 0,
            "ExpireDay": 0  # 当日
        }

        try:
            response = self.session.post(url, json=order_data, timeout=10)
            response.raise_for_status()

            result = response.json()

            # 注文IDを返す
            return result.get('OrderId', '')

        except requests.exceptions.RequestException as e:
            raise KabuAPIError(f"Order placement failed: {e}")

    def cancel_order(self, order_id: str):
        """
        注文取り消し

        Args:
            order_id: 注文ID
        """
        url = f"{self.endpoint}/cancelorder"

        cancel_data = {
            "OrderId": order_id,
            "Password": self.api_token
        }

        try:
            response = self.session.put(url, json=cancel_data, timeout=5)
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise KabuAPIError(f"Order cancellation failed: {e}")

    def get_positions(self) -> List[Dict]:
        """
        現在のポジション取得

        Returns:
            List[Dict]: ポジション一覧
        """
        url = f"{self.endpoint}/positions"

        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()

            positions = response.json()
            return positions

        except requests.exceptions.RequestException as e:
            raise KabuAPIError(f"Failed to get positions: {e}")

    def get_orders(self) -> List[Dict]:
        """
        注文一覧取得

        Returns:
            List[Dict]: 注文一覧
        """
        url = f"{self.endpoint}/orders"

        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()

            orders = response.json()
            return orders

        except requests.exceptions.RequestException as e:
            raise KabuAPIError(f"Failed to get orders: {e}")


# モック実装（開発・テスト用）
class MockKabuSteIO:
    """kabuステーション APIのモック"""

    def __init__(self, *args, **kwargs):
        self.orders = []
        self.positions = []

    def fetch_bars(self, symbol: str, start: datetime, end: datetime,
                   interval: str = '1min') -> pd.DataFrame:
        """モックデータ返す"""
        import numpy as np

        dates = pd.date_range(start, end, freq='1min', tz='UTC')

        # ランダムなOHLCVデータ
        data = pd.DataFrame({
            'open': np.random.uniform(2400, 2600, len(dates)),
            'high': np.random.uniform(2500, 2700, len(dates)),
            'low': np.random.uniform(2300, 2500, len(dates)),
            'close': np.random.uniform(2400, 2600, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

        return data

    def get_current_price(self, symbol: str) -> float:
        """モック価格"""
        import random
        return random.uniform(2400, 2600)

    def place_order(self, symbol: str, side: str, size: int,
                    price: float, order_type: str = 'LIMIT') -> str:
        """モック発注"""
        order_id = f"MOCK-{len(self.orders)+1:05d}"

        self.orders.append({
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'status': 'PENDING'
        })

        print(f"[MOCK] Order placed: {side} {size}@{price:.1f}, ID={order_id}")
        return order_id

    def cancel_order(self, order_id: str):
        """モック取り消し"""
        print(f"[MOCK] Order cancelled: {order_id}")

    def get_positions(self) -> List[Dict]:
        """モックポジション"""
        return self.positions

    def get_orders(self) -> List[Dict]:
        """モック注文一覧"""
        return self.orders


# 使用例
if __name__ == "__main__":
    # 開発環境ではモック使用
    USE_MOCK = True

    if USE_MOCK:
        client = MockKabuSteIO()
    else:
        client = KabuSteIO()

    # 現在価格取得
    price = client.get_current_price("7203")
    print(f"現在価格: {price:.1f}")

    # 発注
    order_id = client.place_order("7203", "BUY", 100, price, "LIMIT")
    print(f"注文ID: {order_id}")

# ファクトリー関数（追加）


def create_kabu_client(use_mock: bool = True, db_path: str = "runtime.db"):
    """
    kabuクライアント作成

    Args:
        use_mock: Trueならモック、Falseなら本番API
        db_path: モック用DB（デフォルト: runtime.db）

    Returns:
        KabuSteIO or MockKabuSteIO
    """
    # scripts/ フォルダ基準のパス調整
    if use_mock:
        if not os.path.isabs(db_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(os.path.dirname(script_dir), db_path)

        print("[INFO] Using MOCK KabuSte client")
        return MockKabuSteIO(db_path=db_path)
    else:
        print("[INFO] Using REAL KabuSte client")
        return KabuSteIO()

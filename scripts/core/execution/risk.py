# core/execution/risk.py
"""
発注前リスクチェック
"""

import sqlite3
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Optional
import pandas as pd

JST = ZoneInfo("Asia/Tokyo")


class RiskViolation(Exception):
    """リスクチェック違反"""
    pass


class PreTradeRiskCheck:
    """発注前リスクチェック"""
    
    def __init__(self, config: dict, db_path: str):
        self.config = config
        self.db_path = db_path
        self.risk_config = config.get('risk', {})
        self.position_config = config.get('position', {})
    
    def validate_order(self, symbol: str, side: str, size: int, price: float):
        """
        発注前チェック
        
        Args:
            symbol: 銘柄コード
            side: BUY or SELL
            size: 株数
            price: 価格
        
        Raises:
            RiskViolation: リスクチェック違反
        """
        conn = sqlite3.connect(self.db_path)
        
        try:
            # 1. ポジション上限チェック
            self._check_position_limit(conn, symbol, side, size)
            
            # 2. 日次取引回数チェック
            self._check_daily_trade_limit(conn, symbol)
            
            # 3. 取引時間チェック
            self._check_market_hours()
            
            # 4. 価格妥当性チェック
            self._check_price_sanity(conn, symbol, price)
            
            # 5. 日次損失上限チェック
            self._check_daily_loss_limit(conn, symbol)
        
        finally:
            conn.close()
    
    def _check_position_limit(self, conn, symbol: str, side: str, size: int):
        """ポジション上限チェック"""
        max_position = self.position_config.get('max_position', 500)
        
        # 現在のポジション取得
        current_position = self._get_current_position(conn, symbol)
        
        if side == 'BUY':
            new_position = current_position + size
        else:  # SELL
            new_position = current_position - size
        
        if abs(new_position) > max_position:
            raise RiskViolation(
                f"ポジション上限超過: 現在={current_position}, "
                f"新規={new_position}, 上限={max_position}"
            )
    
    def _check_daily_trade_limit(self, conn, symbol: str):
        """日次取引回数チェック"""
        max_trades = self.position_config.get('max_trades_per_day', 15)
        today = datetime.now(JST).date()
        
        # 今日の取引回数
        legacy_trades = conn.execute(
            "SELECT COUNT(*) FROM fills WHERE symbol = ? AND date(ts) = ?",
            (symbol, today.isoformat())
        ).fetchone()[0]
        
        v3_trades = conn.execute(
            "SELECT COUNT(*) FROM v3_positions WHERE symbol = ? AND date(entry_time) = ? AND status = 'CLOSED'",
            (symbol, today.isoformat())
        ).fetchone()[0]
        
        total_trades = legacy_trades + v3_trades
        
        if total_trades >= max_trades:
            raise RiskViolation(
                f"日次取引回数上限: {total_trades}/{max_trades}"
            )
    
    def _check_market_hours(self):
        """取引時間チェック"""
        now = datetime.now(JST)
        hour = now.hour
        minute = now.minute
        
        # JST 09:00-10:00のみ
        if hour == 9:
            return  # OK
        elif hour == 10 and minute == 0:
            return  # OK
        else:
            raise RiskViolation(
                f"取引時間外: {now.strftime('%H:%M')} (許可: 09:00-10:00)"
            )
    
    def _check_price_sanity(self, conn, symbol: str, price: float):
        """価格妥当性チェック"""
        max_deviation = self.risk_config.get('max_price_deviation', 0.05)
        
        # 直近の平均価格を取得
        query = """
        SELECT AVG(close) 
        FROM bars_1m
        WHERE symbol = ? AND ts >= datetime('now', '-30 minutes')
        """
        
        row = conn.execute(query, (symbol,)).fetchone()
        
        if not row or row[0] is None:
            # データがない場合はスキップ
            return
        
        avg_price = row[0]
        deviation = abs(price - avg_price) / avg_price
        
        if deviation > max_deviation:
            raise RiskViolation(
                f"価格異常: {price:.1f} (平均: {avg_price:.1f}, "
                f"乖離率: {deviation:.1%} > {max_deviation:.1%})"
            )
    
    def _check_daily_loss_limit(self, conn, symbol: str):
        """日次損失上限チェック"""
        daily_loss_limit = self.risk_config.get('daily_loss_limit', 50000)
        today = datetime.now(JST).date()
        
        # 今日のP&L
        legacy_pnl = self._get_legacy_pnl(conn, symbol, today)
        v3_pnl = self._get_v3_pnl(conn, symbol, today)
        total_pnl = legacy_pnl + v3_pnl
        
        if total_pnl < -daily_loss_limit:
            raise RiskViolation(
                f"日次損失上限到達: {total_pnl:,.0f} JPY / "
                f"上限: {-daily_loss_limit:,.0f} JPY"
            )
    
    def _get_current_position(self, conn, symbol: str) -> int:
        """現在のポジション（簡易実装）"""
        # Legacy
        legacy_query = """
        SELECT COALESCE(SUM(CASE WHEN side='BUY' THEN size ELSE -size END), 0)
        FROM fills
        WHERE symbol = ? AND date(ts) = ?
        """
        today = datetime.now(JST).date()
        legacy_pos = conn.execute(legacy_query, (symbol, today.isoformat())).fetchone()[0]
        
        # V3
        v3_query = """
        SELECT COALESCE(SUM(CASE WHEN direction='LONG' THEN size ELSE -size END), 0)
        FROM v3_positions
        WHERE symbol = ? AND status = 'OPEN'
        """
        v3_pos = conn.execute(v3_query, (symbol,)).fetchone()[0]
        
        return legacy_pos + v3_pos
    
    def _get_legacy_pnl(self, conn, symbol: str, date: date) -> float:
        """Legacy P&L"""
        query = "SELECT COALESCE(SUM(pnl), 0.0) FROM fills WHERE symbol = ? AND date(ts) = ?"
        row = conn.execute(query, (symbol, date.isoformat())).fetchone()
        return row[0] if row else 0.0
    
    def _get_v3_pnl(self, conn, symbol: str, date: date) -> float:
        """V3 P&L"""
        query = """
        SELECT COALESCE(SUM(pnl), 0.0) 
        FROM v3_positions 
        WHERE symbol = ? AND date(entry_time) = ? AND status = 'CLOSED'
        """
        row = conn.execute(query, (symbol, date.isoformat())).fetchone()
        return row[0] if row else 0.0


# 使用例
if __name__ == "__main__":
    import yaml
    
    with open("config/production.yaml") as f:
        config = yaml.safe_load(f)
    
    risk_check = PreTradeRiskCheck(config, "runtime.db")
    
    try:
        risk_check.validate_order("7203.T", "BUY", 100, 2500.0)
        print("✅ リスクチェック合格")
    except RiskViolation as e:
        print(f"❌ リスクチェック違反: {e}")

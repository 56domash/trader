# monitoring/alerting.py
"""
アラート管理システム
Slack/LINE通知、ヘルスチェック
"""

import os
import sqlite3
from datetime import datetime, date
from typing import Optional
from zoneinfo import ZoneInfo
import requests

JST = ZoneInfo("Asia/Tokyo")


class AlertManager:
    """アラート管理"""
    
    def __init__(self, config: dict):
        self.config = config
        self.slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
        self.line_token = os.environ.get('LINE_TOKEN')
        
        self.enabled = config.get('monitoring', {}).get('enabled', False)
        self.slack_enabled = config.get('monitoring', {}).get('slack', {}).get('enabled', False)
        self.line_enabled = config.get('monitoring', {}).get('line', {}).get('enabled', False)
    
    def send_alert(self, level: str, message: str, details: str = ""):
        """
        アラート送信
        
        Args:
            level: INFO, WARNING, CRITICAL
            message: アラートメッセージ
            details: 詳細情報
        """
        if not self.enabled:
            return
        
        # フォーマット
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        
        if level == "INFO":
            emoji = "✅"
        elif level == "WARNING":
            emoji = "⚠️"
        else:  # CRITICAL
            emoji = "🚨"
        
        formatted_message = f"{emoji} [{level}] {message}\n時刻: {timestamp}"
        if details:
            formatted_message += f"\n詳細: {details}"
        
        # Slack送信
        if self.slack_enabled and self.slack_webhook:
            self._send_to_slack(formatted_message)
        
        # LINE送信
        if self.line_enabled and self.line_token:
            self._send_to_line(formatted_message)
        
        # コンソール出力
        print(formatted_message)
    
    def _send_to_slack(self, message: str):
        """Slack Webhook"""
        try:
            payload = {"text": message}
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Slack送信エラー: {e}")
    
    def _send_to_line(self, message: str):
        """LINE Notify"""
        try:
            headers = {"Authorization": f"Bearer {self.line_token}"}
            payload = {"message": message}
            response = requests.post(
                "https://notify-api.line.me/api/notify",
                headers=headers,
                data=payload,
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            print(f"LINE送信エラー: {e}")
    
    def check_daily_health(self, db_path: str, symbol: str = "7203.T"):
        """日次ヘルスチェック"""
        today = datetime.now(JST).date()
        
        conn = sqlite3.connect(db_path)
        
        try:
            # P&L取得
            legacy_pnl = self._get_legacy_pnl(conn, symbol, today)
            v3_pnl = self._get_v3_pnl(conn, symbol, today)
            total_pnl = legacy_pnl + v3_pnl
            
            # 取引回数
            total_trades = self._get_trade_count(conn, symbol, today)
            
            # 勝率
            win_rate = self._get_win_rate(conn, symbol, today)
            
            # アラート条件チェック
            alerts = self.config.get('monitoring', {}).get('alerts', {})
            
            # 勝率アラート
            if win_rate < alerts.get('win_rate_threshold', 0.30) and total_trades >= 5:
                self.send_alert(
                    "WARNING",
                    f"勝率低下: {win_rate:.1%}",
                    f"取引数: {total_trades}"
                )
            
            # ドローダウンアラート
            daily_loss_limit = self.config.get('risk', {}).get('daily_loss_limit', 50000)
            if total_pnl < 0:
                drawdown_pct = abs(total_pnl) / daily_loss_limit
                threshold = alerts.get('drawdown_threshold_pct', 0.80)
                
                if drawdown_pct > threshold:
                    self.send_alert(
                        "CRITICAL",
                        f"ドローダウン警告: {drawdown_pct:.1%}",
                        f"P&L: {total_pnl:,.0f} JPY / 上限: {daily_loss_limit:,.0f} JPY"
                    )
            
            # 取引なしアラート
            if total_trades == 0:
                self.send_alert(
                    "WARNING",
                    "本日の取引がありません",
                    "システム停止の可能性"
                )
            
            # 日次サマリー（INFO）
            self.send_alert(
                "INFO",
                f"日次サマリー: {today}",
                f"P&L: {total_pnl:+,.0f} JPY (Legacy: {legacy_pnl:+,.0f}, V3: {v3_pnl:+,.0f})\n"
                f"取引数: {total_trades}\n"
                f"勝率: {win_rate:.1%}"
            )
        
        finally:
            conn.close()
    
    def check_feature_health(self, db_path: str, symbol: str = "7203.T"):
        """特徴量のヘルスチェック"""
        today = datetime.now(JST).date()
        
        conn = sqlite3.connect(db_path)
        
        try:
            # 特徴量のNaN率チェック
            query = """
            SELECT feature_name, 
                   COUNT(*) as total,
                   SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) as null_count
            FROM feature_values
            WHERE symbol = ? AND date(ts) = ?
            GROUP BY feature_name
            """
            
            import pandas as pd
            df = pd.read_sql(query, conn, params=(symbol, today.isoformat()))
            
            if df.empty:
                return
            
            df['nan_ratio'] = df['null_count'] / df['total']
            
            threshold = self.config.get('monitoring', {}).get('alerts', {}).get(
                'nan_feature_threshold', 0.30
            )
            
            # NaN率が高い特徴量
            problematic = df[df['nan_ratio'] > threshold]
            
            if not problematic.empty:
                features = problematic['feature_name'].tolist()
                self.send_alert(
                    "WARNING",
                    f"特徴量のNaN率が高い: {len(features)}個",
                    f"該当: {', '.join(features)}"
                )
        
        finally:
            conn.close()
    
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
    
    def _get_trade_count(self, conn, symbol: str, date: date) -> int:
        """取引回数"""
        legacy_count = conn.execute(
            "SELECT COUNT(*) FROM fills WHERE symbol = ? AND date(ts) = ?",
            (symbol, date.isoformat())
        ).fetchone()[0]
        
        v3_count = conn.execute(
            "SELECT COUNT(*) FROM v3_positions WHERE symbol = ? AND date(entry_time) = ? AND status = 'CLOSED'",
            (symbol, date.isoformat())
        ).fetchone()[0]
        
        return legacy_count + v3_count
    
    def _get_win_rate(self, conn, symbol: str, date: date) -> float:
        """勝率"""
        legacy_query = "SELECT COUNT(*) FROM fills WHERE symbol = ? AND date(ts) = ? AND pnl > 0"
        legacy_wins = conn.execute(legacy_query, (symbol, date.isoformat())).fetchone()[0]
        
        v3_query = "SELECT COUNT(*) FROM v3_positions WHERE symbol = ? AND date(entry_time) = ? AND status = 'CLOSED' AND pnl > 0"
        v3_wins = conn.execute(v3_query, (symbol, date.isoformat())).fetchone()[0]
        
        total_trades = self._get_trade_count(conn, symbol, date)
        
        if total_trades == 0:
            return 0.0
        
        return (legacy_wins + v3_wins) / total_trades


# 使用例
if __name__ == "__main__":
    import yaml
    
    # 設定読み込み
    with open("config/production.yaml") as f:
        config = yaml.safe_load(f)
    
    alert_mgr = AlertManager(config)
    
    # ヘルスチェック
    alert_mgr.check_daily_health("runtime.db")
    alert_mgr.check_feature_health("runtime.db")

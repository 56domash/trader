# scripts/backtest_v3.py
"""
Toyota Trading System V3 - Backtesting Framework
過去データでV3システムと既存システムの性能を比較
"""

import argparse
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

# V3システムのインポート
from strategy_loop_v3 import run_v3_strategy, jst_window_utc

JST = ZoneInfo("Asia/Tokyo")


@dataclass
class Trade:
    """取引記録"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: str = "LONG"  # LONG or SHORT
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: int = 100
    pnl: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED
    system: str = "V3"  # V3 or LEGACY


@dataclass
class BacktestResult:
    """バックテスト結果"""
    system: str
    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    def calculate_stats(self):
        """統計計算"""
        if not self.trades:
            return

        self.total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t.status == "CLOSED"]

        if not closed_trades:
            return

        pnls = [t.pnl for t in closed_trades]
        self.total_pnl = sum(pnls)

        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        self.winning_trades = len(winning)
        self.losing_trades = len(losing)
        self.win_rate = self.winning_trades / \
            len(closed_trades) if closed_trades else 0

        self.avg_win = np.mean(winning) if winning else 0
        self.avg_loss = np.mean(losing) if losing else 0

        # Sharpe ratio (簡易版)
        if len(pnls) > 1:
            returns = np.array(pnls)
            self.sharpe_ratio = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        self.max_drawdown = drawdown.min()


class BacktestEngine:
    """バックテストエンジン"""

    def __init__(self, db_path: str, symbol: str = "7203.T"):
        self.db_path = db_path
        self.symbol = symbol
        self.fee_rate = 0.0002  # 0.02%

    def get_trading_dates(self, days: int = 7) -> List[datetime.date]:
        """過去N日間の取引日を取得"""
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT DISTINCT date(ts) as trade_date
        FROM bars_1m
        WHERE symbol = ?
        ORDER BY trade_date DESC
        LIMIT ?
        """

        df = pd.read_sql(query, conn, params=(self.symbol, days))
        conn.close()

        dates = [datetime.strptime(d, "%Y-%m-%d").date()
                 for d in df['trade_date']]
        return sorted(dates)

    def get_bars_for_date(self, conn: sqlite3.Connection,
                          target_date: datetime.date) -> pd.DataFrame:
        """指定日のバーデータ取得"""
        start_utc, end_utc = jst_window_utc(target_date, "09:00", "10:00")

        query = """
        SELECT ts, open, high, low, close, volume
        FROM bars_1m
        WHERE symbol = ? AND ts >= ? AND ts < ?
        ORDER BY ts
        """

        df = pd.read_sql(
            query, conn,
            params=(self.symbol, start_utc.isoformat(), end_utc.isoformat()),
            parse_dates=['ts']
        )

        if not df.empty:
            df.set_index('ts', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)

        return df

    def get_v3_signals(self, conn: sqlite3.Connection,
                       target_date: datetime.date) -> pd.DataFrame:
        """V3シグナル取得"""
        start_utc, end_utc = jst_window_utc(target_date, "09:00", "10:00")

        query = """
        SELECT ts, V3_S, V3_S_ema, V3_action
        FROM signals_1m
        WHERE symbol = ? AND ts >= ? AND ts < ?
        ORDER BY ts
        """

        df = pd.read_sql(
            query, conn,
            params=(self.symbol, start_utc.isoformat(), end_utc.isoformat()),
            parse_dates=['ts']
        )

        if not df.empty:
            df.set_index('ts', inplace=True)

        return df

    def get_legacy_signals(self, conn: sqlite3.Connection,
                           target_date: datetime.date) -> pd.DataFrame:
        """既存システムのシグナル取得"""
        start_utc, end_utc = jst_window_utc(target_date, "09:00", "10:00")

        query = """
        SELECT ts, S, S_buy, S_sell
        FROM signals_1m
        WHERE symbol = ? AND ts >= ? AND ts < ?
        ORDER BY ts
        """

        df = pd.read_sql(
            query, conn,
            params=(self.symbol, start_utc.isoformat(), end_utc.isoformat()),
            parse_dates=['ts']
        )

        if not df.empty:
            df.set_index('ts', inplace=True)

        return df

    def simulate_v3_trading(self, bars: pd.DataFrame,
                            signals: pd.DataFrame) -> List[Trade]:
        """V3シグナルで仮想取引"""
        trades = []
        current_position = None

        for ts, row in signals.iterrows():
            if ts not in bars.index:
                continue

            action = row.get('V3_action', 'HOLD')
            price = bars.loc[ts, 'close']

            # エントリー
            if action == 'ENTRY_LONG' and current_position is None:
                current_position = Trade(
                    entry_time=ts,
                    direction='LONG',
                    entry_price=price,
                    size=100,
                    system='V3'
                )

            elif action == 'ENTRY_SHORT' and current_position is None:
                current_position = Trade(
                    entry_time=ts,
                    direction='SHORT',
                    entry_price=price,
                    size=100,
                    system='V3'
                )

            # エグジット
            elif action == 'EXIT' and current_position is not None:
                current_position.exit_time = ts
                current_position.exit_price = price
                current_position.status = 'CLOSED'

                # P&L計算
                if current_position.direction == 'LONG':
                    gross_pnl = (
                        price - current_position.entry_price) * current_position.size
                else:
                    gross_pnl = (current_position.entry_price -
                                 price) * current_position.size

                # 手数料
                entry_fee = current_position.entry_price * current_position.size * self.fee_rate
                exit_fee = price * current_position.size * self.fee_rate

                current_position.pnl = gross_pnl - entry_fee - exit_fee
                trades.append(current_position)
                current_position = None

        # 未決済ポジションをクローズ（最終価格で）
        if current_position is not None:
            last_price = bars.iloc[-1]['close']
            current_position.exit_time = bars.index[-1]
            current_position.exit_price = last_price
            current_position.status = 'CLOSED'

            if current_position.direction == 'LONG':
                gross_pnl = (
                    last_price - current_position.entry_price) * current_position.size
            else:
                gross_pnl = (current_position.entry_price -
                             last_price) * current_position.size

            entry_fee = current_position.entry_price * current_position.size * self.fee_rate
            exit_fee = last_price * current_position.size * self.fee_rate

            current_position.pnl = gross_pnl - entry_fee - exit_fee
            trades.append(current_position)

        return trades

    def simulate_legacy_trading(self, bars: pd.DataFrame,
                                signals: pd.DataFrame,
                                threshold_long: float = 0.15,
                                threshold_short: float = -0.15,
                                exit_threshold: float = 0.05) -> List[Trade]:
        """既存システムで仮想取引"""
        trades = []
        current_position = None

        for ts, row in signals.iterrows():
            if ts not in bars.index:
                continue

            S = row.get('S', None)
            if S is None:
                continue  # S が None の場合はスキップ

            price = bars.loc[ts, 'close']

            # エントリー判定
            if current_position is None:
                if S >= threshold_long:
                    current_position = Trade(
                        entry_time=ts,
                        direction='LONG',
                        entry_price=price,
                        size=100,
                        system='LEGACY'
                    )
                elif S <= threshold_short:
                    current_position = Trade(
                        entry_time=ts,
                        direction='SHORT',
                        entry_price=price,
                        size=100,
                        system='LEGACY'
                    )

            # エグジット判定
            elif current_position is not None:
                should_exit = False

                if current_position.direction == 'LONG' and S <= exit_threshold:
                    should_exit = True
                elif current_position.direction == 'SHORT' and S >= -exit_threshold:
                    should_exit = True

                if should_exit:
                    current_position.exit_time = ts
                    current_position.exit_price = price
                    current_position.status = 'CLOSED'

                    if current_position.direction == 'LONG':
                        gross_pnl = (
                            price - current_position.entry_price) * current_position.size
                    else:
                        gross_pnl = (current_position.entry_price -
                                     price) * current_position.size

                    entry_fee = current_position.entry_price * current_position.size * self.fee_rate
                    exit_fee = price * current_position.size * self.fee_rate

                    current_position.pnl = gross_pnl - entry_fee - exit_fee
                    trades.append(current_position)
                    current_position = None

        # 未決済をクローズ
        if current_position is not None:
            last_price = bars.iloc[-1]['close']
            current_position.exit_time = bars.index[-1]
            current_position.exit_price = last_price
            current_position.status = 'CLOSED'

            if current_position.direction == 'LONG':
                gross_pnl = (
                    last_price - current_position.entry_price) * current_position.size
            else:
                gross_pnl = (current_position.entry_price -
                             last_price) * current_position.size

            entry_fee = current_position.entry_price * current_position.size * self.fee_rate
            exit_fee = last_price * current_position.size * self.fee_rate

            current_position.pnl = gross_pnl - entry_fee - exit_fee
            trades.append(current_position)

        return trades

    def run_backtest(self, days: int = 7, regenerate_v3: bool = False) -> Dict[str, BacktestResult]:
        """バックテスト実行"""
        trading_dates = self.get_trading_dates(days)

        if not trading_dates:
            print("取引日が見つかりません")
            return {}

        print(f"\n{'='*70}")
        print(
            f"Backtesting: {trading_dates[0]} ~ {trading_dates[-1]} ({len(trading_dates)} days)")
        print(f"{'='*70}\n")

        v3_result = BacktestResult(system="V3")
        legacy_result = BacktestResult(system="LEGACY")

        conn = sqlite3.connect(self.db_path, timeout=10_000)

        try:
            for date in trading_dates:
                print(f"処理中: {date}")

                # V3シグナル生成（必要に応じて）
                if regenerate_v3:
                    run_v3_strategy(self.db_path, self.symbol,
                                    date, verbose=False)

                # データ取得
                bars = self.get_bars_for_date(conn, date)
                if bars.empty:
                    print(f"  ⚠️  データなし")
                    continue

                # V3バックテスト
                v3_signals = self.get_v3_signals(conn, date)
                if not v3_signals.empty:
                    v3_trades = self.simulate_v3_trading(bars, v3_signals)
                    v3_result.trades.extend(v3_trades)
                    print(f"  V3: {len(v3_trades)} trades")

                # Legacyバックテスト
                legacy_signals = self.get_legacy_signals(conn, date)
                if not legacy_signals.empty:
                    legacy_trades = self.simulate_legacy_trading(
                        bars, legacy_signals)
                    legacy_result.trades.extend(legacy_trades)
                    print(f"  Legacy: {len(legacy_trades)} trades")

        finally:
            conn.close()

        # 統計計算
        v3_result.calculate_stats()
        legacy_result.calculate_stats()

        return {
            "V3": v3_result,
            "LEGACY": legacy_result
        }

    def print_report(self, results: Dict[str, BacktestResult]):
        """レポート出力"""
        print(f"\n{'='*70}")
        print(f"Backtest Results")
        print(f"{'='*70}\n")

        for system, result in results.items():
            print(f"{system} System:")
            print(f"  Total Trades: {result.total_trades}")
            print(
                f"  Winning: {result.winning_trades} | Losing: {result.losing_trades}")
            print(f"  Win Rate: {result.win_rate*100:.1f}%")
            print(f"  Total P&L: {result.total_pnl:,.0f} JPY")
            print(f"  Avg Win: {result.avg_win:,.0f} JPY")
            print(f"  Avg Loss: {result.avg_loss:,.0f} JPY")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {result.max_drawdown:,.0f} JPY")
            print()

        # 比較
        if "V3" in results and "LEGACY" in results:
            v3 = results["V3"]
            legacy = results["LEGACY"]

            print("Comparison (V3 vs LEGACY):")
            print(
                f"  P&L Difference: {v3.total_pnl - legacy.total_pnl:+,.0f} JPY")
            print(
                f"  Win Rate Diff: {(v3.win_rate - legacy.win_rate)*100:+.1f}%")
            print(
                f"  Sharpe Diff: {v3.sharpe_ratio - legacy.sharpe_ratio:+.2f}")
            print()


def main():
    parser = argparse.ArgumentParser(description="V3 Backtest")
    parser.add_argument("--db", default="runtime.db", help="Database path")
    parser.add_argument("--symbol", default="7203.T", help="Symbol")
    parser.add_argument("--days", type=int, default=7,
                        help="Backtest period (days)")
    parser.add_argument("--regenerate-v3", action="store_true",
                        help="Regenerate V3 signals")
    parser.add_argument("--export-csv", help="Export trades to CSV")

    args = parser.parse_args()

    engine = BacktestEngine(args.db, args.symbol)
    results = engine.run_backtest(args.days, args.regenerate_v3)
    engine.print_report(results)

    # CSV出力
    if args.export_csv and results:
        all_trades = []
        for system, result in results.items():
            for trade in result.trades:
                all_trades.append({
                    'system': system,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'pnl': trade.pnl
                })

        df = pd.DataFrame(all_trades)
        df.to_csv(args.export_csv, index=False)
        print(f"\nTrades exported to {args.export_csv}")


if __name__ == "__main__":
    main()

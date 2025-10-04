# scripts/daily_report.py
"""
日次レポート生成
"""

import argparse
import sqlite3
from datetime import datetime, date
from zoneinfo import ZoneInfo
import pandas as pd

JST = ZoneInfo("Asia/Tokyo")


def generate_daily_report(db_path: str, target_date: date, symbol: str = "7203.T"):
    """日次レポート生成"""
    
    conn = sqlite3.connect(db_path)
    
    try:
        print(f"\n{'='*70}")
        print(f"日次トレーディングレポート: {target_date}")
        print(f"{'='*70}\n")
        
        # === 1. P&L サマリー ===
        print("📊 P&L サマリー")
        print("-" * 70)
        
        legacy_pnl = _get_legacy_pnl(conn, symbol, target_date)
        v3_pnl = _get_v3_pnl(conn, symbol, target_date)
        total_pnl = legacy_pnl + v3_pnl
        
        print(f"  Legacy P&L:  {legacy_pnl:>12,.0f} JPY")
        print(f"  V3 P&L:      {v3_pnl:>12,.0f} JPY")
        print(f"  {'─'*40}")
        print(f"  Total P&L:   {total_pnl:>12,.0f} JPY")
        print()
        
        # === 2. トレード統計 ===
        print("📈 トレード統計")
        print("-" * 70)
        
        legacy_trades = _get_legacy_trades(conn, symbol, target_date)
        v3_trades = _get_v3_trades(conn, symbol, target_date)
        
        legacy_stats = _calculate_stats(legacy_trades)
        v3_stats = _calculate_stats(v3_trades)
        
        print(f"  {'Metric':<20} {'Legacy':>12} {'V3':>12} {'Total':>12}")
        print(f"  {'-'*60}")
        print(f"  {'Total Trades':<20} {legacy_stats['count']:>12} {v3_stats['count']:>12} {legacy_stats['count']+v3_stats['count']:>12}")
        print(f"  {'Winning Trades':<20} {legacy_stats['wins']:>12} {v3_stats['wins']:>12} {legacy_stats['wins']+v3_stats['wins']:>12}")
        print(f"  {'Losing Trades':<20} {legacy_stats['losses']:>12} {v3_stats['losses']:>12} {legacy_stats['losses']+v3_stats['losses']:>12}")
        print(f"  {'Win Rate':<20} {legacy_stats['win_rate']:>11.1%} {v3_stats['win_rate']:>11.1%} {_combined_win_rate(legacy_stats, v3_stats):>11.1%}")
        print(f"  {'Avg Win':<20} {legacy_stats['avg_win']:>11,.0f} {v3_stats['avg_win']:>11,.0f}")
        print(f"  {'Avg Loss':<20} {legacy_stats['avg_loss']:>11,.0f} {v3_stats['avg_loss']:>11,.0f}")
        print()
        
        # === 3. 特徴量寄与度（V3のみ） ===
        if v3_stats['count'] > 0:
            print("🔍 特徴量寄与度（V3）")
            print("-" * 70)
            
            contrib_df = _get_feature_contributions(conn, symbol, target_date)
            
            if not contrib_df.empty:
                top_5 = contrib_df.head(5)
                
                for _, row in top_5.iterrows():
                    name = row['feature_name']
                    avg = row['avg_contrib']
                    max_val = row['max_contrib']
                    
                    if avg > 0:
                        bias = "🟢 BUY "
                    elif avg < 0:
                        bias = "🔴 SELL"
                    else:
                        bias = "⚪ NEUTRAL"
                    
                    print(f"  {name:<25s} {bias} avg={avg:+.4f} max={max_val:.4f}")
            print()
        
        # === 4. 警告事項 ===
        warnings = []
        
        if legacy_stats['win_rate'] < 0.40 and legacy_stats['count'] >= 5:
            warnings.append(f"⚠️  Legacy勝率低下: {legacy_stats['win_rate']:.1%}")
        
        if v3_stats['win_rate'] < 0.40 and v3_stats['count'] >= 5:
            warnings.append(f"⚠️  V3勝率低下: {v3_stats['win_rate']:.1%}")
        
        if total_pnl < -10000:
            warnings.append(f"⚠️  損失が大きい: {total_pnl:,.0f} JPY")
        
        if legacy_stats['count'] + v3_stats['count'] == 0:
            warnings.append(f"⚠️  取引なし")
        
        if warnings:
            print("⚠️  警告事項")
            print("-" * 70)
            for w in warnings:
                print(f"  {w}")
            print()
        
        print(f"{'='*70}\n")
        
        # 結果を辞書で返す
        return {
            'date': target_date,
            'total_pnl': total_pnl,
            'legacy_pnl': legacy_pnl,
            'v3_pnl': v3_pnl,
            'total_trades': legacy_stats['count'] + v3_stats['count'],
            'win_rate': _combined_win_rate(legacy_stats, v3_stats),
            'warnings': warnings
        }
    
    finally:
        conn.close()


def _get_legacy_pnl(conn, symbol: str, date: date) -> float:
    query = "SELECT COALESCE(SUM(pnl), 0.0) FROM fills WHERE symbol = ? AND date(ts) = ?"
    row = conn.execute(query, (symbol, date.isoformat())).fetchone()
    return row[0] if row else 0.0


def _get_v3_pnl(conn, symbol: str, date: date) -> float:
    query = """
    SELECT COALESCE(SUM(pnl), 0.0) 
    FROM v3_positions 
    WHERE symbol = ? AND date(entry_time) = ? AND status = 'CLOSED'
    """
    row = conn.execute(query, (symbol, date.isoformat())).fetchone()
    return row[0] if row else 0.0


def _get_legacy_trades(conn, symbol: str, date: date) -> pd.DataFrame:
    query = "SELECT pnl FROM fills WHERE symbol = ? AND date(ts) = ?"
    return pd.read_sql(query, conn, params=(symbol, date.isoformat()))


def _get_v3_trades(conn, symbol: str, date: date) -> pd.DataFrame:
    query = """
    SELECT pnl FROM v3_positions 
    WHERE symbol = ? AND date(entry_time) = ? AND status = 'CLOSED'
    """
    return pd.read_sql(query, conn, params=(symbol, date.isoformat()))


def _calculate_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            'count': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
    
    pnls = trades_df['pnl']
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    
    return {
        'count': len(pnls),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(pnls) if len(pnls) > 0 else 0.0,
        'avg_win': wins.mean() if len(wins) > 0 else 0.0,
        'avg_loss': losses.mean() if len(losses) > 0 else 0.0
    }


def _combined_win_rate(legacy_stats: dict, v3_stats: dict) -> float:
    total_wins = legacy_stats['wins'] + v3_stats['wins']
    total_trades = legacy_stats['count'] + v3_stats['count']
    return total_wins / total_trades if total_trades > 0 else 0.0


def _get_feature_contributions(conn, symbol: str, date: date) -> pd.DataFrame:
    query = """
    SELECT 
        feature_name,
        AVG(contribution) as avg_contrib,
        MAX(ABS(contribution)) as max_contrib
    FROM feature_contributions
    WHERE symbol = ? AND date(ts) = ?
    GROUP BY feature_name
    ORDER BY ABS(avg_contrib) DESC
    LIMIT 10
    """
    
    return pd.read_sql(query, conn, params=(symbol, date.isoformat()))


def main():
    parser = argparse.ArgumentParser(description="Daily Report")
    parser.add_argument("--db", default="runtime.db", help="Database path")
    parser.add_argument("--symbol", default="7203.T", help="Symbol")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD), default=today")
    
    args = parser.parse_args()
    
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = datetime.now(JST).date()
    
    generate_daily_report(args.db, target_date, args.symbol)


if __name__ == "__main__":
    main()

import optuna
import sqlite3
from datetime import datetime
from scripts.trader_loop import run_trader, Config
from optuna.pruners import MedianPruner

DB_PATH = "runtime.db"

# 固定の日付リスト（例）
DATES = [
    datetime(2025, 9, 15),
    datetime(2025, 9, 16),
    datetime(2025, 9, 17),
    datetime(2025, 9, 18),
    datetime(2025, 9, 19),
    datetime(2025, 9, 22),
]


def objective(trial):
    cfg = Config()
    cfg.db_path = DB_PATH
    cfg.symbol = "7203.T"

    # Optuna探索パラメータ
    cfg.thr_long = trial.suggest_float("thr_long", 0.01, 0.5)
    cfg.thr_short = trial.suggest_float("thr_short", -1.0, -0.01)
    cfg.exit_long = trial.suggest_float("exit_long", 0.01, 0.3)
    cfg.exit_short = trial.suggest_float("exit_short", -0.3, -0.01)
    cfg.ema_span = trial.suggest_int("ema_span", 2, 20)
    cfg.confirm_bars = trial.suggest_int("confirm_bars", 1, 3)
    cfg.min_edge = trial.suggest_float("min_edge", 0.0, 0.05)

    conn = sqlite3.connect(DB_PATH)

    total_pnl = 0.0
    for tgt in DATES:
        pnl = run_trader(conn, cfg, tgt, verbose=False) or 0.0
        # if pnl is None or pnl == 0:
        #     pnl = -1000  # 取引ゼロは損扱い

        total_pnl += pnl
        status = "✅目標達成" if pnl >= 25000 else "❌未達"
        print(f"[LOG] {tgt.strftime('%Y-%m-%d')} | PnL={pnl:,.0f} ({status})")

    conn.close()

    print(f"[LOG] 合計PnL={total_pnl:,.0f} over {len(DATES)} days")
    return total_pnl


if __name__ == "__main__":
    # study = optuna.create_study(direction="maximize")
    study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    study.optimize(objective, n_trials=1000)

    print("Best params:", study.best_params)
    print("Best total PnL:", study.best_value)

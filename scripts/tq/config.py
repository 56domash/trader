
from __future__ import annotations
import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    db_path: str = "./runtime.db"
    data_dir: str = "./toyota_backtest"
    symbol: str = "7203.T"
    jst_start: str = "09:00"
    jst_end: str = "10:00"
    thr_long: float = 0.6
    thr_short: float = -0.6
    exit_long: float = -0.2
    exit_short: float = 0.2
    ttl_min: int = 8
    lot_size: int = 100
    fee_rate: float = 0.0002
    ingest_period: str = "1d"
    use_yfinance: bool = True

def load(path: str|None) -> Settings:
    s = Settings()
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(s, k): setattr(s, k, v)
    return s

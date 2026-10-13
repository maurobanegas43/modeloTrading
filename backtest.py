from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data import utc_iso


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    allocation: float
    pnl: float
    reason: str


def simulate(candles: pd.DataFrame, probabilities: np.ndarray, threshold: float, cfg: StrategyConfig) -> dict:
    """Chronological, long-only, single-position, next-open simulated fills.

    Every prediction is made after bar i closes. Buying cannot occur until bar
    i+1 opens. Stops check each subsequent bar's low (conservative gap fill).
    Equity is marked each minute. No exchange order is ever sent.
    """
    if len(candles) != len(probabilities):
        raise ValueError("Una probabilidad por vela es obligatoria")
    cfg.validate()
    if len(candles) == 0:
        return {"trades": 0, "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "win_rate": 0.0, "profit_factor": None, "closed_trades": []}
    data = candles.reset_index(drop=True)
    probs = np.asarray(probabilities, dtype=float)
    cash = cfg.initial_cash
    qty, entry_price, allocation = 0.0, 0.0, 0.0
    entry_time = -1
    scheduled = -1
    peak = cfg.initial_cash
    max_dd = 0.0
    day = ""
    day_start = cfg.initial_cash
    halted_today = False
    halt_forever = False
    trades: list[Trade] = []

    def sell(i: int, raw_price: float, reason: str) -> None:
        nonlocal cash, qty, entry_price, entry_time, allocation
        exit_price = raw_price * (1 - cfg.execution_bps / 10_000)
        proceeds = qty * exit_price * (1 - cfg.fee_rate)
        pnl = proceeds - allocation
        cash += proceeds
        trades.append(Trade(utc_iso(int(data.iloc[entry_time]["open_time"])),
                            utc_iso(int(data.iloc[i]["open_time"])),
                            float(entry_price), float(exit_price), float(allocation), float(pnl), reason))
        qty, entry_price, allocation, entry_time = 0.0, 0.0, 0.0, -1

    for i in range(len(data)):
        bar = data.iloc[i]
        if i > 0 and int(bar["open_time"]) - int(data.iloc[i - 1]["open_time"]) != 60_000:
            # A market-data gap must never turn a 15-minute position into a
            # position held for an unknowable amount of wall-clock time.
            scheduled = -1
            if qty > 0:
                sell(i, float(bar["open"]), "missing_data_gap")
        bar_day = utc_iso(int(bar["open_time"]))[:10]
        if bar_day != day:
            day, day_start, halted_today = bar_day, cash + qty * float(bar["open"]) * (1 - cfg.fee_rate), False

        if scheduled == i and qty == 0 and not halted_today and not halt_forever:
            allocation = cash * cfg.allocation
            entry_price = float(bar["open"]) * (1 + cfg.execution_bps / 10_000)
            qty = allocation * (1 - cfg.fee_rate) / entry_price
            cash -= allocation
            entry_time = i
        scheduled = -1

        if qty > 0:
            stop_price = entry_price * (1 - cfg.stop_loss)
            if float(bar["low"]) <= stop_price:
                # A gap below the stop is filled at the lower open, not optimistically at the stop.
                sell(i, min(float(bar["open"]), stop_price), "stop_loss")
            elif i - entry_time + 1 >= cfg.horizon_bars:
                sell(i, float(bar["close"]), "horizon")

        equity = cash + qty * float(bar["close"]) * (1 - cfg.execution_bps / 10_000) * (1 - cfg.fee_rate)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
        if equity <= day_start * (1 - cfg.daily_loss_limit):
            halted_today = True
            if qty > 0:
                sell(i, float(bar["close"]), "daily_loss_limit")
        if dd >= cfg.max_drawdown:
            halt_forever = True
            if qty > 0:
                sell(i, float(bar["close"]), "max_drawdown")

        # Signal for NEXT bar's opening; no same-bar execution.
        if (i + 1 < len(data) and qty == 0 and not halted_today and not halt_forever
                and int(data.iloc[i + 1]["open_time"]) - int(bar["open_time"]) == 60_000):
            if np.isfinite(probs[i]) and probs[i] >= threshold:
                scheduled = i + 1

    if qty > 0:
        sell(len(data) - 1, float(data.iloc[-1]["close"]), "end_of_backtest")

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [-t.pnl for t in trades if t.pnl < 0]
    gross_loss = sum(losses)
    return {
        "trades": len(trades),
        "total_return_pct": round(100 * (cash / cfg.initial_cash - 1), 4),
        "final_equity_usdt": round(cash, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
        "profit_factor": round(sum(wins) / gross_loss, 4) if gross_loss else None,
        "closed_trades": [asdict(t) for t in trades],
    }


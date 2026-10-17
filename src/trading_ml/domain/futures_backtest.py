"""Simulación de posiciones futures long/short con costes, funding y margen."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .config import FuturesConfig
from .features import interval_decision_mask
from .futures_wallet import DirectionalFuturesWallet
from .models import DirectionalFuturesModel


def simulate_futures(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    funding: pd.DataFrame | None,
    cfg: FuturesConfig,
    probability_threshold: float | None = None,
) -> dict:
    cfg.validate()
    if len(frame) != len(probabilities):
        raise ValueError("Se requiere una fila de probabilidades por vela")
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probabilities debe tener columnas SHORT, FLAT, LONG")
    threshold = cfg.probability_threshold if probability_threshold is None else probability_threshold
    n = len(frame)
    if n == 0:
        return {
            "trades": 0,
            "total_return_pct": 0.0,
            "final_equity_usdt": cfg.initial_balance,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": None,
            "closed_trades": [],
            "action_counts": {},
        }

    trade_events: list[dict] = []
    event_counts: Counter[str] = Counter()

    def sink(event: dict) -> None:
        event_counts[str(event.get("type", "unknown"))] += 1
        if str(event.get("type", "")).startswith("PAPER_CLOSE_"):
            trade_events.append(event)

    wallet = DirectionalFuturesWallet(cfg, event_sink=sink)
    times = frame["open_time"].to_numpy(dtype=np.int64)
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    mark_highs = frame["mark_high"].to_numpy(dtype=float) if "mark_high" in frame else highs
    mark_lows = frame["mark_low"].to_numpy(dtype=float) if "mark_low" in frame else lows
    mark_closes = frame["mark_close"].to_numpy(dtype=float) if "mark_close" in frame else closes
    decision_mask = interval_decision_mask(frame, cfg.decision_interval_minutes)
    funding_by_time: dict[int, list[tuple[float, float]]] = {}
    if funding is not None and len(funding):
        for row in funding.itertuples(index=False):
            event_time = int(getattr(row, "fundingTime"))
            rate = float(getattr(row, "fundingRate"))
            mark = getattr(row, "markPrice", np.nan)
            funding_by_time.setdefault(event_time, []).append((rate, float(mark) if pd.notna(mark) else 0.0))

    equity_curve = np.empty(n, dtype=float)
    pending_action: int | None = None
    action_counts: Counter[str] = Counter()
    skipped = 0
    for i in range(n):
        action_to_apply = pending_action
        pending_action = None
        funding_events = funding_by_time.get(int(times[i]), [])
        note = wallet.on_bar(
            int(times[i]),
            float(opens[i]),
            float(highs[i]),
            float(lows[i]),
            float(closes[i]),
            desired_side=action_to_apply,
            funding_events=funding_events,
            mark_high=float(mark_highs[i]) if np.isfinite(mark_highs[i]) else None,
            mark_low=float(mark_lows[i]) if np.isfinite(mark_lows[i]) else None,
            mark_close=float(mark_closes[i]) if np.isfinite(mark_closes[i]) else None,
        )
        if action_to_apply is not None:
            action_counts[{1: "LONG", 0: "FLAT", -1: "SHORT"}[action_to_apply]] += 1
            if "bloqueada" in note:
                skipped += 1
        equity_curve[i] = wallet.equity(float(mark_closes[i]) if np.isfinite(mark_closes[i]) else float(closes[i]))

        if i < n - 1 and decision_mask[i]:
            p_short, p_flat, p_long = probs[i]
            winner = int(np.argmax((p_short, p_flat, p_long)))
            if winner == 1 or max(p_short, p_long) < threshold:
                pending_action = 0
            else:
                pending_action = 1 if winner == 2 else -1

    if wallet.s["side"]:
        wallet.close(int(times[-1]), float(closes[-1]), "end_of_backtest")
        final_mark = float(mark_closes[-1]) if np.isfinite(mark_closes[-1]) else float(closes[-1])
        equity_curve[-1] = wallet.equity(final_mark)

    changes = np.diff(np.r_[cfg.initial_balance, equity_curve])
    wins = [event for event in trade_events if float(event.get("pnl_usdt", 0.0)) > 0]
    losses = [event for event in trade_events if float(event.get("pnl_usdt", 0.0)) < 0]
    gross_profit = float(sum(float(event["pnl_usdt"]) for event in wins))
    gross_loss = float(abs(sum(float(event["pnl_usdt"]) for event in losses)))
    running_peak = np.maximum.accumulate(np.r_[cfg.initial_balance, equity_curve])
    drawdowns = (running_peak - np.r_[cfg.initial_balance, equity_curve]) / np.maximum(running_peak, 1e-12)
    return {
        "trades": len(trade_events),
        "total_return_pct": round((equity_curve[-1] / cfg.initial_balance - 1) * 100, 5),
        "final_equity_usdt": round(float(equity_curve[-1]), 4),
        "realized_pnl_usdt": round(float(wallet.s["realized_pnl"]), 4),
        "max_drawdown_pct": round(float(np.max(drawdowns, initial=0.0)) * 100, 5),
        "win_rate": round(len(wins) / len(trade_events), 4) if trade_events else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else (None if gross_profit == 0 else 999.0),
        "average_trade_pnl_usdt": round(float(np.mean([event["pnl_usdt"] for event in trade_events])), 4) if trade_events else 0.0,
        "funding_events": int(event_counts["PAPER_FUNDING"]),
        "liquidations": sum(bool(event.get("liquidated")) for event in trade_events),
        "skipped_margin_entries": skipped,
        "action_counts": dict(action_counts),
        "event_counts": dict(event_counts),
        "closed_trades": trade_events,
    }


def probabilities_for_frame(
    model: DirectionalFuturesModel | None,
    frame: pd.DataFrame,
    samples: pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Alinea probabilidades del clasificador a velas 1m; el resto queda FLAT."""
    probabilities = np.zeros((len(frame), 3), dtype=float)
    probabilities[:, 1] = 1.0
    if model is None or samples is None or samples.empty or not feature_names:
        return probabilities
    valid = samples
    X = valid[feature_names].to_numpy(dtype=float)
    predicted = model.predict_proba(X)
    row_by_time = {int(t): i for i, t in enumerate(frame["open_time"].to_numpy(dtype=np.int64))}
    for time_value, row_probs in zip(valid["open_time"].to_numpy(dtype=np.int64), predicted):
        position = row_by_time.get(int(time_value))
        if position is not None:
            probabilities[position] = row_probs
    return probabilities

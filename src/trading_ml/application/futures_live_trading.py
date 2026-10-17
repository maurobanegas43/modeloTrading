"""Ejecución paper en vivo del clasificador USDⓈ-M LONG/FLAT/SHORT."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..domain.config import FuturesConfig, StrategyConfig
from ..domain.features import feature_frame, interval_decision_mask
from ..domain.futures_features import attach_futures_context
from ..domain.futures_wallet import DirectionalFuturesWallet
from ..domain.time import utc_iso
from .ports import FuturesMarketStreamPort, PaperStatePort


MAX_LIVE_SPREAD_BPS = 12.0
MAX_QUOTE_AGE_SECONDS = 5.0


def _action(probabilities: np.ndarray, threshold: float) -> tuple[int, str]:
    p_short, p_flat, p_long = (float(x) for x in probabilities)
    winner = int(np.argmax((p_short, p_flat, p_long)))
    if winner == 1:
        return 0, "FLAT es la clase más probable"
    if max(p_short, p_long) < threshold:
        return 0, "confianza direccional debajo del mínimo"
    return (-1, "SHORT más probable") if winner == 0 else (1, "LONG más probable")


def _trend_context(row: pd.Series) -> str:
    trend = float(row.get("trend_spread", 0.0))
    ret_1h = float(row.get("ret_1h", 0.0))
    if trend > 0 and ret_1h > 0:
        return "ALCISTA"
    if trend < 0 and ret_1h < 0:
        return "BAJISTA"
    return "MIXTO"


async def run_live_futures(
    bundle: dict,
    repository: PaperStatePort,
    market_stream: FuturesMarketStreamPort,
    historical_candles: pd.DataFrame,
    funding: pd.DataFrame | None = None,
    max_duration_sec: int | None = 14_400,
) -> None:
    """Evalúa al cierre de 15m, ejecuta paper en el minuto siguiente y registra estado.

    El contexto de 1h entra como variables continuas del clasificador; no es una
    compuerta de dirección. Solo se bloquea una orden nueva si faltan datos/quote o
    el spread live supera el límite. Stops, margen y pérdidas siguen en el wallet.
    """
    cfg = FuturesConfig(**bundle["config"])
    cfg.validate()
    strategy_cfg = StrategyConfig(symbol=cfg.symbol)
    candles = deque(
        historical_candles.sort_values("open_time").drop_duplicates("open_time", keep="last")
        .iloc[-6500:].to_dict("records"),
        maxlen=6500,
    )
    if len(candles) < 65:
        raise ValueError("Se necesitan al menos 65 velas USDⓈ-M para iniciar la simulación")

    wallet = DirectionalFuturesWallet(cfg, state=repository.load_state(), event_sink=repository.append_event)
    last_bar_ts = int(candles[-1]["open_time"])
    wallet.s["last_processed_ts"] = max(int(wallet.s.get("last_processed_ts", 0)), last_bar_ts)
    repository.save_state(wallet.s)

    quote: dict[str, float] = {"bid": float(candles[-1]["close"]), "ask": float(candles[-1]["close"]), "received": 0.0}
    mark_price = float(candles[-1].get("mark_close", candles[-1]["close"]))
    mark_bucket: int | None = None
    mark_high = mark_low = mark_price
    latest_funding_rate = 0.0
    next_funding_time = 0
    applied_funding_times: set[int] = set()
    pending_action: int | None = None
    last_probabilities = (0.0, 1.0, 0.0)
    last_signal_reason = "esperando cierre de vela de 15m"
    last_action_text = "sin operación"
    entry_block_reason = "inicializando datos USDⓈ-M"
    started_at = datetime.now(timezone.utc)
    started_mono = time.monotonic()
    last_market_event_mono: float | None = None
    last_market_event_utc: str | None = None
    market_message_count = 0
    last_status_write_mono = 0.0
    bar_count = 0
    prediction_count = 0
    closed_trades: list[dict] = []
    connection_state = "conectando al stream USDⓈ-M"

    def save_status(status: str, row: pd.Series | None = None) -> None:
        elapsed = time.monotonic() - started_mono
        current_price = float(row.get("close", candles[-1]["close"])) if row is not None else float(candles[-1]["close"])
        bid, ask = quote["bid"], quote["ask"]
        mid = (bid + ask) / 2 if bid > 0 and ask >= bid else current_price
        equity = wallet.equity(mark_price if mark_price > 0 else current_price)
        trend = _trend_context(row) if row is not None else "SIN DATOS"
        indicators = {}
        if row is not None:
            indicators = {
                "momentum_15m_bps": round(float(row.get("ret_15m", 0.0)) * 10_000, 2),
                "momentum_15m_atr_units": round(float(row.get("momentum_15m_atr_units", 0.0)), 3),
                "ret_1h_bps": round(float(row.get("ret_1h", 0.0)) * 10_000, 2),
                "trend_spread_bps": round(float(row.get("trend_spread", 0.0)) * 10_000, 2),
                "macro_spread_bps": round(float(row.get("macro_spread", 0.0)) * 10_000, 2),
                "vpin_1h": round(float(row.get("vpin_1h", 0.0)), 4),
                "bocpd_shock_1h": round(float(row.get("bocpd_shock_1h", 0.0)), 4),
                "atr_pct_1h": round(float(row.get("atr_pct_1h", 0.0)) * 100, 3),
                "rsi_14_1h": round(float(row.get("rsi_14_1h", 50.0)), 2),
            }
        p_short, p_flat, p_long = last_probabilities
        remaining = max_duration_sec - elapsed if max_duration_sec is not None else None
        repository.save_status({
            "version": "USDⓈ-M directional v2",
            "model_name": "HistGradientBoosting + ExtraTrees + LogisticRegression",
            "market": "Binance USDⓈ-M BTCUSDT perpetual · PAPER",
            "status": status,
            "connection": connection_state,
            "last_market_event_utc": last_market_event_utc,
            "market_event_age_sec": (
                round(time.monotonic() - last_market_event_mono, 1)
                if last_market_event_mono is not None else None
            ),
            "market_message_count": market_message_count,
            "started_utc": started_at.isoformat(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(elapsed, 1),
            "target_duration_sec": max_duration_sec,
            "remaining_sec": round(max(0.0, remaining), 1) if remaining is not None else None,
            "bars_processed": bar_count,
            "decisions_processed": prediction_count,
            "last_bar_utc": utc_iso(last_bar_ts),
            "symbol": cfg.symbol,
            "price": current_price,
            "bid": bid,
            "ask": ask,
            "spread_bps": round((ask - bid) / mid * 10_000, 3) if mid > 0 else None,
            "probabilities": {"short": round(p_short, 5), "flat": round(p_flat, 5), "long": round(p_long, 5)},
            "latest_funding_rate_bps": round(latest_funding_rate * 10_000, 4),
            "probability_threshold": cfg.probability_threshold,
            "decision_interval_minutes": cfg.decision_interval_minutes,
            "holding_horizon_minutes": cfg.holding_horizon_minutes,
            "higher_timeframe_role": "1h context for model; not an entry gate",
            "indicators": indicators,
            "trend_context_1h": trend,
            "signal_reason": last_signal_reason,
            "entry_block_reason": entry_block_reason,
            "last_action": last_action_text,
            "wallet": wallet.s,
            "equity_usdt": round(equity, 4),
            "mark_price": round(mark_price, 4) if mark_price > 0 else None,
            "risk": {
                "leverage": cfg.leverage,
                "max_margin_fraction": cfg.max_margin_fraction,
                "daily_loss_limit_pct": cfg.daily_loss_limit * 100,
                "max_drawdown_pct": cfg.max_drawdown * 100,
                "initial_stop_pct": cfg.initial_stop * 100,
                "max_live_spread_bps": MAX_LIVE_SPREAD_BPS,
            },
            "model_trained_through_utc": bundle.get("trained_through_utc"),
        })

    save_status("running")
    connection_state = "stream conectado; esperando mercado"
    save_status("running")
    reconnect_delay = 1.0
    while max_duration_sec is None or time.monotonic() - started_mono < max_duration_sec:
        try:
            async for message in market_stream.events(cfg.symbol):
                if max_duration_sec is not None and time.monotonic() - started_mono >= max_duration_sec:
                    break
                if message.get("_heartbeat"):
                    if last_market_event_mono is None:
                        connection_state = "stream conectado; esperando primer evento de mercado"
                    elif time.monotonic() - last_market_event_mono > 20:
                        connection_state = "stream sin eventos de mercado recientes"
                        entry_block_reason = connection_state
                    else:
                        connection_state = "stream USDⓈ-M activo"
                    save_status("running")
                    last_status_write_mono = time.monotonic()
                    continue
                event = message.get("data", message)
                kind = event.get("e")
                if kind in {"bookTicker", "markPriceUpdate", "kline"}:
                    last_market_event_mono = time.monotonic()
                    last_market_event_utc = datetime.now(timezone.utc).isoformat()
                    market_message_count += 1
                if kind == "bookTicker":
                    bid, ask = float(event["b"]), float(event["a"])
                    if bid > 0 and ask >= bid:
                        quote.update({"bid": bid, "ask": ask, "received": time.monotonic()})
                    if time.monotonic() - last_status_write_mono >= 5:
                        connection_state = "stream USDⓈ-M activo"
                        save_status("running")
                        last_status_write_mono = time.monotonic()
                    continue
                if kind == "markPriceUpdate":
                    mark_price = float(event.get("p", mark_price))
                    latest_funding_rate = float(event.get("r", latest_funding_rate))
                    next_funding_time = int(event.get("T", next_funding_time))
                    event_bucket = (int(event.get("E", 0)) // 60_000) * 60_000
                    if event_bucket != mark_bucket:
                        mark_bucket, mark_high, mark_low = event_bucket, mark_price, mark_price
                    else:
                        mark_high = max(mark_high, mark_price)
                        mark_low = min(mark_low, mark_price)
                    if time.monotonic() - last_status_write_mono >= 5:
                        connection_state = "stream USDⓈ-M activo"
                        save_status("running")
                        last_status_write_mono = time.monotonic()
                    continue
                if kind != "kline" or not event.get("k", {}).get("x", False):
                    continue

                kline = event["k"]
                t = int(kline["t"])
                if t <= last_bar_ts:
                    continue
                if t != last_bar_ts + 60_000:
                    connection_state = f"hueco de datos: {utc_iso(last_bar_ts)} → {utc_iso(t)}"
                bar = {
                    "open_time": t,
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": float(kline["c"]),
                    "volume": float(kline["v"]),
                    "taker_buy_volume": float(kline["V"]),
                }
                candles.append(bar)
                last_bar_ts = t
                bar_count += 1
                frame = feature_frame(pd.DataFrame(candles), strategy_cfg)
                frame = attach_futures_context(frame, funding)
                row = frame.iloc[-1]
                valid_row = bool(row.get("valid_features", False))

                age = time.monotonic() - quote["received"]
                bid, ask = quote["bid"], quote["ask"]
                mid = (bid + ask) / 2 if bid > 0 and ask >= bid else 0.0
                spread_bps = (ask - bid) / mid * 10_000 if mid > 0 else float("inf")
                quote_ok = age <= MAX_QUOTE_AGE_SECONDS and spread_bps <= MAX_LIVE_SPREAD_BPS
                action_to_apply = pending_action
                pending_action = None
                if action_to_apply is not None and not quote_ok:
                    action_to_apply = None
                    entry_block_reason = (
                        "cotización desactualizada" if age > MAX_QUOTE_AGE_SECONDS
                        else f"spread live {spread_bps:.1f} bps supera {MAX_LIVE_SPREAD_BPS:.1f} bps"
                    )

                funding_events: list[tuple[float, float]] = []
                if (
                    wallet.s["side"]
                    and next_funding_time > t
                    and next_funding_time <= t + 60_000
                    and next_funding_time not in applied_funding_times
                    and mark_price > 0
                ):
                    funding_events.append((latest_funding_rate, mark_price))
                    applied_funding_times.add(next_funding_time)
                used_mark = mark_price if mark_price > 0 else float(bar["close"])
                used_mark_high = mark_high if mark_bucket == t else used_mark
                used_mark_low = mark_low if mark_bucket == t else used_mark
                prior_closed = int(wallet.s["num_closed_trades"])
                last_action_text = wallet.on_bar(
                    t,
                    bar["open"], bar["high"], bar["low"], bar["close"],
                    desired_side=action_to_apply,
                    funding_events=funding_events,
                    mark_high=used_mark_high,
                    mark_low=used_mark_low,
                    mark_close=used_mark,
                )
                if action_to_apply is not None:
                    entry_block_reason = "orden paper evaluada al siguiente minuto"
                    if "bloqueada" in last_action_text:
                        entry_block_reason = last_action_text
                if int(wallet.s["num_closed_trades"]) > prior_closed:
                    closed_trades.append({"at": utc_iso(t), "last_action": last_action_text, "realized_pnl": wallet.s["realized_pnl"]})
                repository.save_state(wallet.s)

                if valid_row and bool(interval_decision_mask(frame.iloc[[-1]], cfg.decision_interval_minutes)[0]):
                    features = row[bundle["features"]].to_numpy(dtype=float)
                    action = 0
                    if np.isfinite(features).all():
                        last_probabilities = tuple(float(x) for x in bundle["model"].predict_proba(features)[0])
                        action, last_signal_reason = _action(np.asarray(last_probabilities), cfg.probability_threshold)
                        prediction_count += 1
                        if quote_ok:
                            pending_action = action
                            entry_block_reason = (
                                last_signal_reason if action == 0 else "señal direccional lista; se aplica en apertura de vela 1m"
                            )
                        else:
                            pending_action = None
                            entry_block_reason = (
                                "cotización desactualizada" if age > MAX_QUOTE_AGE_SECONDS
                                else f"spread live {spread_bps:.1f} bps supera {MAX_LIVE_SPREAD_BPS:.1f} bps"
                            )
                    else:
                        pending_action = None
                        last_signal_reason = "características no finitas"
                        entry_block_reason = last_signal_reason
                    p_short, p_flat, p_long = last_probabilities
                    repository.append_prediction({
                        "at": utc_iso(t + 60_000),
                        "close": bar["close"],
                        "bid": bid,
                        "ask": ask,
                        "p_short": p_short,
                        "p_flat": p_flat,
                        "p_long": p_long,
                        "threshold": cfg.probability_threshold,
                        "action": {-1: "SHORT", 0: "FLAT", 1: "LONG"}[action],
                        "reason": last_signal_reason,
                        "trend_context_1h": _trend_context(row),
                        "momentum_15m_bps": float(row.get("ret_15m", 0.0)) * 10_000,
                        "momentum_15m_atr_units": float(row.get("momentum_15m_atr_units", 0.0)),
                    })
                elif not valid_row:
                    entry_block_reason = "calentando indicadores o recuperando continuidad de datos"
                else:
                    entry_block_reason = "esperando cierre de vela de 15m"

                connection_state = "stream USDⓈ-M activo"
                save_status("running", row)
                last_status_write_mono = time.monotonic()
                reconnect_delay = 1.0
            if max_duration_sec is not None and time.monotonic() - started_mono >= max_duration_sec:
                break
            connection_state = "stream cerrado; reconectando"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            connection_state = f"reconexión: {type(exc).__name__}: {exc}"
            entry_block_reason = connection_state
            save_status("running")
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 30.0)

    if wallet.s["side"]:
        wallet.close(last_bar_ts, mark_price if mark_price > 0 else float(candles[-1]["close"]), "fin_sesion_paper")
    repository.save_state(wallet.s)
    final_equity = wallet.equity(mark_price if mark_price > 0 else float(candles[-1]["close"]))
    repository.save_summary({
        "status": "completed",
        "started_utc": started_at.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "bars_processed": bar_count,
        "decisions_processed": prediction_count,
        "closed_trades": int(wallet.s["num_closed_trades"]),
        "realized_pnl_usdt": round(float(wallet.s["realized_pnl"]), 4),
        "final_equity_usdt": round(float(final_equity), 4),
        "model_format": bundle.get("format"),
    })
    entry_block_reason = "sesión de simulación completada"
    last_action_text = wallet.s.get("last_action", "FLAT")
    save_status("completed")

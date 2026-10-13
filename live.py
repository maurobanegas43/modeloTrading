from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from websockets.asyncio.client import connect

from .config import FEATURE_WARMUP, MINUTE_MS, StrategyConfig
from .data import recent_closed_candles, utc_iso
from .features import FEATURES, feature_frame
from .modeling import learn_mature_outcome, load_bundle, predict_pair


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def _append_event(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")


class VirtualWallet:
    """Long-only paper wallet. No client or method for authenticated orders."""

    def __init__(self, cfg: StrategyConfig, path: Path, events: Path):
        self.cfg, self.path, self.events = cfg, path, events
        self.s = {
            "cash": cfg.initial_cash, "qty": 0.0, "entry_fill": 0.0,
            "entry_allocation": 0.0, "entry_signal_ts": 0,
            "last_processed_ts": 0, "cooldown_ts": 0,
            "day": "", "day_equity": cfg.initial_cash, "halt_today": False,
            "peak_equity": cfg.initial_cash, "frozen": False,
            "num_closed_trades": 0, "realized_pnl": 0.0,
        }
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.s.update(stored)
            if "cash" not in stored or "qty" not in stored:
                raise ValueError("Estado del paper wallet dañado; no continuar")

    def equity(self, bid: float) -> float:
        return float(self.s["cash"]) + float(self.s["qty"]) * bid * (1 - self.cfg.fee_rate)

    def close(self, t: int, bid: float, reason: str) -> None:
        if self.s["qty"] <= 0:
            return
        fill = bid * (1 - self.cfg.execution_bps / 20_000)
        proceeds = self.s["qty"] * fill * (1 - self.cfg.fee_rate)
        pnl = proceeds - self.s["entry_allocation"]
        self.s["cash"] += proceeds
        self.s["realized_pnl"] += pnl
        self.s["num_closed_trades"] += 1
        self.s["cooldown_ts"] = t + self.cfg.horizon_bars * MINUTE_MS
        _append_event(self.events, {"type": "PAPER_SELL", "at": utc_iso(t), "fill": fill,
                                   "proceeds": proceeds, "pnl": pnl, "reason": reason})
        self.s.update(qty=0.0, entry_fill=0.0, entry_allocation=0.0, entry_signal_ts=0)

    def on_bar(self, t: int, high: float, low: float, close: float, bid: float, ask: float,
               prob: float, challenger: float, threshold: float, allow_paper: bool,
               quote_fresh: bool) -> str:
        if t <= self.s["last_processed_ts"]:
            return "repetida"
        day = utc_iso(t)[:10]
        if day != self.s["day"]:
            self.s["day"], self.s["halt_today"] = day, False
            self.s["day_equity"] = self.equity(bid)
        # A stale quote prevents simulated fills and entries; it never activates real trading.
        if not quote_fresh:
            self.s["last_processed_ts"] = t
            _atomic_json(self.path, self.s)
            return "cotización desactualizada"
        note = "sin operación"
        if self.s["qty"] > 0:
            stop = self.s["entry_fill"] * (1 - self.cfg.stop_loss)
            if low <= stop:
                self.close(t, min(bid, stop), "stop_loss")
                note = "stop virtual"
            elif t >= self.s["entry_signal_ts"] + self.cfg.horizon_bars * MINUTE_MS:
                self.close(t, bid, "horizon")
                note = "cierre por horizonte"

        equity = self.equity(bid)
        self.s["peak_equity"] = max(self.s["peak_equity"], equity)
        if equity <= self.s["day_equity"] * (1 - self.cfg.daily_loss_limit):
            self.s["halt_today"] = True
            self.close(t, bid, "daily_loss_limit")
            note = "límite de pérdida diaria"
        if equity <= self.s["peak_equity"] * (1 - self.cfg.max_drawdown):
            self.s["frozen"] = True
            self.close(t, bid, "max_drawdown")
            note = "bloqueo por drawdown"

        if (allow_paper and self.s["qty"] == 0 and not self.s["halt_today"]
                and not self.s["frozen"] and t >= self.s["cooldown_ts"]
                and prob >= threshold and challenger >= 0.52
                and bid > 0 and ask >= bid and (ask - bid) / bid <= 0.0015):
            allocation = self.s["cash"] * self.cfg.allocation
            if allocation >= 10:
                fill = ask * (1 + self.cfg.execution_bps / 20_000)
                self.s["qty"] = allocation * (1 - self.cfg.fee_rate) / fill
                self.s["cash"] -= allocation
                self.s["entry_fill"] = fill
                self.s["entry_allocation"] = allocation
                self.s["entry_signal_ts"] = t
                _append_event(self.events, {"type": "PAPER_BUY", "at": utc_iso(t),
                                           "fill": fill, "allocation": allocation,
                                           "champion_p": prob, "challenger_p": challenger})
                note = "compra virtual"
        self.s["last_processed_ts"] = t
        _atomic_json(self.path, self.s)
        return note


async def run_live(model_path: Path, paper_dir: Path, allow_unvalidated_paper: bool = False) -> None:
    bundle = load_bundle(model_path)
    cfg = StrategyConfig(**bundle["config"])
    paper_dir.mkdir(parents=True, exist_ok=True)
    wallet = VirtualWallet(cfg, paper_dir / "state.json", paper_dir / "events.jsonl")
    online_path = paper_dir / "online_checkpoint.joblib"
    if online_path.is_file():
        checkpoint = joblib.load(online_path)  # local checkpoints only, never external files
        if checkpoint.get("base_training_utc") == bundle["last_training_utc"]:
            bundle["online"] = checkpoint["online"]
    allowed = bool(bundle["paper_gate"] or allow_unvalidated_paper)
    print(f"MODE: SOLO PAPER / SIN CLAVES API / sin ejecución real. {cfg.symbol}; horizonte {cfg.horizon_bars}m")
    print(f"Señales virtuales: {'habilitadas' if allowed else 'deshabilitadas (backtest insuficiente)'}")
    if allow_unvalidated_paper and not bundle["paper_gate"]:
        print("ADVERTENCIA: --allow-unvalidated-paper solo habilita experimentos con saldo FICTICIO")

    historical = await asyncio.to_thread(recent_closed_candles, cfg.symbol, 250)
    candles = deque(historical.to_dict("records"), maxlen=300)
    last_ts = int(historical.iloc[-1]["open_time"])
    pending: deque[dict] = deque()
    quote: dict = {"bid": 0.0, "ask": 0.0, "received": 0.0}
    matured_since_save = 0
    reconnect_delay = 2
    uri = ("wss://stream.binance.com:9443/stream?streams="
           f"{cfg.symbol.lower()}@bookTicker/{cfg.symbol.lower()}@kline_1m")
    while True:
        try:
            async with connect(uri, ping_interval=20, ping_timeout=20, close_timeout=5, max_size=2**20) as ws:
                print("WebSocket conectado; esperando velas cerradas...")
                quote["received"] = 0.0  # reject quotes from an older connection
                reconnect_delay = 2
                async for message in ws:
                    event = json.loads(message)
                    data = event.get("data", {})
                    if event.get("stream", "").endswith("@bookTicker"):
                        bid, ask = float(data.get("b", 0)), float(data.get("a", 0))
                        if bid > 0 and ask >= bid:
                            quote.update(bid=bid, ask=ask, received=time.monotonic())
                        continue
                    if data.get("e") != "kline" or not data["k"].get("x"):
                        continue
                    k = data["k"]
                    t = int(k["t"])
                    if t <= last_ts:
                        continue
                    if t - last_ts != MINUTE_MS:
                        # A gap invalidates labels and live decisions. Refresh input and
                        # freeze paper entries until inspected/restarted manually.
                        print("AVISO: hueco de datos; se congela el paper y se recupera el historial")
                        wallet.s["frozen"] = True
                        pending.clear()
                        refreshed = await asyncio.to_thread(recent_closed_candles, cfg.symbol, 250)
                        candles = deque(refreshed.to_dict("records"), maxlen=300)
                        last_ts = int(refreshed.iloc[-1]["open_time"])
                        _atomic_json(wallet.path, wallet.s)
                        continue
                    last_ts = t
                    bar = {
                        "open_time": t, "open": float(k["o"]), "high": float(k["h"]),
                        "low": float(k["l"]), "close": float(k["c"]),
                        "volume": float(k["v"]), "taker_buy_volume": float(k["V"]),
                    }
                    candles.append(bar)
                    df = feature_frame(pd.DataFrame(list(candles)))
                    last = df.iloc[-1]
                    if not bool(last["valid_features"]):
                        print(f"{utc_iso(t)} | calentando features, todavía sin señal")
                        continue

                    # Mature previously predicted examples BEFORE examining this new signal.
                    for item in list(pending):
                        if item["entry_open"] is None and t == item["origin"] + MINUTE_MS:
                            item["entry_open"] = bar["open"]
                        if t >= item["origin"] + cfg.horizon_bars * MINUTE_MS:
                            pending.remove(item)
                            if t != item["origin"] + cfg.horizon_bars * MINUTE_MS or item["entry_open"] is None:
                                continue
                            entry = item["entry_open"] * (1 + cfg.execution_bps / 10_000)
                            exit_px = bar["close"] * (1 - cfg.execution_bps / 10_000)
                            net = exit_px * (1 - cfg.fee_rate) / (entry * (1 + cfg.fee_rate)) - 1
                            learn_mature_outcome(bundle, item["features"], int(net > 0))
                            matured_since_save += 1

                    vector = last[FEATURES].to_numpy(dtype=float)
                    p_champion, p_challenger = predict_pair(bundle, vector)
                    pending.append({"origin": t, "entry_open": None, "features": vector})
                    fresh = (time.monotonic() - quote["received"]) <= 5
                    note = wallet.on_bar(t, bar["high"], bar["low"], bar["close"],
                                         quote["bid"], quote["ask"], p_champion,
                                         p_challenger, float(bundle["threshold"]), allowed, fresh)
                    _append_event(paper_dir / "predictions.jsonl", {
                        "at": utc_iso(t), "close": bar["close"], "champion_p": p_champion,
                        "challenger_p": p_challenger, "threshold": bundle["threshold"],
                        "paper_event": note, "fresh_bid_ask": fresh,
                    })
                    print(f"{utc_iso(t)} | BTC {bar['close']:.2f} | XGB {p_champion:.3f} "
                          f"| online {p_challenger:.3f} | {note}")
                    if matured_since_save >= 10:
                        joblib.dump({"online": bundle["online"], "base_training_utc": bundle["last_training_utc"]},
                                    online_path)
                        matured_since_save = 0
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            print(f"Conexión interrumpida: {type(exc).__name__}: {exc}")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)


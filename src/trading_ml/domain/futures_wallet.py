"""Billetera paper de futuros lineales USDⓈ-M con soporte LONG y SHORT."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .config import FuturesConfig
from .time import utc_iso


class DirectionalFuturesWallet:
    def __init__(
        self,
        cfg: FuturesConfig,
        state: dict | None = None,
        event_sink: Callable[[dict], None] | None = None,
    ):
        cfg.validate()
        self.cfg = cfg
        self._event_sink = event_sink or (lambda event: None)
        self.s = {
            "balance": cfg.initial_balance,
            "side": 0,
            "qty": 0.0,
            "entry_price": 0.0,
            "entry_notional": 0.0,
            "initial_margin": 0.0,
            "entry_time": 0,
            "best_price": 0.0,
            "entry_fee": 0.0,
            "position_funding_pnl": 0.0,
            "realized_pnl": 0.0,
            "num_closed_trades": 0,
            "peak_equity": cfg.initial_balance,
            "day": "",
            "day_equity": cfg.initial_balance,
            "halt_today": False,
            "frozen": False,
            "last_processed_ts": 0,
            "last_action": "FLAT",
        }
        if state is not None:
            self.s.update(state)
            if "balance" not in state or "side" not in state or "qty" not in state:
                raise ValueError("Estado del futures wallet dañado; no continuar")

    def equity(self, mark_price: float) -> float:
        side = int(self.s["side"])
        if side == 0:
            return float(self.s["balance"])
        expected_exit = mark_price * (1 - side * self.cfg.execution_bps / 10_000)
        unrealized = side * self.s["qty"] * (expected_exit - self.s["entry_price"])
        unrealized -= abs(self.s["qty"] * expected_exit) * self.cfg.fee_rate
        return float(self.s["balance"] + unrealized)

    def _open(self, t: int, side: int, reference_price: float) -> bool:
        if side not in (-1, 1) or reference_price <= 0 or not np.isfinite(reference_price):
            return False
        equity = self.equity(reference_price)
        notional = equity * self.cfg.max_margin_fraction * self.cfg.leverage
        margin = notional / self.cfg.leverage
        entry_price = reference_price * (1 + side * self.cfg.execution_bps / 10_000)
        qty = notional / entry_price
        fee = notional * self.cfg.fee_rate
        if notional <= 0 or margin + fee > self.s["balance"] or qty <= 0:
            return False
        self.s.update({
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "entry_notional": notional,
            "initial_margin": margin,
            "entry_time": int(t),
            "best_price": entry_price,
            "entry_fee": fee,
            "position_funding_pnl": 0.0,
        })
        self.s["balance"] -= fee
        direction = "LONG" if side > 0 else "SHORT"
        self.s["last_action"] = f"OPEN_{direction}"
        self._event_sink({
            "type": f"PAPER_OPEN_{direction}",
            "at": utc_iso(t),
            "side": direction,
            "fill": entry_price,
            "qty": qty,
            "notional_usdt": notional,
            "initial_margin_usdt": margin,
            "leverage": self.cfg.leverage,
            "fee_usdt": fee,
        })
        return True

    def close(self, t: int, reference_price: float, reason: str, liquidation: bool = False) -> float:
        side = int(self.s["side"])
        if side == 0 or self.s["qty"] <= 0:
            return 0.0
        fill = reference_price * (1 - side * self.cfg.execution_bps / 10_000)
        qty = float(self.s["qty"])
        gross_pnl = side * qty * (fill - float(self.s["entry_price"]))
        close_fee = abs(qty * fill) * self.cfg.fee_rate
        liquidation_fee = abs(qty * fill) * self.cfg.liquidation_fee_rate if liquidation else 0.0
        trade_pnl = (
            gross_pnl
            - float(self.s["entry_fee"])
            - close_fee
            - liquidation_fee
            + float(self.s["position_funding_pnl"])
        )
        self.s["balance"] += gross_pnl - close_fee - liquidation_fee
        self.s["realized_pnl"] += trade_pnl
        self.s["num_closed_trades"] += 1
        direction = "LONG" if side > 0 else "SHORT"
        self.s["last_action"] = f"CLOSE_{direction}"
        self._event_sink({
            "type": f"PAPER_CLOSE_{direction}",
            "at": utc_iso(t),
            "side": direction,
            "fill": fill,
            "qty": qty,
            "gross_pnl_usdt": gross_pnl,
            "funding_pnl_usdt": float(self.s["position_funding_pnl"]),
            "entry_fee_usdt": float(self.s["entry_fee"]),
            "close_fee_usdt": close_fee,
            "liquidation_fee_usdt": liquidation_fee,
            "pnl_usdt": trade_pnl,
            "reason": reason,
            "liquidated": bool(liquidation),
        })
        self.s.update({
            "side": 0,
            "qty": 0.0,
            "entry_price": 0.0,
            "entry_notional": 0.0,
            "initial_margin": 0.0,
            "entry_time": 0,
            "best_price": 0.0,
            "entry_fee": 0.0,
            "position_funding_pnl": 0.0,
        })
        return float(trade_pnl)

    def apply_action(self, t: int, reference_price: float, desired_side: int) -> str:
        if desired_side not in (-1, 0, 1):
            raise ValueError("desired_side debe ser -1 SHORT, 0 FLAT o 1 LONG")
        current = int(self.s["side"])
        if desired_side == current:
            return "mantener posición" if current else "mantener FLAT"
        if current:
            reason = "señal SHORT" if desired_side < 0 else "señal LONG" if desired_side > 0 else "señal FLAT"
            self.close(t, reference_price, reason)
        if desired_side and not self.s["halt_today"] and not self.s["frozen"]:
            if self._open(t, desired_side, reference_price):
                return "abrir LONG" if desired_side > 0 else "abrir SHORT"
            return "entrada bloqueada por margen disponible"
        return "cerrar y quedar FLAT" if current else "quedar FLAT"

    def on_bar(
        self,
        t: int,
        open_price: float,
        high: float,
        low: float,
        close: float,
        desired_side: int | None = None,
        funding_events: list[tuple[float, float]] | None = None,
        mark_high: float | None = None,
        mark_low: float | None = None,
        mark_close: float | None = None,
    ) -> str:
        if t <= int(self.s["last_processed_ts"]):
            return "repetida"
        day = utc_iso(t)[:10]
        if day != self.s["day"]:
            self.s["day"] = day
            self.s["halt_today"] = False
            self.s["day_equity"] = self.equity(open_price)

        for rate, mark in funding_events or []:
            if self.s["side"] and mark > 0:
                payment = -self.s["side"] * self.s["qty"] * mark * rate
                self.s["balance"] += payment
                self.s["position_funding_pnl"] += payment
                self._event_sink({
                    "type": "PAPER_FUNDING",
                    "at": utc_iso(t),
                    "side": "LONG" if self.s["side"] > 0 else "SHORT",
                    "rate": rate,
                    "mark_price": mark,
                    "payment_usdt": payment,
                })

        note = "sin operación"
        if desired_side is not None:
            note = self.apply_action(t, open_price, int(desired_side))

        side = int(self.s["side"])
        if side:
            if side > 0:
                self.s["best_price"] = max(float(self.s["best_price"]), high)
                stop = float(self.s["entry_price"]) * (1 - self.cfg.initial_stop)
                if self.s["best_price"] >= self.s["entry_price"] * (1 + self.cfg.trailing_activation):
                    stop = max(stop, float(self.s["best_price"]) * (1 - self.cfg.trailing_delta))
                liq_price = float(self.s["entry_price"]) * (1 - 1 / self.cfg.leverage + self.cfg.maintenance_margin_rate)
                observed_low = low if mark_low is None else mark_low
                if low <= stop:
                    self.close(t, min(open_price, stop), "trailing_stop_or_stop_loss")
                    note = "cierre LONG por stop"
                elif observed_low <= liq_price:
                    self.close(t, liq_price, "liquidation", liquidation=True)
                    note = "liquidación paper LONG"
            else:
                self.s["best_price"] = min(float(self.s["best_price"]), low)
                stop = float(self.s["entry_price"]) * (1 + self.cfg.initial_stop)
                if self.s["best_price"] <= self.s["entry_price"] * (1 - self.cfg.trailing_activation):
                    stop = min(stop, float(self.s["best_price"]) * (1 + self.cfg.trailing_delta))
                liq_price = float(self.s["entry_price"]) * (1 + 1 / self.cfg.leverage - self.cfg.maintenance_margin_rate)
                observed_high = high if mark_high is None else mark_high
                if high >= stop:
                    self.close(t, max(open_price, stop), "trailing_stop_or_stop_loss")
                    note = "cierre SHORT por stop"
                elif observed_high >= liq_price:
                    self.close(t, liq_price, "liquidation", liquidation=True)
                    note = "liquidación paper SHORT"

        equity = self.equity(close if mark_close is None else mark_close)
        self.s["peak_equity"] = max(float(self.s["peak_equity"]), equity)
        if equity <= float(self.s["day_equity"]) * (1 - self.cfg.daily_loss_limit):
            self.s["halt_today"] = True
            if self.s["side"]:
                self.close(t, close, "daily_loss_limit")
            note = "límite diario de pérdida"
        if equity <= float(self.s["peak_equity"]) * (1 - self.cfg.max_drawdown):
            self.s["frozen"] = True
            if self.s["side"]:
                self.close(t, close, "max_drawdown")
            note = "bloqueo por drawdown"
        self.s["last_processed_ts"] = int(t)
        return note

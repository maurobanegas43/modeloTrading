from __future__ import annotations

from typing import Callable

import numpy as np

from .backtest import _compute_position_fraction
from .config import MINUTE_MS, StrategyConfig
from .time import utc_iso


class PaperWallet:
    """Long-only paper trading wallet with asymmetric trailing stops, BOCPD shock exits, and ACI sizing."""

    def __init__(self, cfg: StrategyConfig, state: dict | None = None, event_sink: Callable[[dict], None] | None = None):
        self.cfg = cfg
        self._event_sink = event_sink or (lambda event: None)
        self.s = {
            "cash": cfg.initial_cash, "qty": 0.0, "entry_fill": 0.0,
            "entry_allocation": 0.0, "entry_signal_ts": 0, "pos_peak": 0.0,
            "last_processed_ts": 0, "cooldown_ts": 0,
            "day": "", "day_equity": cfg.initial_cash, "halt_today": False,
            "peak_equity": cfg.initial_cash, "frozen": False,
            "num_closed_trades": 0, "realized_pnl": 0.0,
            "last_close_ret_pct": 0.0,
            "entry_is_calm": False, "behavior_state": "CALM_ACCUMULATION",
        }
        if state is not None:
            self.s.update(state)
            if "cash" not in state or "qty" not in state:
                raise ValueError("Estado del paper wallet dañado; no continuar")

    def equity(self, bid: float) -> float:
        exit_cost = (1 - self.cfg.execution_bps / 10_000) * (1 - self.cfg.fee_rate)
        return float(self.s["cash"]) + float(self.s["qty"]) * bid * exit_cost

    def close(self, t: int, bid: float, reason: str, quiet: bool = False) -> float:
        if self.s["qty"] <= 0:
            return 0.0
        fill = bid * (1 - self.cfg.execution_bps / 10_000)
        proceeds = self.s["qty"] * fill * (1 - self.cfg.fee_rate)
        pnl = proceeds - self.s["entry_allocation"]
        ret_pct = (pnl / self.s["entry_allocation"]) * 100.0 if self.s["entry_allocation"] > 0 else 0.0
        self.s["cash"] += proceeds
        self.s["realized_pnl"] += pnl
        self.s["num_closed_trades"] += 1
        self.s["last_close_ret_pct"] = ret_pct
        self.s["cooldown_ts"] = t + 30 * MINUTE_MS
        self._event_sink({
            "type": "PAPER_SELL", "at": utc_iso(t), "fill": fill,
            "proceeds": proceeds, "pnl": pnl, "ret_pct": ret_pct,
            "reason": reason,
        })
        self.s.update(qty=0.0, entry_fill=0.0, entry_allocation=0.0, entry_signal_ts=0, pos_peak=0.0)
        return float(pnl)

    def on_bar(
        self,
        t: int,
        high: float,
        low: float,
        close: float,
        bid: float,
        ask: float,
        prob: float,
        challenger: float,
        threshold: float,
        allow_paper: bool,
        quote_fresh: bool,
        is_hour_start: bool = True,
        is_trigger: bool = True,
        ema20_1h: float | None = None,
        ema50_1h: float | None = None,
        atr_pct_1h: float | None = None,
        vpin_1h: float | None = None,
        bocpd_shock_1h: float | None = None,
        ret_15m: float | None = None,
        ret_1h: float | None = None,
        vol_ratio_1h: float | None = None,
        taker_buy_ratio_1h: float | None = None,
        entry_ask: float | None = None,
        process_entry_bar_ohlc: bool = False,
        open_price: float | None = None,
        quiet: bool = False,
    ) -> str:
        if t <= self.s["last_processed_ts"]:
            return "repetida"
        day = utc_iso(t)[:10]
        if day != self.s["day"]:
            self.s["day"], self.s["halt_today"] = day, False
            self.s["day_equity"] = self.equity(bid)

        if not quote_fresh:
            self.s["last_processed_ts"] = t
            return "cotización desactualizada"

        trend_floor = ema50_1h if ema50_1h is not None else ema20_1h
        note = "sin operación"
        if self.s["qty"] > 0:
            entry_px = self.s["entry_fill"]
            peak = max(self.s.get("pos_peak", entry_px), high)
            self.s["pos_peak"] = peak

            is_calm_pos = bool(self.s.get("entry_is_calm", False))
            eff_act = min(self.cfg.trailing_activation, 0.0058) if is_calm_pos else self.cfg.trailing_activation
            eff_del = min(self.cfg.trailing_delta, 0.0012) if is_calm_pos else self.cfg.trailing_delta

            trailing_active = peak >= entry_px * (1 + eff_act)
            if trailing_active:
                trail_stop = peak * (1 - eff_del)
            else:
                trail_stop = entry_px * (1 - self.cfg.initial_stop)

            drastic_reversal = (
                t > self.s["entry_signal_ts"]
                and bocpd_shock_1h is not None
                and bocpd_shock_1h >= self.cfg.bocpd_exit_threshold
                and (
                    (vpin_1h is not None and vpin_1h >= self.cfg.vpin_exit_threshold and ret_15m is not None and ret_15m <= -0.008)
                    or (ret_1h is not None and ret_1h <= -0.0050 and taker_buy_ratio_1h is not None and taker_buy_ratio_1h <= 0.45)
                )
            )

            if low <= trail_stop:
                reason = "trailing_stop" if trailing_active else "stop_loss"
                stop_fill = open_price if open_price is not None else bid
                self.close(t, min(stop_fill, trail_stop), reason, quiet=quiet)
                note = "trailing stop virtual" if trailing_active else "stop virtual"
            elif drastic_reversal:
                self.close(t, bid, "bocpd_shock_exit", quiet=quiet)
                note = "salida por cambio drástico de tendencia"
            elif is_hour_start and (t > self.s["entry_signal_ts"]) and (trend_floor is not None) and np.isfinite(trend_floor) and (close < trend_floor):
                self.close(t, bid, "trend_exit", quiet=quiet)
                note = "cierre por cambio de tendencia"
            elif t >= self.s["entry_signal_ts"] + (self.cfg.horizon_bars - 1) * MINUTE_MS:
                self.close(t, bid, "horizon", quiet=quiet)
                note = "cierre por horizonte"

        equity = self.equity(bid)
        self.s["peak_equity"] = max(self.s["peak_equity"], equity)
        if equity <= self.s["day_equity"] * (1 - self.cfg.daily_loss_limit):
            self.s["halt_today"] = True
            self.close(t, bid, "daily_loss_limit", quiet=quiet)
            note = "límite de pérdida diaria"
        if equity <= self.s["peak_equity"] * (1 - self.cfg.max_drawdown):
            self.s["frozen"] = True
            self.close(t, bid, "max_drawdown", quiet=quiet)
            note = "bloqueo por drawdown"

        if (allow_paper and self.s["qty"] == 0 and not self.s["halt_today"]
                and not self.s["frozen"] and t >= self.s["cooldown_ts"]
                and is_trigger and prob >= threshold
                and bid > 0 and ask >= bid and (ask - bid) / bid <= 0.0015):
            if atr_pct_1h is not None and np.isfinite(atr_pct_1h):
                pos_frac = _compute_position_fraction(
                    prob,
                    threshold,
                    atr_pct_1h,
                    self.cfg,
                    vpin=vpin_1h if vpin_1h is not None else np.nan,
                    bocpd_shock=bocpd_shock_1h if bocpd_shock_1h is not None else np.nan,
                )
            else:
                pos_frac = self.cfg.allocation
            allocation = self.s["cash"] * pos_frac
            if allocation >= 10:
                fill = (entry_ask if entry_ask is not None else ask) * (1 + self.cfg.execution_bps / 10_000)
                self.s["qty"] = allocation * (1 - self.cfg.fee_rate) / fill
                self.s["cash"] -= allocation
                self.s["entry_fill"] = fill
                self.s["entry_allocation"] = allocation
                self.s["entry_signal_ts"] = t
                self.s["pos_peak"] = fill
                self.s["entry_is_calm"] = bool(
                    vol_ratio_1h is not None and np.isfinite(vol_ratio_1h) and vol_ratio_1h < 1.50
                    and taker_buy_ratio_1h is not None and np.isfinite(taker_buy_ratio_1h) and taker_buy_ratio_1h < 0.505
                )
                self._event_sink({
                    "type": "PAPER_BUY", "at": utc_iso(t),
                    "fill": fill, "allocation": allocation,
                    "champion_p": prob, "challenger_p": challenger,
                    "aci_threshold": threshold,
                })
                note = "compra virtual"
                if process_entry_bar_ohlc:
                    peak = max(fill, high)
                    self.s["pos_peak"] = peak
                    is_calm_pos = self.s["entry_is_calm"]
                    eff_act = min(self.cfg.trailing_activation, 0.0058) if is_calm_pos else self.cfg.trailing_activation
                    eff_del = min(self.cfg.trailing_delta, 0.0012) if is_calm_pos else self.cfg.trailing_delta
                    trailing_active = peak >= fill * (1 + eff_act)
                    entry_stop = peak * (1 - eff_del) if trailing_active else fill * (1 - self.cfg.initial_stop)
                    if low <= entry_stop:
                        raw_exit = min(float(entry_ask or ask), entry_stop)
                        self.close(t, raw_exit, "trailing_stop" if trailing_active else "stop_loss", quiet=quiet)
                        note = "trailing stop virtual" if trailing_active else "stop virtual"

        self.s["last_processed_ts"] = t
        return note



from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import FEATURE_WARMUP, MINUTE_MS, StrategyConfig
from .time import utc_iso


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    allocation: float
    pnl: float
    return_pct: float
    reason: str


class AdaptiveConformalThreshold:
    """Adaptive Conformal Inference (ACI — Gibbs & Candes 2021) dynamic probability threshold.

    Updates the effective miscoverage parameter alpha_t online without assuming exchangeability:
        alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
    When a trade loses (err_t = 1) or a BOCPD structural break occurs, alpha_t contracts and
    the effective threshold tau_t increases immediately, filtering out noisy post-shock signals.
    """

    def __init__(self, base_threshold: float, gamma: float = 0.015, target_alpha: float = 0.30):
        self.base_threshold = float(base_threshold)
        self.gamma = float(gamma)
        self.target_alpha = float(target_alpha)
        self.alpha_t = float(target_alpha)

    def current_threshold(self, bocpd_shock: float = 0.0) -> float:
        # Convert alpha_t shift into threshold adjustment + instantaneous BOCPD shock penalty
        alpha_shift = self.target_alpha - self.alpha_t
        shock_penalty = 0.04 * max(0.0, float(bocpd_shock) - 0.50) if np.isfinite(bocpd_shock) else 0.0
        tau_t = self.base_threshold + alpha_shift + shock_penalty
        return float(np.clip(tau_t, max(0.25, self.base_threshold - 0.02), min(0.85, self.base_threshold + 0.08)))

    def update(self, trade_won: bool, ret_pct: float = 0.0) -> float:
        # Increased Winner Reward / Penalty Index: reward strong winning orders faster to prevent over-protection
        if trade_won:
            win_boost = 1.45 * (1.0 + min(1.5, max(0.0, ret_pct) / 1.0))
            step = self.gamma * win_boost * self.target_alpha
        else:
            step = -self.gamma * 1.15 * (1.0 - self.target_alpha)
        self.alpha_t = float(np.clip(self.alpha_t + step, 0.10, 0.50))
        return self.current_threshold()


def _compute_position_fraction(
    prob: float,
    threshold: float,
    atr_pct: float,
    cfg: StrategyConfig,
    vpin: float = np.nan,
    bocpd_shock: float = np.nan,
    regime_code: int = 2,
) -> float:
    """Regime-Conditional Fractional Kelly, inverse-volatility, and VPIN/BOCPD toxicity-adjusted position sizing."""
    if regime_code < 0:
        return 0.0
    if not cfg.dynamic_sizing or not np.isfinite(atr_pct) or atr_pct <= 0:
        return cfg.allocation
    # Payoff ratio b = trailing_activation / initial_stop; Fractional Kelly edge
    margin = max(0.0, prob - threshold) / max(0.05, 1.0 - threshold)
    conviction_mult = 0.85 + 0.40 * min(1.0, margin * 2.0)
    vol_mult = np.clip(0.0040 / max(0.0015, atr_pct), 0.75, 1.20)
    toxicity_mult = 1.0
    if np.isfinite(vpin) and vpin > 0.25:
        toxicity_mult *= float(np.clip(1.0 - 0.50 * (vpin - 0.25), 0.75, 1.0))
    if np.isfinite(bocpd_shock) and bocpd_shock > 0.40:
        toxicity_mult *= float(np.clip(1.0 - 0.40 * (bocpd_shock - 0.40), 0.75, 1.0))
    max_cap = 0.35 if cfg.allocation <= 0.25 else min(0.72, cfg.allocation * 1.40)
    return float(np.clip(cfg.allocation * conviction_mult * vol_mult * toxicity_mult, 0.10, max_cap))


def compute_chow_utility_score(
    bt: dict,
    target_trades: int = 6,
    chow_weight: float = 0.80,
    require_gate: bool = True,
    val_ret_pct: float | None = None,
    test_ret_pct: float | None = None,
) -> dict:
    """Compute Chow (1970) Abstention-Penalized + Omega/Sortino + Statistical Sufficiency self-evaluation score."""
    n_trades = int(bt.get("trades", 0))
    ret_pct = float(bt.get("total_return_pct", 0.0))
    max_dd = float(bt.get("max_drawdown_pct", 0.0))
    pf = float(bt.get("profit_factor") or 0.0)
    omega = float(bt.get("omega_ratio") or pf or 0.0)
    sortino = float(bt.get("sortino_ratio", 0.0))

    target_n = max(1, int(target_trades))
    stat_sufficiency = float(min(1.0, np.sqrt(n_trades / target_n)))
    abstention_rate = float(max(0.0, 1.0 - (n_trades / target_n)))
    chow_penalty = float(chow_weight * abstention_rate)

    gate_ok = True
    if require_gate:
        gate_ok = (
            n_trades >= 2
            and ret_pct > 0.0
            and (val_ret_pct is None or val_ret_pct >= 0.0)
            and (test_ret_pct is None or test_ret_pct > 0.0)
        )

    win_rate = float(bt.get("win_rate", 0.0))
    avg_win = float(bt.get("avg_win_usdt", 0.0))
    avg_loss = float(bt.get("avg_loss_usdt", 0.0))
    payoff_ratio = avg_win / max(0.25, avg_loss) if avg_loss > 0 else (3.0 if avg_win > 0 else 0.0)
    sample_ratio = float(min(1.0, n_trades / target_n))
    # Winner Reward / Penalty Index: strongly reward high win-rate, payoff asymmetry, and net profit
    winner_reward_index = (1.35 * (win_rate ** 1.3) + 0.30 * min(4.0, payoff_ratio)) * sample_ratio
    omega_bonus = 0.45 * float(np.log1p(min(10.0, max(0.0, omega)))) * sample_ratio
    sortino_bonus = 0.12 * float(min(5.0, max(0.0, sortino))) * sample_ratio
    dd_penalty = 1.15 * max_dd
    gate_term = 2.5 if gate_ok else -5.0

    utility = (1.30 * ret_pct * stat_sufficiency) + winner_reward_index + omega_bonus + sortino_bonus - dd_penalty - (1.80 * chow_penalty) + gate_term
    return {
        "chow_utility_score": round(float(utility), 4),
        "statistical_sufficiency": round(stat_sufficiency, 4),
        "abstention_rate": round(abstention_rate, 4),
        "chow_abstention_penalty": round(chow_penalty, 4),
        "omega_ratio": round(omega, 4),
        "paper_gate_passed": bool(gate_ok),
    }


def simulate(
    candles: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    cfg: StrategyConfig,
    use_aci: bool = True,
) -> dict:
    """Chronological, long-only, single-position, next-open simulated fills with trailing stops, BOCPD & ACI."""
    if probabilities is not None and len(candles) != len(probabilities):
        raise ValueError("Una probabilidad por vela es obligatoria")
    cfg.validate()
    if len(candles) == 0:
        return {
            "trades": 0, "total_return_pct": 0.0, "final_equity_usdt": cfg.initial_cash,
            "max_drawdown_pct": 0.0, "win_rate": 0.0, "profit_factor": None, "omega_ratio": None,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
            "expectancy_usdt": 0.0, "avg_win_usdt": 0.0, "avg_loss_usdt": 0.0,
            "chow_utility_score": -getattr(cfg, "chow_penalty_weight", 0.80),
            "statistical_sufficiency": 0.0, "chow_abstention_penalty": getattr(cfg, "chow_penalty_weight", 0.80),
            "final_aci_threshold": threshold, "closed_trades": [],
        }

    data = candles.reset_index(drop=True)
    probs = np.asarray(probabilities, dtype=float) if probabilities is not None else np.ones(len(data), dtype=float)
    aci = AdaptiveConformalThreshold(threshold, gamma=cfg.aci_gamma, target_alpha=cfg.aci_target_alpha)

    cash = cfg.initial_cash
    qty, entry_price, allocation = 0.0, 0.0, 0.0
    entry_time = -1
    scheduled = -1
    scheduled_prob = 0.0
    scheduled_thr = threshold
    peak_price = 0.0
    peak_equity = cfg.initial_cash
    max_dd = 0.0
    day = ""
    day_start = cfg.initial_cash
    halted_today = False
    halt_forever = False
    cooldown_until_ts = 0
    trades: list[Trade] = []

    opens = data["open"].to_numpy(dtype=float)
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
    closes = data["close"].to_numpy(dtype=float)
    open_times = data["open_time"].to_numpy(dtype=np.int64)

    has_ema50 = "ema50_1h" in data.columns
    has_ema20 = "ema20_1h" in data.columns
    ema_trend = data["ema50_1h"].to_numpy(dtype=float) if has_ema50 else (
        data["ema20_1h"].to_numpy(dtype=float) if has_ema20 else np.full(len(data), np.nan)
    )
    atr_pcts = data["atr_pct_1h"].to_numpy(dtype=float) if "atr_pct_1h" in data.columns else np.full(len(data), np.nan)
    vpins = data["vpin_1h"].to_numpy(dtype=float) if "vpin_1h" in data.columns else np.full(len(data), np.nan)
    bocpd_shocks = data["bocpd_shock_1h"].to_numpy(dtype=float) if "bocpd_shock_1h" in data.columns else np.zeros(len(data), dtype=float)
    ret_15ms = data["ret_15m"].to_numpy(dtype=float) if "ret_15m" in data.columns else np.zeros(len(data), dtype=float)
    ret_1hs = data["ret_1h"].to_numpy(dtype=float) if "ret_1h" in data.columns else np.zeros(len(data), dtype=float)
    vol_ratios = data["vol_ratio_1h"].to_numpy(dtype=float) if "vol_ratio_1h" in data.columns else np.ones(len(data), dtype=float)
    tbrs = data["taker_buy_ratio_1h"].to_numpy(dtype=float) if "taker_buy_ratio_1h" in data.columns else np.full(len(data), 0.52, dtype=float)
    is_hour_close = data["is_hour_close"].to_numpy(dtype=bool) if "is_hour_close" in data.columns else (
        ((open_times + MINUTE_MS) % (60 * MINUTE_MS) == 0)
    )

    if "hourly_trigger" in data.columns:
        raw_triggers = data["hourly_trigger"].to_numpy(dtype=bool)
        if not raw_triggers.any() and len(data) < FEATURE_WARMUP:
            triggers = np.ones(len(data), dtype=bool)
        else:
            triggers = raw_triggers
    else:
        triggers = np.ones(len(data), dtype=bool)

    slip_in = 1 + cfg.execution_bps / 10_000
    slip_out = 1 - cfg.execution_bps / 10_000
    fee_in = 1 - cfg.fee_rate
    fee_out = 1 - cfg.fee_rate

    def sell(i: int, raw_price: float, reason: str) -> None:
        nonlocal cash, qty, entry_price, entry_time, allocation, peak_price, cooldown_until_ts
        exit_p = raw_price * slip_out
        proceeds = qty * exit_p * fee_out
        pnl = proceeds - allocation
        ret_pct = (pnl / allocation) * 100.0 if allocation > 0 else 0.0
        cash += proceeds
        trades.append(Trade(
            entry_time=utc_iso(int(open_times[entry_time])),
            exit_time=utc_iso(int(open_times[i])),
            entry_price=float(entry_price),
            exit_price=float(exit_p),
            allocation=float(allocation),
            pnl=float(pnl),
            return_pct=round(float(ret_pct), 4),
            reason=reason,
        ))
        if use_aci and reason != "end_of_backtest":
            aci.update(pnl > 0.0, ret_pct)
        if reason != "end_of_backtest":
            cooldown_until_ts = int(open_times[i]) + 30 * MINUTE_MS
        qty, entry_price, allocation, entry_time, peak_price = 0.0, 0.0, 0.0, -1, 0.0

    for i in range(len(data)):
        b_o, b_h, b_l, b_c = opens[i], highs[i], lows[i], closes[i]
        t = int(open_times[i])

        if i > 0 and t - int(open_times[i - 1]) != MINUTE_MS:
            scheduled = -1
            if qty > 0:
                sell(i, b_o, "missing_data_gap")

        bar_day = utc_iso(t)[:10]
        if bar_day != day:
            day = bar_day
            day_start = cash + qty * b_o * fee_out
            halted_today = False

        if scheduled == i and qty == 0 and not halted_today and not halt_forever:
            pos_frac = _compute_position_fraction(
                scheduled_prob, scheduled_thr, atr_pcts[i], cfg, vpins[i], bocpd_shocks[i]
            )
            allocation = cash * pos_frac
            entry_price = b_o * slip_in
            qty = allocation * fee_in / entry_price
            cash -= allocation
            entry_time = i
            peak_price = entry_price
        scheduled = -1

        if qty > 0:
            if b_h > peak_price:
                peak_price = b_h

            # Identify Market Behavior Pattern (Active Impulse vs. Calm Accumulation vs. Passive Exhaustion)
            entry_is_calm_or_passive = (
                np.isfinite(vol_ratios[entry_time])
                and vol_ratios[entry_time] < 1.50
                and np.isfinite(tbrs[entry_time])
                and tbrs[entry_time] < 0.505
            )
            eff_trail_act = min(cfg.trailing_activation, 0.0058) if entry_is_calm_or_passive else cfg.trailing_activation
            eff_trail_delta = min(cfg.trailing_delta, 0.0012) if entry_is_calm_or_passive else cfg.trailing_delta

            trailing_active = peak_price >= entry_price * (1 + eff_trail_act)
            if trailing_active:
                stop_price = peak_price * (1 - eff_trail_delta)
            else:
                stop_price = entry_price * (1 - cfg.initial_stop)

            # Drastic Trend Reversal detection (BOCPD structural break + seller dominance or velocity drop)
            drastic_reversal = (
                i > entry_time
                and bocpd_shocks[i] >= cfg.bocpd_exit_threshold
                and (
                    (np.isfinite(vpins[i]) and vpins[i] >= cfg.vpin_exit_threshold and np.isfinite(ret_15ms[i]) and ret_15ms[i] <= -0.008)
                    or (np.isfinite(ret_1hs[i]) and ret_1hs[i] <= -0.0050 and np.isfinite(tbrs[i]) and tbrs[i] <= 0.45)
                )
            )

            if b_l <= stop_price:
                reason = "trailing_stop" if trailing_active else "stop_loss"
                sell(i, min(b_o, stop_price), reason)
            elif drastic_reversal:
                sell(i, b_c, "bocpd_shock_exit")
            elif is_hour_close[i] and (i > entry_time) and np.isfinite(ema_trend[i]) and (b_c < ema_trend[i]):
                sell(i, b_c, "trend_exit")
            elif i - entry_time + 1 >= cfg.horizon_bars:
                sell(i, b_c, "horizon")

        equity = cash + qty * b_c * slip_out * fee_out
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity
        max_dd = max(max_dd, dd)

        if equity <= day_start * (1 - cfg.daily_loss_limit):
            halted_today = True
            if qty > 0:
                sell(i, b_c, "daily_loss_limit")
        if dd >= cfg.max_drawdown:
            halt_forever = True
            if qty > 0:
                sell(i, b_c, "max_drawdown")

        if (i + 1 < len(data) and qty == 0 and not halted_today and not halt_forever
                and t >= cooldown_until_ts
                and int(open_times[i + 1]) - t == MINUTE_MS and triggers[i]):
            p = probs[i]
            eff_thr = aci.current_threshold(bocpd_shocks[i]) if use_aci else threshold
            if np.isfinite(p) and p >= eff_thr:
                scheduled = i + 1
                scheduled_prob = float(p)
                scheduled_thr = float(eff_thr)

    if qty > 0:
        sell(len(data) - 1, closes[-1], "end_of_backtest")

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [-t.pnl for t in trades if t.pnl < 0]
    gross_win = float(sum(wins))
    gross_loss = float(sum(losses))
    profit_factor = round(gross_win / gross_loss, 4) if gross_loss > 0 else (99.0 if gross_win > 0 else None)

    total_ret_pct = round(100 * (cash / cfg.initial_cash - 1), 4)
    max_dd_pct = round(max_dd * 100, 4)
    trade_returns = np.array([t.return_pct / 100.0 for t in trades], dtype=float)
    if len(trade_returns) >= 2 and np.std(trade_returns) > 1e-9:
        sharpe = float(np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(len(trade_returns)))
        downside = trade_returns[trade_returns < 0]
        down_std = float(np.std(downside)) if len(downside) >= 2 else (float(abs(downside[0])) if len(downside) == 1 else 1e-4)
        sortino = float(np.mean(trade_returns) / max(1e-6, down_std) * np.sqrt(len(trade_returns)))
    else:
        sharpe = 0.0
        sortino = 0.0
    calmar = round(total_ret_pct / max_dd_pct, 4) if max_dd_pct > 0 else 0.0
    expectancy = round(float(np.mean([t.pnl for t in trades])), 4) if trades else 0.0

    omega_ratio = profit_factor
    base_res = {
        "trades": len(trades),
        "total_return_pct": total_ret_pct,
        "final_equity_usdt": round(cash, 4),
        "max_drawdown_pct": max_dd_pct,
        "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
        "profit_factor": profit_factor,
        "omega_ratio": omega_ratio,
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": calmar,
        "expectancy_usdt": expectancy,
        "avg_win_usdt": round(gross_win / len(wins), 4) if wins else 0.0,
        "avg_loss_usdt": round(gross_loss / len(losses), 4) if losses else 0.0,
        "final_aci_threshold": round(aci.current_threshold(), 4),
        "closed_trades": [asdict(t) for t in trades],
    }
    chow_eval = compute_chow_utility_score(
        base_res,
        target_trades=getattr(cfg, "target_oos_trades", 6),
        chow_weight=getattr(cfg, "chow_penalty_weight", 0.80),
        require_gate=False,
    )
    base_res.update({
        "chow_utility_score": chow_eval["chow_utility_score"],
        "statistical_sufficiency": chow_eval["statistical_sufficiency"],
        "chow_abstention_penalty": chow_eval["chow_abstention_penalty"],
    })
    return base_res

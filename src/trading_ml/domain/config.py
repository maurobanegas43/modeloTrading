from __future__ import annotations

from dataclasses import dataclass


MINUTE_MS = 60_000
M15_MS = 15 * MINUTE_MS
HOUR_MS = 60 * MINUTE_MS
FEATURE_WARMUP = 3_000  # 50 hours of 1-minute bars minimum warmup


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "BTCUSDT"
    fee_rate: float = 0.001  # 0.10% per fill, conservative regular Spot baseline
    execution_bps: float = 3.0  # half-spread plus slippage per fill (3 bps on BTCUSDT)
    initial_cash: float = 1_000.0
    allocation: float = 0.25
    dynamic_sizing: bool = True  # volatility, VPIN, BOCPD & Fractional Kelly regime-adjusted sizing
    initial_stop: float = 0.012  # 1.20% initial stop loss
    trailing_activation: float = 0.015  # +1.50% profit activates trailing stop
    trailing_delta: float = 0.010  # trails by 1.00% behind peak price
    horizon_bars: int = 1440  # up to 24 hours max hold if macro trend holds
    breakout_lookback_hours: int = 8  # 8-hour close breakout confirmation
    min_macro_spread: float = 0.0005  # (EMA50_1h - EMA100_1h) / Close >= 5 bps macro trend filter
    enable_tactical_regime_b: bool = True  # enable Regime B (tactical absorption/continuation) + 30m intra-hour trigger
    target_oos_trades: int = 6  # target minimum OOS trade count for Chow abstention penalty
    chow_penalty_weight: float = 0.80  # Chow (1970) cost of abstention weight in non-toxic regimes
    max_entry_rsi: float = 70.0  # avoid buying overbought exhaustion spikes
    bocpd_hazard: float = 1.0 / 24.0  # expected prior changepoint hazard rate (1 per 24h)
    bocpd_exit_threshold: float = 0.85  # BOCPD shock probability threshold for flash-crash circuit breaker
    vpin_exit_threshold: float = 0.35  # VPIN order-flow toxicity threshold for flash-crash exit
    aci_gamma: float = 0.015  # Adaptive Conformal Inference step size (Gibbs & Candes 2021)
    aci_target_alpha: float = 0.30  # target miscoverage rate (70% target accuracy)
    daily_loss_limit: float = 0.02
    max_drawdown: float = 0.10
    threshold: float = 0.35  # base calibrated ensemble probability threshold
    stop_loss: float | None = None  # optional alias for backward compatibility

    def __post_init__(self) -> None:
        if self.stop_loss is not None:
            object.__setattr__(self, "initial_stop", float(self.stop_loss))

    def validate(self) -> None:
        if not self.symbol.isalnum() or self.symbol != self.symbol.upper():
            raise ValueError("symbol debe ser alfanumérico y estar en mayúsculas")
        if not 1 <= self.horizon_bars <= 10_080:
            raise ValueError("horizon_bars fuera de rango (1..10080)")
        if not 0 <= self.fee_rate < 0.02 or not 0 <= self.execution_bps < 100:
            raise ValueError("Comisiones o costos inválidos")
        if not 0 < self.allocation <= 1 or self.initial_cash <= 0:
            raise ValueError("Capital o asignación inválidos")
        if not 0 < self.initial_stop < 1:
            raise ValueError("Límite de stop loss inválido")
        if not 0 < self.trailing_activation < 1 or not 0 < self.trailing_delta < 1:
            raise ValueError("Parámetros de trailing stop inválidos")
        if not 0 < self.daily_loss_limit < 1 or not 0 < self.max_drawdown < 1:
            raise ValueError("Límites de riesgo de cuenta inválidos")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Umbral inválido")


@dataclass(frozen=True)
class FuturesConfig:
    """Parámetros conservadores para investigación y paper de futuros USDⓈ-M lineales."""

    symbol: str = "BTCUSDT"
    initial_balance: float = 1_000.0
    fee_rate: float = 0.0005  # taker fee conservadora por lado
    execution_bps: float = 3.0
    leverage: float = 1.0
    max_margin_fraction: float = 0.25
    maintenance_margin_rate: float = 0.005
    liquidation_fee_rate: float = 0.005
    # The model evaluates every completed 15m candle. 1h features remain context,
    # never a separate entry veto.
    decision_interval_minutes: int = 15
    holding_horizon_minutes: int = 60
    min_net_edge_bps: float = 5.0
    probability_threshold: float = 0.40
    initial_stop: float = 0.012
    trailing_activation: float = 0.015
    trailing_delta: float = 0.010
    daily_loss_limit: float = 0.02
    max_drawdown: float = 0.10

    def validate(self) -> None:
        if not self.symbol.isalnum() or self.symbol != self.symbol.upper():
            raise ValueError("symbol debe ser alfanumérico y estar en mayúsculas")
        if self.initial_balance <= 0 or not 0 < self.leverage <= 5:
            raise ValueError("capital o leverage fuera de rango; leverage paper máximo: 5x")
        if not 0 < self.max_margin_fraction <= 1:
            raise ValueError("max_margin_fraction debe estar en (0, 1]")
        if not 0 <= self.fee_rate < 0.02 or not 0 <= self.execution_bps < 100:
            raise ValueError("comisiones o costos inválidos")
        if not 0 < self.maintenance_margin_rate < 0.1:
            raise ValueError("maintenance_margin_rate inválido")
        if not 0 <= self.liquidation_fee_rate < 0.1:
            raise ValueError("liquidation_fee_rate inválido")
        if (
            self.decision_interval_minutes < 1
            or self.decision_interval_minutes > 60
            or 60 % self.decision_interval_minutes != 0
            or self.holding_horizon_minutes < 1
        ):
            raise ValueError("decision_interval_minutes debe dividir la hora; los intervalos deben ser positivos")
        if self.min_net_edge_bps < 0 or not 0 <= self.probability_threshold <= 1:
            raise ValueError("umbral de edge o probabilidad inválido")
        if not 0 < self.initial_stop < 1:
            raise ValueError("initial_stop inválido")
        if not 0 < self.trailing_activation < 1 or not 0 < self.trailing_delta < 1:
            raise ValueError("trailing stop inválido")
        if not 0 < self.daily_loss_limit < 1 or not 0 < self.max_drawdown < 1:
            raise ValueError("límites de riesgo inválidos")

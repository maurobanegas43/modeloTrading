from __future__ import annotations

from dataclasses import dataclass


MINUTE_MS = 60_000
FEATURE_WARMUP = 65


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "BTCUSDT"
    horizon_bars: int = 15
    fee_rate: float = 0.001  # 0.10% per fill, conservative regular Spot baseline
    execution_bps: float = 5.0  # half-spread plus slippage, EACH fill
    initial_cash: float = 1_000.0
    allocation: float = 0.25
    stop_loss: float = 0.015
    daily_loss_limit: float = 0.02
    max_drawdown: float = 0.10
    threshold: float = 0.60

    def validate(self) -> None:
        if not self.symbol.isalnum() or self.symbol != self.symbol.upper():
            raise ValueError("symbol debe ser alfanumérico y estar en mayúsculas")
        if not 1 <= self.horizon_bars <= 240:
            raise ValueError("horizon_bars fuera de rango")
        if not 0 <= self.fee_rate < 0.02 or not 0 <= self.execution_bps < 100:
            raise ValueError("Comisiones o costos inválidos")
        if not 0 < self.allocation <= 1 or self.initial_cash <= 0:
            raise ValueError("Capital o asignación inválidos")
        if not 0 < self.stop_loss < 1 or not 0 < self.daily_loss_limit < 1:
            raise ValueError("Límites de riesgo inválidos")
        if not 0 < self.max_drawdown < 1 or not 0 <= self.threshold <= 1:
            raise ValueError("Umbrales inválidos")


"""Compatibilidad de importación; la implementación vive en ``domain``."""

from .domain.backtest import (
    AdaptiveConformalThreshold,
    Trade,
    _compute_position_fraction,
    compute_chow_utility_score,
    simulate,
)

__all__ = [
    "AdaptiveConformalThreshold", "Trade", "_compute_position_fraction",
    "compute_chow_utility_score", "simulate",
]

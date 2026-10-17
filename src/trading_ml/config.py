"""Compatibilidad de importación; la implementación vive en ``domain``."""

from .domain.config import FEATURE_WARMUP, HOUR_MS, M15_MS, MINUTE_MS, StrategyConfig

__all__ = ["FEATURE_WARMUP", "HOUR_MS", "M15_MS", "MINUTE_MS", "StrategyConfig"]

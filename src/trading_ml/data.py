"""Compatibilidad de importación; el adaptador vive en ``adapters``."""

from .adapters.market_data import (
    COLUMNS,
    FLOATS,
    download_history,
    normalize_klines,
    read_history,
    recent_closed_candles,
    utc_iso,
)

__all__ = [
    "COLUMNS", "FLOATS", "download_history", "normalize_klines",
    "read_history", "recent_closed_candles", "utc_iso",
]

"""Contexto causal propio del perpetual USDⓈ-M."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES


FUTURES_FEATURES = FEATURES + [
    "momentum_15m_atr_units",
    "mark_basis_bps",
    "last_funding_rate_bps",
]


def attach_futures_context(frame: pd.DataFrame, funding: pd.DataFrame | None) -> pd.DataFrame:
    """Añade momentum 15m normalizado, basis y funding ya conocido sin lookahead."""
    result = frame.copy()
    ret_15m = pd.to_numeric(result["ret_15m"], errors="coerce").to_numpy(dtype=float)
    atr_1h = pd.to_numeric(result["atr_pct_1h"], errors="coerce").to_numpy(dtype=float)
    expected_15m_vol = np.maximum(atr_1h * np.sqrt(15.0 / 60.0), 1e-6)
    momentum_units = ret_15m / expected_15m_vol
    result["momentum_15m_atr_units"] = np.clip(
        np.where(np.isfinite(momentum_units), momentum_units, 0.0), -8.0, 8.0
    )
    if "mark_close" in result:
        mark = pd.to_numeric(result["mark_close"], errors="coerce").to_numpy(dtype=float)
        close = result["close"].to_numpy(dtype=float)
        basis = (mark - close) / np.maximum(close, 1e-12) * 10_000
        result["mark_basis_bps"] = np.where(np.isfinite(basis), basis, 0.0)
    else:
        result["mark_basis_bps"] = 0.0

    rates = np.zeros(len(result), dtype=float)
    if funding is not None and len(funding):
        times = pd.to_numeric(funding["fundingTime"], errors="raise").to_numpy(dtype=np.int64)
        values = pd.to_numeric(funding["fundingRate"], errors="raise").to_numpy(dtype=float)
        order = np.argsort(times)
        times, values = times[order], values[order]
        # La barra de decisión todavía no conoce un pago publicado después de su apertura.
        idx = np.searchsorted(times, result["open_time"].to_numpy(dtype=np.int64), side="right") - 1
        known = idx >= 0
        rates[known] = values[idx[known]] * 10_000
    result["last_funding_rate_bps"] = rates
    return result

"""Datos públicos históricos para Binance USDⓈ-M perpetual futures."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from ..domain.config import MINUTE_MS
from .market_data import COLUMNS


BASE_URL = "https://fapi.binance.com"


def _get_json(path: str, params: dict, attempts: int = 5):
    url = BASE_URL + path + "?" + urlencode(params)
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "binance-ml-usdm-paper-research/1.0"})
            with urlopen(request, timeout=20) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise RuntimeError(f"Binance USDⓈ-M HTTP {exc.code}: {exc.reason}") from exc
            retry = exc.headers.get("Retry-After")
            delay = min(60.0, float(retry)) if retry and retry.replace(".", "", 1).isdigit() else min(2**attempt, 30)
        except (URLError, TimeoutError, OSError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"No se pudo conectar con Binance USDⓈ-M: {exc}") from exc
            delay = min(2**attempt, 30)
        time.sleep(delay)
    raise RuntimeError("Falló la descarga de datos USDⓈ-M")


def _normalize_klines(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLUMNS[:-1])
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame = frame.drop(columns=["ignore"])
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="raise").astype("int64")
    frame["close_time"] = pd.to_numeric(frame["close_time"], errors="raise").astype("int64")
    frame["trades"] = pd.to_numeric(frame["trades"], errors="raise").astype("int64")
    float_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]
    for column in float_cols:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return frame.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)


def download_usdm_history(symbol: str = "BTCUSDT", days: int = 90) -> pd.DataFrame:
    """Descarga klines 1m del perpetual; excluye siempre la vela abierta."""
    if not symbol.isalnum() or symbol != symbol.upper():
        raise ValueError("symbol debe ser alfanumérico y estar en mayúsculas")
    if not 1 <= days <= 200:
        raise ValueError("days debe estar entre 1 y 200 (límite histórico del endpoint)")
    end_ms = (int(time.time() * 1000) // MINUTE_MS) * MINUTE_MS
    start_ms = end_ms - days * 24 * 60 * MINUTE_MS
    cursor = start_ms
    rows: list[list] = []
    while cursor < end_ms:
        batch = _get_json("/fapi/v1/klines", {
            "symbol": symbol,
            "interval": "1m",
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": 1500,
        })
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + MINUTE_MS
        if next_cursor <= cursor:
            raise RuntimeError("La API no avanzó; se interrumpe para evitar un bucle")
        cursor = next_cursor
        if len(rows) % 15_000 == 0 or cursor >= end_ms:
            print(f"Klines USDⓈ-M descargadas: {len(rows):,}", flush=True)
        time.sleep(0.08)
    frame = _normalize_klines(rows)
    frame = frame.loc[(frame["open_time"] >= start_ms) & (frame["open_time"] < end_ms)].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("Binance no devolvió klines USDⓈ-M")
    return frame


def download_usdm_mark_history(symbol: str = "BTCUSDT", days: int = 90) -> pd.DataFrame:
    """Descarga mark-price OHLC 1m para medir liquidación como en USDⓈ-M."""
    if not symbol.isalnum() or symbol != symbol.upper():
        raise ValueError("symbol debe ser alfanumérico y estar en mayúsculas")
    if not 1 <= days <= 200:
        raise ValueError("days debe estar entre 1 y 200")
    end_ms = (int(time.time() * 1000) // MINUTE_MS) * MINUTE_MS
    start_ms = end_ms - days * 24 * 60 * MINUTE_MS
    cursor = start_ms
    rows: list[list] = []
    while cursor < end_ms:
        batch = _get_json("/fapi/v1/markPriceKlines", {
            "symbol": symbol,
            "interval": "1m",
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": 1500,
        })
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + MINUTE_MS
        if next_cursor <= cursor:
            raise RuntimeError("La API de mark price no avanzó; se interrumpe")
        cursor = next_cursor
        if len(rows) % 15_000 == 0 or cursor >= end_ms:
            print(f"Klines mark-price descargadas: {len(rows):,}", flush=True)
        time.sleep(0.08)
    if not rows:
        raise RuntimeError("Binance no devolvió klines mark-price")
    mark = pd.DataFrame(rows, columns=["open_time", "mark_open", "mark_high", "mark_low", "mark_close"] + [f"_unused_{i}" for i in range(len(rows[0]) - 5)])
    mark = mark[["open_time", "mark_open", "mark_high", "mark_low", "mark_close"]]
    mark["open_time"] = pd.to_numeric(mark["open_time"], errors="raise").astype("int64")
    for column in ("mark_open", "mark_high", "mark_low", "mark_close"):
        mark[column] = pd.to_numeric(mark[column], errors="raise").astype(float)
    return mark.loc[(mark["open_time"] >= start_ms) & (mark["open_time"] < end_ms)].drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def download_usdm_funding_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Descarga la historia real de pagos de funding, ordenada por fundingTime."""
    if end_ms <= start_ms:
        raise ValueError("end_ms debe ser posterior a start_ms")
    cursor = int(start_ms)
    rows: list[dict] = []
    while cursor <= end_ms:
        batch = _get_json("/fapi/v1/fundingRate", {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": int(end_ms),
            "limit": 1000,
        })
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("La API de funding no avanzó; se interrumpe para evitar un bucle")
        cursor = next_cursor
        if len(batch) < 1000:
            break
        time.sleep(0.08)
    frame = pd.DataFrame(rows, columns=["symbol", "fundingRate", "fundingTime", "markPrice", "rateType"])
    if frame.empty:
        return pd.DataFrame(columns=["fundingTime", "fundingRate", "markPrice"])
    frame["fundingTime"] = pd.to_numeric(frame["fundingTime"], errors="raise").astype("int64")
    frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="raise").astype(float)
    frame["markPrice"] = pd.to_numeric(frame["markPrice"], errors="coerce")
    return frame.sort_values("fundingTime").drop_duplicates("fundingTime", keep="last").reset_index(drop=True)


def read_usdm_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas USDⓈ-M: {sorted(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["open_time"] = frame["open_time"].astype("int64")
    frame = frame.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Los precios USDⓈ-M deben ser positivos")
    return frame


def read_usdm_funding(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    required = {"fundingTime", "fundingRate"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Faltan columnas funding: {sorted(required - set(frame.columns))}")
    frame["fundingTime"] = pd.to_numeric(frame["fundingTime"], errors="raise").astype("int64")
    frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="raise").astype(float)
    if "markPrice" in frame:
        frame["markPrice"] = pd.to_numeric(frame["markPrice"], errors="coerce")
    else:
        frame["markPrice"] = float("nan")
    return frame.sort_values("fundingTime").reset_index(drop=True)


def attach_mark_prices(candles: pd.DataFrame, mark_path: Path | None) -> pd.DataFrame:
    if mark_path is None:
        return candles
    mark = pd.read_csv(mark_path)
    required = {"open_time", "mark_open", "mark_high", "mark_low", "mark_close"}
    if not required.issubset(mark.columns):
        raise ValueError(f"Faltan columnas mark-price: {sorted(required - set(mark.columns))}")
    mark["open_time"] = pd.to_numeric(mark["open_time"], errors="raise").astype("int64")
    for column in ("mark_open", "mark_high", "mark_low", "mark_close"):
        mark[column] = pd.to_numeric(mark[column], errors="coerce")
    return candles.merge(mark[list(required)], on="open_time", how="left", validate="one_to_one")

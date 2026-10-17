from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from ..domain.config import MINUTE_MS
from ..domain.time import utc_iso


COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
FLOATS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]
BASE_URL = "https://api.binance.com"


def _get_json(path: str, params: dict, attempts: int = 5):
    url = BASE_URL + path + "?" + urlencode(params)
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers={"User-Agent": "binance-ml-paper-research/0.3"}), timeout=20) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise RuntimeError(f"Binance HTTP {exc.code}: {exc.reason}") from exc
            retry = exc.headers.get("Retry-After")
            delay = min(60.0, float(retry)) if retry and retry.replace(".", "", 1).isdigit() else min(2**attempt, 30)
        except (URLError, TimeoutError, OSError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"No se pudo conectar con Binance: {exc}") from exc
            delay = min(2**attempt, 30)
        time.sleep(delay)
    raise RuntimeError("Falló la descarga")


def normalize_klines(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLUMNS[:-1])
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame = frame.drop(columns=["ignore"])
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="raise").astype("int64")
    frame["close_time"] = pd.to_numeric(frame["close_time"], errors="raise").astype("int64")
    frame["trades"] = pd.to_numeric(frame["trades"], errors="raise").astype("int64")
    for col in FLOATS + ["taker_buy_quote_volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    return frame.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)


def download_history(symbol: str, days: int, existing_path: Path | None = None) -> pd.DataFrame:
    if not 1 <= days <= 730:
        raise ValueError("days debe estar entre 1 y 730")
    now_ms = int(time.time() * 1000)
    end_ms = (now_ms // MINUTE_MS) * MINUTE_MS  # exclude currently open candle
    start_ms = end_ms - days * 24 * 60 * MINUTE_MS

    cached_df: pd.DataFrame | None = None
    if existing_path is not None and existing_path.is_file():
        try:
            cached_df = read_history(existing_path)
            if len(cached_df) > 0 and int(cached_df.iloc[0]["open_time"]) <= start_ms:
                last_cached = int(cached_df.iloc[-1]["open_time"])
                if last_cached >= start_ms:
                    start_ms = last_cached + MINUTE_MS
        except Exception:
            cached_df = None

    all_rows: list = []
    cursor = start_ms
    while cursor < end_ms:
        rows = _get_json("/api/v3/klines", {
            "symbol": symbol, "interval": "1m", "startTime": cursor,
            "endTime": end_ms - 1, "limit": 1000,
        })
        if not rows:
            break
        all_rows.extend(rows)
        next_cursor = int(rows[-1][0]) + MINUTE_MS
        if next_cursor <= cursor:
            raise RuntimeError("La API no avanzó; se interrumpe para evitar un bucle")
        cursor = next_cursor
        time.sleep(0.12)

    fresh = normalize_klines(all_rows)
    fresh = fresh.loc[fresh["open_time"] < end_ms].copy()
    if cached_df is not None and len(cached_df) > 0:
        combined = pd.concat([cached_df, fresh], ignore_index=True)
        combined = combined.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)
        min_cutoff = end_ms - days * 24 * 60 * MINUTE_MS
        return combined.loc[combined["open_time"] >= min_cutoff].reset_index(drop=True)
    return fresh


def recent_closed_candles(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch closed 1-minute candles via REST, supporting multi-page warmup (>1000 bars)."""
    current_open_ms = (int(time.time() * 1000) // MINUTE_MS) * MINUTE_MS
    if limit <= 1000:
        rows = _get_json("/api/v3/klines", {"symbol": symbol, "interval": "1m", "limit": limit})
        frame = normalize_klines(rows)
        return frame.loc[frame["open_time"] < current_open_ms].copy()

    start_ms = current_open_ms - limit * MINUTE_MS
    all_rows: list = []
    cursor = start_ms
    while cursor < current_open_ms:
        batch_limit = min(1000, max(1, (current_open_ms - cursor) // MINUTE_MS))
        rows = _get_json("/api/v3/klines", {
            "symbol": symbol, "interval": "1m", "startTime": cursor,
            "endTime": current_open_ms - 1, "limit": batch_limit,
        })
        if not rows:
            break
        all_rows.extend(rows)
        next_cursor = int(rows[-1][0]) + MINUTE_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.08)
    frame = normalize_klines(all_rows)
    return frame.loc[frame["open_time"] < current_open_ms].copy()


def read_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}. Ejecutá primero 'download'.")
    df = pd.read_csv(path)
    required = {"open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas: {sorted(required - set(df.columns))}")
    for key in required:
        df[key] = pd.to_numeric(df[key], errors="raise")
    df["open_time"] = df["open_time"].astype("int64")
    df = df.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Los precios deben ser positivos")
    return df



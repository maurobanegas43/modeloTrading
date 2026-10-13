from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .config import MINUTE_MS


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
            with urlopen(Request(url, headers={"User-Agent": "binance-ml-paper-research/0.1"}), timeout=20) as response:
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
    return frame.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)


def download_history(symbol: str, days: int) -> pd.DataFrame:
    if not 1 <= days <= 730:
        raise ValueError("days debe estar entre 1 y 730")
    now_ms = int(time.time() * 1000)
    end_ms = (now_ms // MINUTE_MS) * MINUTE_MS  # exclude currently open candle
    start_ms = end_ms - days * 24 * 60 * MINUTE_MS
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
    frame = normalize_klines(all_rows)
    return frame.loc[frame["open_time"] < end_ms].copy()


def recent_closed_candles(symbol: str, limit: int = 250) -> pd.DataFrame:
    rows = _get_json("/api/v3/klines", {"symbol": symbol, "interval": "1m", "limit": limit})
    frame = normalize_klines(rows)
    current_open_ms = (int(time.time() * 1000) // MINUTE_MS) * MINUTE_MS
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


def utc_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


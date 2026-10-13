from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_WARMUP, MINUTE_MS, StrategyConfig


FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_15", "range_pct", "body_pct",
    "vol_ratio", "taker_buy_ratio", "realized_vol_15", "ema_5_20",
    "ema_20_60", "rsi_14", "hour_sin", "hour_cos",
]


def feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.sort_values("open_time").drop_duplicates("open_time", keep="last").copy()
    df = df.reset_index(drop=True)
    close = df["close"].astype(float)
    op = df["open"].astype(float)
    volume = df["volume"].astype(float)
    log_close = np.log(close)
    for n in (1, 3, 5, 15):
        df[f"ret_{n}"] = log_close.diff(n)
    df["range_pct"] = (df["high"] - df["low"]) / close
    df["body_pct"] = (close - op) / op
    df["vol_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    df["taker_buy_ratio"] = df["taker_buy_volume"] / volume.replace(0, np.nan)
    df["realized_vol_15"] = log_close.diff().rolling(15).std()
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    df["ema_5_20"] = ema5 / ema20 - 1
    df["ema_20_60"] = ema20 / ema60 - 1
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi_14"] = 100 * gain / (gain + loss).replace(0, np.nan)
    hour = (df["open_time"] // 3_600_000) % 24
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # Never bridge a missing candle or use the partially formed current bar.
    continuity = df["open_time"].diff().eq(MINUTE_MS)
    df["valid_features"] = continuity.rolling(FEATURE_WARMUP - 1, min_periods=FEATURE_WARMUP - 1).sum().eq(FEATURE_WARMUP - 1)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df["valid_features"] &= df[FEATURES].notna().all(axis=1)
    return df


def with_labels(candles: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = feature_frame(candles)
    h = cfg.horizon_bars
    entry = df["open"].shift(-1) * (1 + cfg.execution_bps / 10_000)
    exit_px = df["close"].shift(-h) * (1 - cfg.execution_bps / 10_000)
    df["net_return"] = exit_px * (1 - cfg.fee_rate) / (entry * (1 + cfg.fee_rate)) - 1
    df["target"] = (df["net_return"] > 0).astype(int)
    # Require h consecutive minute bars between decision and label maturity.
    forward_continuity = df["open_time"].shift(-h) - df["open_time"] == h * MINUTE_MS
    return df.loc[df["valid_features"] & forward_continuity & df["net_return"].notna()].reset_index(drop=True)


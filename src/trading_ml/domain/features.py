from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_WARMUP, HOUR_MS, MINUTE_MS, FuturesConfig, StrategyConfig


FEATURES = [
    "ret_15m",
    "ret_1h",
    "ret_4h",
    "ret_12h",
    "ret_24h",
    "trend_spread",
    "macro_spread",
    "macd_hist_1h",
    "vol_ratio_1h",
    "taker_buy_ratio_1h",
    "taker_imbalance_1h",
    "vpin_1h",
    "bocpd_shock_1h",
    "atr_pct_1h",
    "rsi_14_1h",
    "donchian_pos_10h",
    "hour_sin",
    "hour_cos",
]


def interval_decision_mask(frame: pd.DataFrame, interval_minutes: int) -> np.ndarray:
    """Mark completed bars whose close lands on the configured decision interval."""
    if interval_minutes < 1 or interval_minutes > 60 or 60 % interval_minutes != 0:
        raise ValueError("interval_minutes debe dividir la hora y estar entre 1 y 60")
    times = frame["open_time"].to_numpy(dtype=np.int64)
    valid = frame.get("valid_features", pd.Series(False, index=frame.index)).to_numpy(dtype=bool)
    return (((times + MINUTE_MS) % (interval_minutes * MINUTE_MS)) == 0) & valid


def _compute_bocpd_shock_prob(returns: pd.Series, hazard: float = 1.0 / 24.0) -> pd.Series:
    """Compute causal Bayesian Online Changepoint Detection (BOCPD) shock probability P(r_t = 0)."""
    r = returns.to_numpy(dtype=float)
    n = len(r)
    shock_probs = np.zeros(n, dtype=float)
    mean_est = 0.0
    var_est = 1e-5
    run_len = 1.0

    for i in range(n):
        x = r[i]
        if not np.isfinite(x):
            shock_probs[i] = 0.0
            continue
        std_est = np.sqrt(max(1e-8, var_est))
        z = abs(x - mean_est) / std_est
        surprise = 1.0 - np.exp(-0.5 * min(25.0, z * z))
        post_cp = np.clip(hazard + (1.0 - hazard) * (surprise ** 2) * (1.0 if z > 1.5 else 0.25), 0.0, 1.0)
        shock_probs[i] = float(post_cp)

        alpha = max(1.0 / (run_len + 1.0), 0.08 + 0.50 * post_cp)
        diff = x - mean_est
        mean_est += alpha * diff
        var_est = (1.0 - alpha) * var_est + alpha * (diff * diff)
        run_len = (1.0 - post_cp) * (run_len + 1.0) + post_cp * 1.0

    return pd.Series(shock_probs, index=returns.index, dtype=float)


def feature_frame(candles: pd.DataFrame, cfg: StrategyConfig | None = None) -> pd.DataFrame:
    """Build strictly causal multi-scale (15m / 1h / 4h / 24h) + VPIN + BOCPD features."""
    if cfg is None:
        cfg = StrategyConfig()
    df = candles.sort_values("open_time").drop_duplicates("open_time", keep="last").copy()
    df = df.reset_index(drop=True)
    if len(df) == 0:
        for col in FEATURES + ["valid_features", "is_hour_close", "is_hour_start", "hourly_trigger", "ema20_1h", "ema50_1h"]:
            df[col] = pd.Series(dtype=float)
        return df

    close_1m = df["close"].astype(float)
    df["ret_15m"] = close_1m.pct_change(15)

    df["hour_bucket"] = (df["open_time"] // HOUR_MS) * HOUR_MS
    df_1h = df.groupby("hour_bucket", as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        taker_buy_volume=("taker_buy_volume", "sum"),
        bar_count=("open", "count"),
    )

    c_h = df_1h["close"].astype(float)
    h_h = df_1h["high"].astype(float)
    l_h = df_1h["low"].astype(float)
    v_h = df_1h["volume"].astype(float)
    tb_h = df_1h["taker_buy_volume"].astype(float)

    ema20_h = c_h.ewm(span=20, adjust=False).mean()
    ema50_h = c_h.ewm(span=50, adjust=False).mean()
    ema100_h = c_h.ewm(span=100, adjust=False).mean()
    macd_h = c_h.ewm(span=12, adjust=False).mean() - c_h.ewm(span=26, adjust=False).mean()
    macd_sig_h = macd_h.ewm(span=9, adjust=False).mean()
    hist_h = macd_h - macd_sig_h

    prev_c_h = c_h.shift(1).fillna(df_1h["open"].astype(float))
    tr_h = np.maximum(h_h - l_h, np.maximum((h_h - prev_c_h).abs(), (l_h - prev_c_h).abs()))
    atr14_h = tr_h.ewm(alpha=1 / 14, adjust=False).mean()

    delta_h = c_h.diff().fillna(0.0)
    gain_h = delta_h.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss_h = (-delta_h.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi14_h = (100 * gain_h / (gain_h + loss_h).replace(0, np.nan)).fillna(50.0)

    lookback_h = max(2, int(cfg.breakout_lookback_hours))
    high10_h = h_h.rolling(10, min_periods=1).max().shift(1).fillna(h_h)
    low10_h = l_h.rolling(10, min_periods=1).min().shift(1).fillna(l_h)
    close_brk_h = c_h.rolling(lookback_h, min_periods=1).max().shift(1)

    df_1h["ret_1h"] = c_h.pct_change(1).fillna(0.0)
    df_1h["ret_4h"] = c_h.pct_change(4).fillna(df_1h["ret_1h"])
    df_1h["ret_12h"] = c_h.pct_change(12).fillna(df_1h["ret_4h"])
    df_1h["ret_24h"] = c_h.pct_change(24).fillna(df_1h["ret_12h"])
    df_1h["trend_spread"] = (ema20_h - ema50_h) / c_h
    df_1h["macro_spread"] = (ema50_h - ema100_h) / c_h
    df_1h["macd_hist_1h"] = hist_h / c_h
    vol_ma = v_h.rolling(20, min_periods=1).mean().replace(0, np.nan)
    df_1h["vol_ratio_1h"] = (v_h / vol_ma).fillna(1.0)
    df_1h["taker_buy_ratio_1h"] = (tb_h / v_h.replace(0, np.nan)).fillna(0.5)
    tbr_ma = df_1h["taker_buy_ratio_1h"].rolling(24, min_periods=1).mean()
    df_1h["taker_imbalance_1h"] = (df_1h["taker_buy_ratio_1h"] - tbr_ma).fillna(0.0)

    abs_order_imbalance = (2.0 * tb_h - v_h).abs()
    rolling_imb_sum = abs_order_imbalance.rolling(12, min_periods=1).sum()
    rolling_vol_sum = v_h.rolling(12, min_periods=1).sum().replace(0, np.nan)
    df_1h["vpin_1h"] = (rolling_imb_sum / rolling_vol_sum).fillna(0.10).clip(0.0, 1.0)

    df_1h["bocpd_shock_1h"] = _compute_bocpd_shock_prob(df_1h["ret_1h"], hazard=cfg.bocpd_hazard).clip(0.0, 1.0)

    df_1h["atr_pct_1h"] = (atr14_h / c_h).fillna(0.002)
    df_1h["rsi_14_1h"] = rsi14_h
    denom = (high10_h - low10_h).replace(0, np.nan)
    df_1h["donchian_pos_10h"] = ((c_h - low10_h) / denom).fillna(0.5)
    df_1h["ema20_1h"] = ema20_h
    df_1h["ema50_1h"] = ema50_h

    toxic_1h = (
        ((df_1h["bocpd_shock_1h"] >= cfg.bocpd_exit_threshold) & (df_1h["ret_1h"] < 0))

    )
    setup_regime_a = (
        (c_h > ema50_h)
        & (ema20_h > ema50_h)
        & (hist_h > 0)
        & (c_h > close_brk_h)
        & (df_1h["macro_spread"] >= cfg.min_macro_spread)
        & (df_1h["rsi_14_1h"] <= cfg.max_entry_rsi)
        & (~toxic_1h)
    )
    df_1h["trigger_1h"] = (setup_regime_a & (~setup_regime_a.shift(1, fill_value=False))).astype(int)

    df["completed_hour_bucket"] = ((df["open_time"] + MINUTE_MS) // HOUR_MS - 1) * HOUR_MS
    cols_1h = [
        "hour_bucket", "ret_1h", "ret_4h", "ret_12h", "ret_24h",
        "trend_spread", "macro_spread", "macd_hist_1h", "vol_ratio_1h",
        "taker_buy_ratio_1h", "taker_imbalance_1h", "vpin_1h", "bocpd_shock_1h",
        "atr_pct_1h", "rsi_14_1h", "donchian_pos_10h", "ema20_1h", "ema50_1h", "trigger_1h",
    ]
    df = pd.merge_asof(
        df.sort_values("completed_hour_bucket"),
        df_1h[cols_1h].sort_values("hour_bucket"),
        left_on="completed_hour_bucket",
        right_on="hour_bucket",
        direction="backward",
        suffixes=("", "_h"),
    ).sort_values("open_time").reset_index(drop=True)
    drop_cols = [c for c in ("hour_bucket", "hour_bucket_h", "completed_hour_bucket") if c in df.columns]
    df = df.drop(columns=drop_cols)
    df["vpin_1h"] = df["vpin_1h"].fillna(0.10).clip(0.0, 1.0)
    df["bocpd_shock_1h"] = df["bocpd_shock_1h"].fillna(0.0).clip(0.0, 1.0)

    decision_hour = ((df["open_time"] + MINUTE_MS) // HOUR_MS) % 24
    df["hour_sin"] = np.sin(2 * np.pi * decision_hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * decision_hour / 24)

    continuity = df["open_time"].diff().eq(MINUTE_MS)
    warmup_bars = FEATURE_WARMUP if len(df) >= FEATURE_WARMUP * 2 else 65
    no_recent_gap = continuity.rolling(64, min_periods=64).sum().eq(64)

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df["valid_features"] = df[FEATURES].notna().all(axis=1) & no_recent_gap & (df.index >= warmup_bars - 1)

    df["is_hour_close"] = ((df["open_time"] + MINUTE_MS) % HOUR_MS == 0)
    df["is_hour_start"] = (df["open_time"] % HOUR_MS == 0)
    is_30m_close = ((df["open_time"] + MINUTE_MS) % (30 * MINUTE_MS) == 0)

    # Classify 3 Quantitative Regimes:
    # -1 = Regime C (Toxic Cascade / Freefall Shield -> 100% Cash)
    #  2 = Regime A (Macro Bull Expansion Breakout -> Risk-On Fractional Kelly)
    #  1 = Regime B (Tactical Absorption / Continuation -> Calibrated Risk)
    #  0 = Neutral / Wait
    is_toxic = (
        (df["bocpd_shock_1h"] >= 0.45)
        | (df["vpin_1h"] >= 0.60)
        | (df["ret_15m"] <= -0.0030)
    )
    is_reg_a = (
        (~is_toxic)
        & (df["close"] > df["ema50_1h"])
        & (df["ema20_1h"] > df["ema50_1h"])
        & (df["macd_hist_1h"] > 0)
        & (df["macro_spread"] >= cfg.min_macro_spread)
    )
    is_reg_b = (
        (~is_toxic)
        & (~is_reg_a)
        & (df["close"] > df["ema20_1h"])
        & (df["macd_hist_1h"] > 0)
        & (df["macro_spread"] >= cfg.min_macro_spread * 0.5)
        & (df["donchian_pos_10h"] >= 0.85)
        & (df["taker_buy_ratio_1h"] >= 0.505)
    )
    df["regime_code"] = np.select([is_toxic, is_reg_a, is_reg_b], [-1, 2, 1], default=0)

    base_trigger = df["is_hour_close"] & (df["trigger_1h"] == 1) & df["valid_features"]
    df["primary_trigger"] = base_trigger.fillna(False).astype(bool)
    if getattr(cfg, "enable_tactical_regime_b", True):
        reg_b_trigger = (
            df["is_hour_close"]
            & df["valid_features"]
            & (df["close"] > df["ema50_1h"])
            & (df["ema20_1h"] > df["ema50_1h"])
            & (df["macd_hist_1h"] > 0)
            & (df["macro_spread"] >= cfg.min_macro_spread * 0.5)
            & (df["donchian_pos_10h"] >= 0.85)
            & (df["taker_buy_ratio_1h"] >= 0.505)
            & (df["bocpd_shock_1h"] < 0.45)
        )
        reg_a_30m_trigger = (
            is_30m_close
            & df["valid_features"]
            & (df["trigger_1h"] == 1)
            & (df["ret_15m"] >= 0.0008)
            & (df["bocpd_shock_1h"] < 0.40)
        )
        df["hourly_trigger"] = (base_trigger | reg_b_trigger | reg_a_30m_trigger).fillna(False).astype(bool)
    else:
        df["hourly_trigger"] = base_trigger.fillna(False).astype(bool)
    return df


def with_labels(candles: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Labels candidate signals using Triple-Barrier / Asymmetric Trailing Stop + BOCPD Shock Exit."""
    df = feature_frame(candles, cfg)
    n = len(df)
    if n == 0:
        return df

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    open_times = df["open_time"].to_numpy(dtype=np.int64)
    ema50_1h = df["ema50_1h"].to_numpy(dtype=float)
    bocpd_shocks = df["bocpd_shock_1h"].to_numpy(dtype=float)
    vpins = df["vpin_1h"].to_numpy(dtype=float)
    ret_15ms = df["ret_15m"].to_numpy(dtype=float)
    ret_1hs = df["ret_1h"].to_numpy(dtype=float)
    vol_ratios = df["vol_ratio_1h"].to_numpy(dtype=float)
    taker_buy_ratios = df["taker_buy_ratio_1h"].to_numpy(dtype=float)
    is_hour_close = df["is_hour_close"].to_numpy(dtype=bool)
    triggers = df["hourly_trigger"].to_numpy(dtype=bool)
    valid = df["valid_features"].to_numpy(dtype=bool)

    tail_guard = cfg.horizon_bars if cfg.horizon_bars <= 240 else min(cfg.horizon_bars, max(60, n // 20))
    max_start_idx = max(0, n - tail_guard)

    is_synthetic_fallback = False
    if n < FEATURE_WARMUP and not (triggers[:max_start_idx] & valid[:max_start_idx]).any():
        triggers = valid.copy()
        is_synthetic_fallback = True

    targets = np.full(n, np.nan, dtype=float)
    net_returns = np.full(n, np.nan, dtype=float)
    exit_times = np.full(n, -1, dtype=np.int64)

    fee_in = 1 + cfg.fee_rate
    fee_out = 1 - cfg.fee_rate
    slip_in = 1 + cfg.execution_bps / 10_000
    slip_out = 1 - cfg.execution_bps / 10_000

    for i in range(max_start_idx):
        if not triggers[i] or not valid[i]:
            continue
        entry_idx = i + 1
        if entry_idx >= n:
            continue
        if open_times[entry_idx] - open_times[i] != MINUTE_MS:
            continue

        if is_synthetic_fallback:
            if i + cfg.horizon_bars >= n or open_times[i + cfg.horizon_bars] - open_times[i] != cfg.horizon_bars * MINUTE_MS:
                continue

        entry_px = opens[entry_idx] * slip_in
        peak_px = entry_px
        is_calm_entry = (
            np.isfinite(vol_ratios[entry_idx])
            and vol_ratios[entry_idx] < 1.50
            and np.isfinite(taker_buy_ratios[entry_idx])
            and taker_buy_ratios[entry_idx] < 0.505
        )
        eff_trailing_activation = min(cfg.trailing_activation, 0.0058) if is_calm_entry else cfg.trailing_activation
        eff_trailing_delta = min(cfg.trailing_delta, 0.0012) if is_calm_entry else cfg.trailing_delta
        pnl = 0.0
        exit_idx = entry_idx

        max_k = min(entry_idx + cfg.horizon_bars, n)
        for k in range(entry_idx, max_k):
            if k > entry_idx and open_times[k] - open_times[k - 1] != MINUTE_MS:
                exit_px = opens[k] * slip_out
                pnl = (exit_px * fee_out) / (entry_px * fee_in) - 1
                exit_idx = k
                break

            b_h, b_l, b_o, b_c = highs[k], lows[k], opens[k], closes[k]
            if b_h > peak_px:
                peak_px = b_h

            if peak_px >= entry_px * (1 + eff_trailing_activation):
                stop_px = peak_px * (1 - eff_trailing_delta)
            else:
                stop_px = entry_px * (1 - cfg.initial_stop)

            if b_l <= stop_px:
                exit_px = min(b_o, stop_px) * slip_out
                pnl = (exit_px * fee_out) / (entry_px * fee_in) - 1
                exit_idx = k
                break
            elif (
                not is_synthetic_fallback
                and k > entry_idx
                and bocpd_shocks[k] >= cfg.bocpd_exit_threshold
                and (
                    (vpins[k] >= cfg.vpin_exit_threshold and np.isfinite(ret_15ms[k]) and ret_15ms[k] <= -0.008)
                    or (np.isfinite(ret_1hs[k]) and ret_1hs[k] <= -0.0050 and taker_buy_ratios[k] <= 0.45)
                )
            ):
                exit_px = b_c * slip_out
                pnl = (exit_px * fee_out) / (entry_px * fee_in) - 1
                exit_idx = k
                break
            elif not is_synthetic_fallback and is_hour_close[k] and (k > entry_idx) and np.isfinite(ema50_1h[k]) and (b_c < ema50_1h[k]):
                exit_px = b_c * slip_out
                pnl = (exit_px * fee_out) / (entry_px * fee_in) - 1
                exit_idx = k
                break
        else:
            exit_idx = max_k - 1
            exit_px = closes[exit_idx] * slip_out
            pnl = (exit_px * fee_out) / (entry_px * fee_in) - 1

        targets[i] = 1 if pnl > 0.0 else 0
        net_returns[i] = pnl
        exit_times[i] = open_times[exit_idx]

    df["target"] = targets
    df["net_return"] = net_returns
    df["exit_time"] = exit_times
    valid_samples = df["target"].notna()
    return df.loc[valid_samples].astype({"target": int, "exit_time": "int64"}).reset_index(drop=True)


def with_futures_directional_labels(
    candles: pd.DataFrame,
    futures_cfg: FuturesConfig,
    funding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Etiqueta SHORT/FLAT/LONG con retornos futuros netos y muestras purgables.

    Se evalúa una decisión al cierre del intervalo configurado usando la entrada de la
    vela 1m siguiente y un horizonte fijo. La etiqueta solo supera FLAT cuando la
    dirección paga fees, slippage y funding observado y rebasa el edge mínimo.
    """
    futures_cfg.validate()
    feature_cfg = StrategyConfig(symbol=futures_cfg.symbol)
    df = feature_frame(candles, feature_cfg)
    n = len(df)
    if n == 0:
        return df.assign(target=pd.Series(dtype="int64"), exit_time=pd.Series(dtype="int64"))

    open_times = df["open_time"].to_numpy(dtype=np.int64)
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    candidates = np.flatnonzero(interval_decision_mask(df, futures_cfg.decision_interval_minutes))
    funding_times = np.array([], dtype=np.int64)
    funding_rates = np.array([], dtype=float)
    funding_marks = np.array([], dtype=float)
    if funding is not None and len(funding):
        funding_times = pd.to_numeric(funding["fundingTime"], errors="raise").to_numpy(dtype=np.int64)
        funding_rates = pd.to_numeric(funding["fundingRate"], errors="raise").to_numpy(dtype=float)
        if "markPrice" in funding:
            funding_marks = pd.to_numeric(funding["markPrice"], errors="coerce").to_numpy(dtype=float)
        else:
            funding_marks = np.full(len(funding_times), np.nan, dtype=float)
        order = np.argsort(funding_times)
        funding_times, funding_rates, funding_marks = funding_times[order], funding_rates[order], funding_marks[order]

    records: list[dict] = []
    horizon = int(futures_cfg.holding_horizon_minutes)
    entry_slip = futures_cfg.execution_bps / 10_000.0
    min_edge = futures_cfg.min_net_edge_bps / 10_000.0
    for i in candidates:
        entry_idx = int(i) + 1
        exit_idx = entry_idx + horizon - 1
        if exit_idx >= n:
            continue
        if open_times[entry_idx] - open_times[i] != MINUTE_MS:
            continue
        if open_times[exit_idx] - open_times[entry_idx] != (horizon - 1) * MINUTE_MS:
            continue

        reference_entry = opens[entry_idx]
        raw_exit = closes[exit_idx]
        if not np.isfinite(reference_entry) or not np.isfinite(raw_exit) or reference_entry <= 0 or raw_exit <= 0:
            continue

        first_funding = int(np.searchsorted(funding_times, open_times[entry_idx], side="right"))
        after_exit = int(np.searchsorted(funding_times, open_times[exit_idx] + MINUTE_MS - 1, side="right"))
        funding_cost = 0.0
        if after_exit > first_funding:
            marks = funding_marks[first_funding:after_exit].copy()
            marks[~np.isfinite(marks) | (marks <= 0)] = raw_exit
            funding_cost = float(np.sum(funding_rates[first_funding:after_exit] * marks / reference_entry))

        long_entry = reference_entry * (1 + entry_slip)
        long_exit = raw_exit * (1 - entry_slip)
        long_qty = 1.0 / long_entry
        long_net = (
            long_qty * (long_exit - long_entry)
            - futures_cfg.fee_rate
            - long_qty * long_exit * futures_cfg.fee_rate
            - funding_cost
        )

        short_entry = reference_entry * (1 - entry_slip)
        short_exit = raw_exit * (1 + entry_slip)
        short_qty = 1.0 / short_entry
        short_net = (
            short_qty * (short_entry - short_exit)
            - futures_cfg.fee_rate
            - short_qty * short_exit * futures_cfg.fee_rate
            + funding_cost
        )

        if short_net >= min_edge and short_net > long_net:
            target = 0
        elif long_net >= min_edge and long_net > short_net:
            target = 2
        else:
            target = 1
        row = df.iloc[i].to_dict()
        row.update({
            "target": target,
            "exit_time": int(open_times[exit_idx]),
            "long_net_return": float(long_net),
            "short_net_return": float(short_net),
            "funding_cost_long": float(funding_cost),
        })
        records.append(row)
    if not records:
        empty = df.iloc[:0].copy()
        empty["target"] = pd.Series(dtype="int64")
        empty["exit_time"] = pd.Series(dtype="int64")
        return empty
    return pd.DataFrame.from_records(records).astype({"target": int, "exit_time": "int64"}).reset_index(drop=True)

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain.backtest import AdaptiveConformalThreshold
from ..domain.config import MINUTE_MS, StrategyConfig
from ..domain.features import FEATURES, feature_frame
from ..domain.time import utc_iso
from ..domain.paper_wallet import PaperWallet
from .modeling import learn_mature_outcome, predict_detailed
from .ports import BundleStore, MarketDataPort, MarketStreamPort, PaperStatePort


def run_replay_simulation(
    bundle: dict,
    candles: pd.DataFrame,
    repository: PaperStatePort,
    n_bars: int | None = None,
    allow_unvalidated_paper: bool = False,
    oos_only: bool = False,
) -> dict:
    """Paper trading replay over historical 1-minute Binance candles (v5.0)."""
    cfg = StrategyConfig(**bundle["config"])
    repository.clear_replay()
    wallet = PaperWallet(cfg, event_sink=repository.append_event)
    aci = AdaptiveConformalThreshold(float(bundle["threshold"]), gamma=cfg.aci_gamma, target_alpha=cfg.aci_target_alpha)
    allowed = bool(bundle["paper_gate"] or allow_unvalidated_paper)

    df = feature_frame(candles, cfg)

    if oos_only and "oos_start_open_time" in bundle:
        oos_start = int(bundle["oos_start_open_time"])
        sub = df[df["open_time"] >= oos_start].reset_index(drop=True)
        mode_label = "FUERA DE MUESTRA (OOS: VALIDACIÓN + TEST CIEGO)"
    elif n_bars is not None and n_bars > 0:
        sub = df.iloc[-n_bars:].reset_index(drop=True)
        mode_label = f"ÚLTIMAS {len(sub):,} VELAS"
    else:
        sub = df
        mode_label = "HISTORIAL COMPLETO (CON MODELO OOS)"

    print("=" * 70, flush=True)
    print(f"MODO: SIMULACIÓN PAPER TRADING v5.0 — {mode_label}", flush=True)
    print(f"Par: {cfg.symbol} | Capital Inicial: ${cfg.initial_cash:.2f} USDT", flush=True)
    print("Arquitectura: ensamble regularizado por régimen + VPIN/BOCPD + ACI", flush=True)
    paper_status = (
        "HABILITADAS (gate OOS aprobado)" if bundle["paper_gate"]
        else "HABILITADAS SOLO POR OVERRIDE EXPERIMENTAL (gate OOS NO aprobado)" if allow_unvalidated_paper
        else "DESHABILITADAS (gate OOS no aprobado)"
    )
    print(f"Señales virtuales: {paper_status}", flush=True)
    print("=" * 70, flush=True)
    print(f"Procesando {len(sub):,} velas de 1m desde {utc_iso(int(sub.iloc[0]['open_time']))} hasta {utc_iso(int(sub.iloc[-1]['open_time']))}...\n", flush=True)

    pred_records = []
    active_entry_features: np.ndarray | None = None
    pending_entry: dict | None = None
    prev_closed_trades = 0

    for _, row in sub.iterrows():
        vec = row[FEATURES].to_numpy(dtype=float)
        t = int(row["open_time"])
        h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
        # The backtest's execution_bps already represents total per-side spread/slippage.
        bid, ask = c, c
        is_hour_close = bool(row["is_hour_close"])
        is_trigger = bool(row["hourly_trigger"])
        ema20_1h = float(row["ema20_1h"]) if pd.notna(row["ema20_1h"]) else c
        ema50_1h = float(row["ema50_1h"]) if pd.notna(row["ema50_1h"]) else ema20_1h
        atr_pct_1h = float(row["atr_pct_1h"]) if pd.notna(row["atr_pct_1h"]) else None
        vpin_1h = float(row["vpin_1h"]) if pd.notna(row["vpin_1h"]) else 0.10
        bocpd_shock_1h = float(row["bocpd_shock_1h"]) if pd.notna(row["bocpd_shock_1h"]) else 0.0
        ret_15m = float(row["ret_15m"]) if pd.notna(row["ret_15m"]) else 0.0
        ret_1h = float(row["ret_1h"]) if pd.notna(row["ret_1h"]) else 0.0
        vol_ratio_1h = float(row["vol_ratio_1h"]) if pd.notna(row["vol_ratio_1h"]) else 1.0
        taker_buy_ratio_1h = float(row["taker_buy_ratio_1h"]) if pd.notna(row["taker_buy_ratio_1h"]) else 0.50
        eff_thr = aci.current_threshold(bocpd_shock_1h)

        p_champ, p_chall = 0.0, 0.0
        if is_trigger and np.isfinite(vec).all():
            det = predict_detailed(bundle, vec, use_oos_model=True)
            p_champ, p_chall = det["champion_p"], det["challenger_p"]
            pred_records.append({
                "at": utc_iso(t), "close": c,
                "champion_p": round(p_champ, 4), "challenger_p": round(p_chall, 4),
                "calm_agent_p": round(det["calm_agent_p"], 4),
                "volatile_agent_p": round(det["volatile_agent_p"], 4),
                "volatile_gate_weight": round(det["volatile_gate_weight"], 4),
                "vpin_1h": round(vpin_1h, 4),
                "bocpd_shock_1h": round(bocpd_shock_1h, 4),
                "threshold": round(eff_thr, 4),
            })

        signal_candidate = None
        if is_trigger and np.isfinite(vec).all() and p_champ >= eff_thr and wallet.s["qty"] == 0:
            signal_candidate = {
                "prob": p_champ,
                "challenger": p_chall,
                "threshold": eff_thr,
                "features": vec.copy(),
            }

        executing_pending = pending_entry is not None
        closed_before_bar = wallet.s["num_closed_trades"]
        trade_prob = pending_entry["prob"] if executing_pending else p_champ
        trade_challenger = pending_entry["challenger"] if executing_pending else p_chall
        trade_threshold = pending_entry["threshold"] if executing_pending else eff_thr
        note = wallet.on_bar(
            t, h, l, c, bid, ask, trade_prob, trade_challenger,
            trade_threshold, allowed, True,
            is_hour_start=is_hour_close, is_trigger=executing_pending,
            ema20_1h=ema20_1h, ema50_1h=ema50_1h, atr_pct_1h=atr_pct_1h,
            vpin_1h=vpin_1h, bocpd_shock_1h=bocpd_shock_1h, ret_15m=ret_15m,
            ret_1h=ret_1h, vol_ratio_1h=vol_ratio_1h, taker_buy_ratio_1h=taker_buy_ratio_1h,
            entry_ask=float(row["open"]) if executing_pending else None,
            process_entry_bar_ohlc=executing_pending,
            open_price=float(row["open"]),
        )
        if note == "compra virtual" or (
            executing_pending and wallet.s["num_closed_trades"] > closed_before_bar
        ):
            active_entry_features = pending_entry["features"] if pending_entry is not None else vec.copy()
        if wallet.s["num_closed_trades"] > prev_closed_trades and active_entry_features is not None:
            last_trade_won = wallet.s.get("last_close_ret_pct", 0.0) > 0.0
            learn_mature_outcome(bundle, active_entry_features, int(last_trade_won))
            aci.update(last_trade_won, wallet.s.get("last_close_ret_pct", 0.0))
            active_entry_features = None
            prev_closed_trades = wallet.s["num_closed_trades"]

        pending_entry = signal_candidate if wallet.s["qty"] == 0 else None

    repository.save_state(wallet.s)
    repository.save_checkpoint(
        {"online": bundle["online"], "base_training_utc": bundle["last_training_utc"]}
    )
    repository.write_predictions(pred_records)

    last_close = float(sub.iloc[-1]["close"])
    final_equity = wallet.equity(last_close)
    pnl = final_equity - cfg.initial_cash
    ret_pct = (pnl / cfg.initial_cash) * 100

    print("=" * 70, flush=True)
    print("RESUMEN FINAL DE SIMULACIÓN PAPER TRADING v5.0:", flush=True)
    print(f"Saldo Cash Disponible: ${wallet.s['cash']:.2f} USDT", flush=True)
    print(f"Posición Abierta:      {wallet.s['qty']:.6f} {cfg.symbol[:3]}", flush=True)
    print(f"Equity Total Cartera:  ${final_equity:.2f} USDT", flush=True)
    print(f"PnL Neto Realizado:    {wallet.s['realized_pnl']:+.2f} USDT", flush=True)
    print(f"Retorno Neto Total:    {ret_pct:+.2f}%", flush=True)
    print(f"Operaciones Cerradas:  {wallet.s['num_closed_trades']}", flush=True)
    print(f"Umbral Final ACI:      {aci.current_threshold():.4f}", flush=True)
    print("Estado paper y predicciones guardados por el repositorio configurado.", flush=True)
    print("=" * 70 + "\n", flush=True)

    return {
        "initial_cash": cfg.initial_cash,
        "final_equity": final_equity,
        "realized_pnl": wallet.s["realized_pnl"],
        "return_pct": ret_pct,
        "closed_trades": wallet.s["num_closed_trades"],
        "open_qty": wallet.s["qty"],
        "final_aci_threshold": aci.current_threshold(),
    }


async def run_live(
    bundle_store: BundleStore,
    repository: PaperStatePort,
    market_data: MarketDataPort,
    market_stream: MarketStreamPort,
    fallback_csv: Path,
    allow_unvalidated_paper: bool = False,
    max_duration_sec: int | None = None,
) -> None:
    bundle = bundle_store.load()
    bundle_mtime = bundle_store.modified_at()
    cfg = StrategyConfig(**bundle["config"])
    wallet = PaperWallet(cfg, state=repository.load_state(), event_sink=repository.append_event)
    aci = AdaptiveConformalThreshold(float(bundle["threshold"]), gamma=cfg.aci_gamma, target_alpha=cfg.aci_target_alpha)
    checkpoint = repository.load_checkpoint()
    if checkpoint and checkpoint.get("base_training_utc") == bundle["last_training_utc"]:
        bundle["online"] = checkpoint["online"]
    allowed = bool(bundle["paper_gate"] or allow_unvalidated_paper)

    started_utc = datetime.now(timezone.utc).isoformat()
    print("=" * 70, flush=True)
    print("MODO: PAPER TRADING EN VIVO v5.0 (SIN CLAVES API)", flush=True)
    print(f"Inicio UTC: {started_utc} | Duración objetivo: {max_duration_sec or 'continua'}s", flush=True)
    print(f"Par: {cfg.symbol} | Capital actual: ${wallet.s['cash']:.2f} USDT", flush=True)
    print(f"Estrategia: ensamble regularizado + VPIN/BOCPD + Trailing ({cfg.trailing_activation*100:.1f}% -> {cfg.trailing_delta*100:.1f}%)", flush=True)
    paper_status = (
        "HABILITADAS (gate OOS aprobado)" if bundle["paper_gate"]
        else "HABILITADAS SOLO POR OVERRIDE EXPERIMENTAL (gate OOS NO aprobado)" if allow_unvalidated_paper
        else "DESHABILITADAS (gate OOS no aprobado)"
    )
    print(f"Señales virtuales: {paper_status}", flush=True)
    print("=" * 70, flush=True)

    try:
        df_local = market_data.read_recent_local(fallback_csv, 6500)
    except FileNotFoundError:
        df_local = pd.DataFrame()
    try:
        recent_rest = await asyncio.to_thread(market_data.recent_closed_candles, cfg.symbol, 1500 if len(df_local) > 0 else 6500)
        if len(df_local) > 0:
            common_cols = [c for c in df_local.columns if c in recent_rest.columns]
            historical = (
                pd.concat([df_local[common_cols], recent_rest[common_cols]], ignore_index=True)
                .sort_values("open_time")
                .drop_duplicates("open_time", keep="last")
                .iloc[-6500:]
                .reset_index(drop=True)
            )
            print(f"Historial sincronizado (CSV + Binance REST): {len(historical):,} velas hasta {utc_iso(int(historical.iloc[-1]['open_time']))}.", flush=True)
        else:
            historical = recent_rest
            print(f"Historial descargado de Binance REST: {len(historical):,} velas hasta {utc_iso(int(historical.iloc[-1]['open_time']))}.", flush=True)
    except Exception as exc:
        if len(df_local) > 0:
            historical = df_local.reset_index(drop=True)
            print(f"AVISO REST ({exc}); usando CSV local ({len(historical):,} velas).", flush=True)
        else:
            raise RuntimeError(f"No hay archivo CSV local ni conexión a Binance REST: {exc}")

    candles = deque(historical.to_dict("records"), maxlen=7500)
    last_ts = int(historical.iloc[-1]["open_time"])
    last_close = float(historical.iloc[-1]["close"])
    quote = {"bid": last_close, "ask": last_close, "received": time.monotonic()}
    start_time = time.monotonic()
    active_entry_features: np.ndarray | None = None
    prev_closed_trades = wallet.s["num_closed_trades"]
    bars_processed = 0

    df_init = feature_frame(pd.DataFrame(list(candles)), cfg)
    init_row = df_init.iloc[-1]
    init_vec = init_row[FEATURES].to_numpy(dtype=float)
    init_det = predict_detailed(bundle, init_vec, use_oos_model=False) if np.isfinite(init_vec).all() else {
        "champion_p": 0.0, "challenger_p": 0.0, "calm_agent_p": 0.0, "volatile_agent_p": 0.0, "volatile_gate_weight": 0.5
    }
    init_bocpd = float(init_row.get("bocpd_shock_1h", 0.0))
    init_vpin = float(init_row.get("vpin_1h", 0.10))
    repository.save_status({
        "status": "running",
        "version": "6.0",
        "started_utc": started_utc,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": 0,
        "target_duration_sec": max_duration_sec,
        "bars_processed": 0,
        "last_bar_utc": utc_iso(last_ts),
        "symbol": cfg.symbol,
        "price": last_close,
        "bid": quote["bid"],
        "ask": quote["ask"],
        "champion_prob": round(init_det["champion_p"], 4),
        "challenger_prob": round(init_det["challenger_p"], 4),
        "calm_agent_prob": round(init_det["calm_agent_p"], 4),
        "volatile_agent_prob": round(init_det["volatile_agent_p"], 4),
        "volatile_gate_weight": round(init_det["volatile_gate_weight"], 4),
        "base_threshold": bundle["threshold"],
        "threshold": round(aci.current_threshold(init_bocpd), 4),
        "indicators": {
            "ema20_1h": round(float(init_row.get("ema20_1h", last_close)), 2),
            "ema50_1h": round(float(init_row.get("ema50_1h", last_close)), 2),
            "macro_spread_bps": round(float(init_row.get("macro_spread", 0.0)) * 10_000, 2),
            "rsi_14_1h": round(float(init_row.get("rsi_14_1h", 50.0)), 2),
            "atr_pct_1h": round(float(init_row.get("atr_pct_1h", 0.0)) * 100, 3),
            "vpin_1h": round(init_vpin, 4),
            "bocpd_shock_1h": round(init_bocpd, 4),
        },
        "wallet": wallet.s,
        "equity_usdt": round(wallet.equity(last_close), 4),
    })

    reconnect_delay = 2
    while True:
        elapsed = time.monotonic() - start_time
        if max_duration_sec and (elapsed >= max_duration_sec):
            print(f"\nDuración de simulación en vivo completada ({max_duration_sec}s / {max_duration_sec/3600:.2f}h). Deteniendo.", flush=True)
            break
        try:
            print("Conectando al stream paper configurado; escuchando cotizaciones y velas cerradas...", flush=True)
            quote["received"] = time.monotonic()
            reconnect_delay = 2
            async for event in market_stream.events(cfg.symbol):
                elapsed = time.monotonic() - start_time
                if max_duration_sec and (elapsed >= max_duration_sec):
                    break
                data = event.get("data", {})
                if event.get("stream", "").endswith("@bookTicker"):
                    bid, ask = float(data.get("b", 0)), float(data.get("a", 0))
                    if bid > 0 and ask >= bid:
                        quote.update(bid=bid, ask=ask, received=time.monotonic())
                    continue

                if data.get("e") != "kline" or not data["k"].get("x"):
                    continue

                k = data["k"]
                t = int(k["t"])
                if t <= last_ts:
                    continue
                if t - last_ts != MINUTE_MS:
                    print(f"AVISO: hueco detectado ({utc_iso(last_ts)} -> {utc_iso(t)}); sincronizando vía REST...", flush=True)
                    refreshed = await asyncio.to_thread(market_data.recent_closed_candles, cfg.symbol, 1500)
                    common_cols = [c for c in refreshed.columns if c in ("open_time", "open", "high", "low", "close", "volume", "taker_buy_volume")]
                    merged = (
                        pd.concat([pd.DataFrame(list(candles))[common_cols], refreshed[common_cols]], ignore_index=True)
                        .sort_values("open_time")
                        .drop_duplicates("open_time", keep="last")
                        .iloc[-6500:]
                    )
                    candles = deque(merged.to_dict("records"), maxlen=7500)
                    last_ts = int(merged.iloc[-1]["open_time"])

                last_ts = t
                bar = {
                    "open_time": t, "open": float(k["o"]), "high": float(k["h"]),
                    "low": float(k["l"]), "close": float(k["c"]),
                    "volume": float(k["v"]), "taker_buy_volume": float(k["V"]),
                }
                candles.append(bar)
                bars_processed += 1
                if bundle_store.modified_at() > bundle_mtime:
                    try:
                        bundle = bundle_store.load()
                        bundle_mtime = bundle_store.modified_at()
                        cfg = StrategyConfig(**bundle["config"])
                        wallet.cfg = cfg
                        aci.base_threshold = float(bundle["threshold"])
                        print(f"[HOT-RELOAD] Nuevo bundle auto-optimizado cargado en caliente (thr={aci.base_threshold:.3f}, lookback={cfg.breakout_lookback_hours}h)", flush=True)
                    except Exception as hot_err:
                        print(f"[HOT-RELOAD AVISO] {hot_err}", flush=True)

                df = feature_frame(pd.DataFrame(list(candles)), cfg)
                last_row = df.iloc[-1]
                vector = last_row[FEATURES].to_numpy(dtype=float)

                if not bool(last_row.get("valid_features", False)) or not np.isfinite(vector).all():
                    print(f"{utc_iso(t)} | Calentando indicadores...", flush=True)
                    continue

                det = predict_detailed(bundle, vector, use_oos_model=False)
                p_champion, p_online = det["champion_p"], det["challenger_p"]
                fresh = (time.monotonic() - quote["received"]) <= 10

                is_hour_close = bool(last_row.get("is_hour_close", False))
                regime_code = int(last_row.get("regime_code", 0))
                ret_15m = float(last_row.get("ret_15m", 0.0))
                ret_1h = float(last_row.get("ret_1h", 0.0))
                vol_ratio_1h = float(last_row.get("vol_ratio_1h", 1.0))
                taker_buy_ratio_1h = float(last_row.get("taker_buy_ratio_1h", 0.50))
                bocpd_shock_1h = float(last_row.get("bocpd_shock_1h", 0.0))
                # Use the same validated trigger as feature_frame/backtest; the extra
                # minute-by-minute continuation path was not part of OOS evaluation.
                is_trigger = bool(last_row.get("hourly_trigger", False))
                ema20_1h = float(last_row.get("ema20_1h", bar["close"]))
                ema50_1h = float(last_row.get("ema50_1h", ema20_1h))
                atr_pct_1h = float(last_row.get("atr_pct_1h", np.nan))
                vpin_1h = float(last_row.get("vpin_1h", 0.10))
                macro_bps = float(last_row.get("macro_spread", 0.0)) * 10_000
                rsi_1h = float(last_row.get("rsi_14_1h", 50.0))
                eff_thr = aci.current_threshold(bocpd_shock_1h)

                note = wallet.on_bar(
                    t, bar["high"], bar["low"], bar["close"],
                    quote["bid"], quote["ask"], p_champion, p_online,
                    eff_thr, allowed, fresh,
                    is_hour_start=is_hour_close, is_trigger=is_trigger,
                    ema20_1h=ema20_1h, ema50_1h=ema50_1h, atr_pct_1h=atr_pct_1h,
                    vpin_1h=vpin_1h, bocpd_shock_1h=bocpd_shock_1h, ret_15m=ret_15m,
                    ret_1h=ret_1h, vol_ratio_1h=vol_ratio_1h, taker_buy_ratio_1h=taker_buy_ratio_1h,
                    open_price=float(bar["open"]),
                )
                repository.save_state(wallet.s)
                if note == "compra virtual":
                    entry_block_reason = "entrada ejecutada"
                elif wallet.s["qty"] > 0:
                    entry_block_reason = "posición ya abierta"
                elif not allowed:
                    entry_block_reason = "paper gate deshabilitado"
                elif not fresh:
                    entry_block_reason = "cotización desactualizada"
                elif wallet.s["halt_today"]:
                    entry_block_reason = "límite diario de pérdida"
                elif wallet.s["frozen"]:
                    entry_block_reason = "bloqueo por drawdown máximo"
                elif t < wallet.s["cooldown_ts"]:
                    entry_block_reason = "periodo de espera tras cierre"
                elif not is_trigger:
                    entry_block_reason = "sin trigger de régimen/momentum"
                elif p_champion < eff_thr:
                    entry_block_reason = "probabilidad bajo el umbral"
                elif quote["bid"] <= 0 or quote["ask"] < quote["bid"]:
                    entry_block_reason = "cotización bid/ask inválida"
                elif (quote["ask"] - quote["bid"]) / quote["bid"] > 0.0015:
                    entry_block_reason = "spread superior al máximo"
                else:
                    entry_block_reason = "entrada elegible; revisar tamaño mínimo"
                if note == "compra virtual":
                    active_entry_features = vector.copy()
                elif wallet.s["num_closed_trades"] > prev_closed_trades and active_entry_features is not None:
                    last_trade_won = wallet.s.get("last_close_ret_pct", 0.0) > 0.0
                    learn_mature_outcome(bundle, active_entry_features, int(last_trade_won))
                    aci.update(last_trade_won, wallet.s.get("last_close_ret_pct", 0.0))
                    repository.save_checkpoint(
                        {"online": bundle["online"], "base_training_utc": bundle["last_training_utc"]},
                    )
                    active_entry_features = None
                    prev_closed_trades = wallet.s["num_closed_trades"]

                pos_str = f"LONG (${wallet.s['entry_fill']:.1f})" if wallet.s["qty"] > 0 else "CASH"
                cur_eq = wallet.equity(quote["bid"] if quote["bid"] > 0 else bar["close"])
                print(
                    f"[{utc_iso(t)[11:19]}] BTC: ${bar['close']:,.2f} | Pos: {pos_str:10s} | Eq: ${cur_eq:,.2f} "
                    f"| P(MacroHFT): {p_champion:.3f} (ACI:{eff_thr:.3f}) | VPIN:{vpin_1h:.3f} | BOCPD:{bocpd_shock_1h:.3f} | {note}",
                    flush=True,
                )

                repository.append_prediction({
                    "at": utc_iso(t), "close": bar["close"],
                    "bid": quote["bid"], "ask": quote["ask"],
                    "champion_p": round(p_champion, 4),
                    "challenger_p": round(p_online, 4),
                    "calm_agent_p": round(det["calm_agent_p"], 4),
                    "volatile_agent_p": round(det["volatile_agent_p"], 4),
                    "volatile_gate_weight": round(det["volatile_gate_weight"], 4),
                    "threshold": round(eff_thr, 4),
                    "is_hour_close": is_hour_close,
                    "is_trigger": is_trigger,
                    "macro_spread_bps": round(macro_bps, 2),
                    "rsi_14_1h": round(rsi_1h, 2),
                    "vpin_1h": round(vpin_1h, 4),
                    "bocpd_shock_1h": round(bocpd_shock_1h, 4),
                    "action": note, "position": pos_str,
                    "entry_block_reason": entry_block_reason,
                    "equity_usdt": round(cur_eq, 4),
                })
                repository.save_status({
                    "status": "running",
                    "version": "6.0",
                    "started_utc": started_utc,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.monotonic() - start_time, 1),
                    "target_duration_sec": max_duration_sec,
                    "bars_processed": bars_processed,
                    "last_bar_utc": utc_iso(t),
                    "symbol": cfg.symbol,
                    "price": bar["close"],
                    "bid": quote["bid"],
                    "ask": quote["ask"],
                    "champion_prob": round(p_champion, 4),
                    "challenger_prob": round(p_online, 4),
                    "calm_agent_prob": round(det["calm_agent_p"], 4),
                    "volatile_agent_prob": round(det["volatile_agent_p"], 4),
                    "volatile_gate_weight": round(det["volatile_gate_weight"], 4),
                    "base_threshold": bundle["threshold"],
                    "threshold": round(eff_thr, 4),
                    "is_hour_close": is_hour_close,
                    "is_trigger": is_trigger,
                    "indicators": {
                        "ema20_1h": round(ema20_1h, 2),
                        "ema50_1h": round(ema50_1h, 2),
                        "macro_spread_bps": round(macro_bps, 2),
                        "rsi_14_1h": round(rsi_1h, 2),
                        "atr_pct_1h": round(atr_pct_1h * 100, 3) if np.isfinite(atr_pct_1h) else None,
                        "vpin_1h": round(vpin_1h, 4),
                        "bocpd_shock_1h": round(bocpd_shock_1h, 4),
                    },
                    "wallet": wallet.s,
                    "equity_usdt": round(cur_eq, 4),
                    "last_action": note,
                    "entry_block_reason": entry_block_reason,
                })
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as exc:
            print(f"Conexión Binance interrumpida ({type(exc).__name__}): {exc}. Reconectando en {reconnect_delay}s...", flush=True)
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)

    final_px = quote["bid"] if quote["bid"] > 0 else float(candles[-1]["close"])
    final_eq = wallet.equity(final_px)
    summary = {
        "status": "completed",
        "version": "6.0",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.monotonic() - start_time, 1),
        "bars_processed": bars_processed,
        "final_price": final_px,
        "final_equity_usdt": round(final_eq, 4),
        "realized_pnl_usdt": round(wallet.s["realized_pnl"], 4),
        "closed_trades": wallet.s["num_closed_trades"],
        "open_qty": wallet.s["qty"],
        "final_aci_threshold": round(aci.current_threshold(), 4),
    }
    repository.save_status(summary)
    repository.save_summary(summary)
    print("=" * 70, flush=True)
    print("RESUMEN FINAL DE SESIÓN PAPER-LIVE v5.0:", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("=" * 70, flush=True)

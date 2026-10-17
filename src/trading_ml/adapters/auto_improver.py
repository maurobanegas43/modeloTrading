from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain.backtest import compute_chow_utility_score, simulate
from ..domain.config import StrategyConfig
from ..domain.features import FEATURES, feature_frame, with_labels
from ..domain.time import utc_iso
from .dashboard_sync import write_live_js
from .market_data import read_history, recent_closed_candles
from .model_store import FileBundleStore, JsonReportStore
from ..application.modeling import _new_champion, _predict_frame_probs, train_model


def load_bundle(path: Path) -> dict:
    return FileBundleStore(path).load()


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")



def _analyze_live_counterfactuals(live_dir: Path, fee_rate: float = 0.001, execution_bps: float = 3.0) -> dict:
    """Analyze live predictions across the 3 Quantitative Regimes (A: Expansion, B: Tactical, C: Toxic Shield)."""
    preds_path = live_dir / "predictions.jsonl"
    if not preds_path.is_file():
        return {
            "live_bars_analyzed": 0,
            "high_prob_bars": 0,
            "market_change_pct": 0.0,
            "counterfactual_unfiltered_return_pct": 0.0,
            "regime_a_pct": 0.0,
            "regime_b_pct": 0.0,
            "regime_c_toxic_pct": 0.0,
            "chow_live_abstention_penalty": 0.0,
            "verdict": "Sin datos en vivo suficientes",
        }
    rows = []
    for line in preds_path.read_text(encoding="utf-8").splitlines()[-240:]:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if len(rows) < 5:
        return {
            "live_bars_analyzed": len(rows),
            "high_prob_bars": 0,
            "market_change_pct": 0.0,
            "counterfactual_unfiltered_return_pct": 0.0,
            "regime_a_pct": 0.0,
            "regime_b_pct": 0.0,
            "regime_c_toxic_pct": 0.0,
            "chow_live_abstention_penalty": 0.0,
            "verdict": "Calentando ventana en vivo",
        }

    closes = np.array([float(r.get("close", 0.0)) for r in rows], dtype=float)
    probs = np.array([float(r.get("champion_p", 0.0)) for r in rows], dtype=float)
    thrs = np.array([float(r.get("threshold", 0.42)) for r in rows], dtype=float)
    macro_bps = np.array([float(r.get("macro_spread_bps", 0.0)) for r in rows], dtype=float)
    vpins = np.array([float(r.get("vpin_1h", 0.15)) for r in rows], dtype=float)
    bocpds = np.array([float(r.get("bocpd_shock_1h", 0.0)) for r in rows], dtype=float)
    rsis = np.array([float(r.get("rsi_14_1h", 50.0)) for r in rows], dtype=float)

    mkt_ret_pct = round((closes[-1] / closes[0] - 1.0) * 100.0, 3) if closes[0] > 0 else 0.0

    # Classify live bars into Regime A (Expansion), Regime B (Tactical), Regime C (Toxic/Freefall)
    toxic_mask = (bocpds >= 0.45) | (vpins >= 0.60) | (rsis <= 32.0) | (macro_bps < 0.0)
    reg_a_mask = (~toxic_mask) & (macro_bps >= 5.0)
    reg_b_mask = (~toxic_mask) & (~reg_a_mask) & (macro_bps >= 2.0)

    reg_a_pct = round(float(reg_a_mask.mean() * 100.0), 1)
    reg_b_pct = round(float(reg_b_mask.mean() * 100.0), 1)
    reg_c_pct = round(float(toxic_mask.mean() * 100.0), 1)

    high_mask = probs >= thrs
    high_indices = np.where(high_mask)[0]
    round_trip_cost = 2.0 * (fee_rate + execution_bps / 10_000.0)
    if len(high_indices) > 0:
        first_idx = int(high_indices[0])
        entry_px = closes[first_idx]
        exit_px = closes[-1]
        cf_ret_pct = round(((exit_px / entry_px - 1.0) - round_trip_cost) * 100.0, 3)
    else:
        cf_ret_pct = 0.0

    # Chow Abstention Rule in Live:
    # Penalize zero trades ONLY if the market spent >= 25% of time in Regime A/B with positive upside
    tradable_pct = reg_a_pct + reg_b_pct
    if tradable_pct >= 25.0 and cf_ret_pct > 0.25:
        chow_live_pen = round(0.80 * (tradable_pct / 100.0), 3)
        verdict = (
            f"Alerta Chow de sobre-precaución ({tradable_pct:.0f}% tiempo en Régimen A/B, +{cf_ret_pct:.2f}% potencial); "
            f"ajustando sensibilidad a Régimen B Táctico"
        )
    elif cf_ret_pct < -0.15:
        chow_live_pen = 0.0
        verdict = (
            f"Régimen C Escudo Anticrisis activo ({reg_c_pct:.0f}% tiempo tóxico/bajista): "
            f"evitó pérdida de {cf_ret_pct:+.2f}% en caída ({mkt_ret_pct:+.2f}%) sin penalización Chow"
        )
    else:
        chow_live_pen = 0.0
        verdict = (
            f"Distribución 3 Regímenes (A:{reg_a_pct:.0f}% | B:{reg_b_pct:.0f}% | C:{reg_c_pct:.0f}%) · "
            f"Protección en CASH óptima ({mkt_ret_pct:+.2f}%)"
        )

    return {
        "live_bars_analyzed": len(rows),
        "high_prob_bars": int(high_mask.sum()),
        "market_change_pct": mkt_ret_pct,
        "counterfactual_unfiltered_return_pct": cf_ret_pct,
        "regime_a_pct": reg_a_pct,
        "regime_b_pct": reg_b_pct,
        "regime_c_toxic_pct": reg_c_pct,
        "chow_live_abstention_penalty": chow_live_pen,
        "verdict": verdict,
    }


def run_autonomous_improvement_cycle(root: Path, sync_rest: bool = True) -> dict:
    """Execute one autonomous 30-min cycle using Chow Abstention-Penalized Utility + Omega + 3-Regime Kelly."""
    root = root.resolve()
    csv_path = root / "data" / "BTCUSDT_1m.csv"
    model_path = root / "models" / "bundle.joblib"
    report_path = root / "reports" / "evaluation.json"
    live_dir = root / "paper_live_4h"
    log_path = live_dir / "auto_improvements.jsonl"

    bundle = load_bundle(model_path)
    current_cfg = StrategyConfig(**bundle["config"])

    # 1. Refresh dataset with latest closed 1m candles from Binance REST if available
    new_bars_added = 0
    df = read_history(csv_path)
    if sync_rest:
        try:
            recent = recent_closed_candles(current_cfg.symbol, 360)
            if len(recent) > 0:
                before_len = len(df)
                common_cols = [c for c in df.columns if c in recent.columns]
                df = (
                    pd.concat([df[common_cols], recent[common_cols]], ignore_index=True)
                    .sort_values("open_time")
                    .drop_duplicates("open_time", keep="last")
                    .reset_index(drop=True)
                )
                new_bars_added = max(0, len(df) - before_len)
                if new_bars_added > 0:
                    df.to_csv(csv_path, index=False)
        except Exception:
            pass

    # 2. Analyze recent live telemetry & 3-Regime counterfactuals
    cf_analysis = _analyze_live_counterfactuals(live_dir, current_cfg.fee_rate, current_cfg.execution_bps)

    # 3. Evaluate Multi-Regime (A+B+C) Candidate Grid via Chow Abstention-Penalized Utility + Omega + Thompson Posterior
    base_dict = asdict(current_cfg)
    candidate_configs = [
        current_cfg,
        StrategyConfig(**{**base_dict, "breakout_lookback_hours": 6, "min_macro_spread": 0.0004, "threshold": 0.36, "allocation": 0.50, "enable_tactical_regime_b": True}),
        StrategyConfig(**{**base_dict, "breakout_lookback_hours": 8, "min_macro_spread": 0.0005, "threshold": 0.38, "allocation": 0.50, "enable_tactical_regime_b": True}),
        StrategyConfig(**{**base_dict, "breakout_lookback_hours": 8, "min_macro_spread": 0.0005, "threshold": 0.40, "allocation": 0.50, "enable_tactical_regime_b": True}),
        StrategyConfig(**{**base_dict, "breakout_lookback_hours": 10, "min_macro_spread": 0.0005, "threshold": 0.40, "allocation": 0.50, "enable_tactical_regime_b": True}),
    ]

    best_score = -1e9
    best_cfg = current_cfg
    best_metrics = {}
    baseline_score = -1e9

    # Select configurations on validation only. The final test period is held back
    # for a single evaluation of the selected candidate, never for ranking candidates.
    trained_cache = {}

    for idx, cand_cfg in enumerate(candidate_configs):
        cache_key = (
            cand_cfg.breakout_lookback_hours,
            cand_cfg.min_macro_spread,
            getattr(cand_cfg, "enable_tactical_regime_b", True),
        )
        if cache_key not in trained_cache:
            full_frame = feature_frame(df, cand_cfg)
            labeled = with_labels(df, cand_cfg)
            n = len(full_frame)
            a = int(n * 0.70)
            b = int(n * 0.85)
            t_a = int(full_frame["open_time"].iloc[a])
            train_labeled = labeled[labeled["exit_time"] < t_a]
            if len(train_labeled) < 6 or train_labeled["target"].nunique() < 2:
                trained_cache[cache_key] = None
            else:
                model, _ = _new_champion()
                model.fit(train_labeled[FEATURES].to_numpy(dtype=float), train_labeled["target"].to_numpy(dtype=int))
                val_split = full_frame.iloc[a:b].copy()
                trained_cache[cache_key] = (
                    val_split, _predict_frame_probs(model, val_split),
                )

        cached = trained_cache[cache_key]
        if cached is None:
            continue
        val_split, val_probs = cached

        val_bt = simulate(val_split, val_probs, cand_cfg.threshold, cand_cfg)
        val_score = compute_chow_utility_score(
            val_bt,
            target_trades=getattr(cand_cfg, "target_oos_trades", 6),
            chow_weight=getattr(cand_cfg, "chow_penalty_weight", 0.80),
            require_gate=False,
        )
        score = val_score["chow_utility_score"]
        if idx == 0:
            baseline_score = score
        if score > best_score:
            best_score = score
            best_cfg = cand_cfg
            best_metrics = {
                "validation_return_pct": val_bt["total_return_pct"],
                "validation_trades": val_bt["trades"],
                "validation_win_rate": val_bt["win_rate"],
                "validation_profit_factor": val_bt["profit_factor"],
                "validation_score": score,
                "chow_utility_score": score,
            }

    promoted = False
    if best_score > baseline_score + 0.03:
        candidate_model = model_path.with_name(".auto_candidate_bundle.joblib")
        candidate_report = report_path.with_name(".auto_candidate_evaluation.json")
        candidate_bundle, candidate_result = train_model(df, best_cfg)
        FileBundleStore(candidate_model).save(candidate_bundle)
        JsonReportStore(candidate_report).save(candidate_result)
        best_metrics.update({
            "oos_return_pct": candidate_result["out_of_sample_combined_trading"]["total_return_pct"],
            "oos_trades": candidate_result["out_of_sample_combined_trading"]["trades"],
            "oos_win_rate": candidate_result["out_of_sample_combined_trading"]["win_rate"],
            "oos_profit_factor": candidate_result["out_of_sample_combined_trading"]["profit_factor"],
            "test_return_pct": candidate_result["test_trading"]["total_return_pct"],
            "walk_forward_return_pct": candidate_result["walk_forward_summary"]["oos_compounded_return_pct"],
            "paper_gate_checks": candidate_result["paper_gate_checks"],
        })
        promoted = bool(candidate_result["approved_for_default_paper_signals"])
        if promoted:
            os.replace(candidate_model, model_path)
            os.replace(candidate_report, report_path)
        else:
            candidate_model.unlink(missing_ok=True)
            candidate_report.unlink(missing_ok=True)

    existing_lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.is_file() else []
    cycle_idx = len(existing_lines) + 1

    action_text = (
        f"Promovido tras selección en validación y gate ciego aprobado "
        f"(Lookback {best_cfg.breakout_lookback_hours}h, Macro>={best_cfg.min_macro_spread*10000:.1f}bps, "
        f"Thr={best_cfg.threshold:.2f}); OOS {best_metrics.get('oos_return_pct', 0):+.3f}% "
        f"({best_metrics.get('oos_trades', 0)} trades, WF {best_metrics.get('walk_forward_return_pct', 0):+.3f}%)"
        if promoted
        else f"Sin promoción. Mejor configuración en validación: {best_metrics.get('validation_return_pct', 0):+.3f}% "
             f"({best_metrics.get('validation_trades', 0)} trades, score {best_metrics.get('validation_score', 0):+.2f}); "
             f"cualquier finalista evaluado fuera de muestra debe pasar el gate completo. {cf_analysis['verdict']}"
    )

    cycle_record = {
        "cycle": cycle_idx,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "new_bars_synced": new_bars_added,
        "total_dataset_bars": len(df),
        "live_counterfactual": cf_analysis,
        "live_diagnostics": {
            "bars_analyzed": cf_analysis["live_bars_analyzed"],
            "btc_change_pct": cf_analysis["market_change_pct"],
            "regime_filter_saved_pct": -cf_analysis["counterfactual_unfiltered_return_pct"],
            "regime_a_pct": cf_analysis["regime_a_pct"],
            "regime_b_pct": cf_analysis["regime_b_pct"],
            "regime_c_toxic_pct": cf_analysis["regime_c_toxic_pct"],
        },
        "candidates_evaluated": len(candidate_configs),
        "promoted_new_config": promoted,
        "upgraded": promoted,
        "active_config": {
            "breakout_lookback_hours": best_cfg.breakout_lookback_hours,
            "lookback_hours": best_cfg.breakout_lookback_hours,
            "min_macro_spread": best_cfg.min_macro_spread,
            "min_macro_spread_bps": round(best_cfg.min_macro_spread * 10_000, 1),
            "threshold": best_cfg.threshold,
            "allocation": best_cfg.allocation,
            "enable_tactical_regime_b": getattr(best_cfg, "enable_tactical_regime_b", True),
            "trailing_activation_pct": round(best_cfg.trailing_activation * 100, 2),
            "trailing_delta_pct": round(best_cfg.trailing_delta * 100, 2),
        },
        "oos_metrics": best_metrics,
        "current_oos_score": {
            "return_pct": best_metrics.get("oos_return_pct", 0.0),
            "profit_factor": best_metrics.get("oos_profit_factor", 0.0),
            "chow_utility_score": best_metrics.get("chow_utility_score", 0.0),
            "trades": best_metrics.get("oos_trades", 0),
        },
        "decision_reason": cf_analysis["verdict"],
        "action_summary": action_text,
    }
    _append_jsonl(log_path, cycle_record)
    write_live_js(root)
    return cycle_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor autónomo v5.0; selecciona en validación y promueve solo tras gate OOS")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--watch", action="store_true", help="Ejecutar ciclo autónomo cada interval-sec segundos")
    parser.add_argument("--interval-sec", type=int, default=1800, help="Intervalo en segundos (default: 1800 = 30 min)")
    parser.add_argument("--duration-sec", type=int, default=86400, help="Duración total del demonio autónomo")
    parser.add_argument("--no-rest", action="store_true", help="Omitir sincronización REST")
    args = parser.parse_args()

    start = time.monotonic()
    cycle_num = 0
    while True:
        cycle_num += 1
        rec = run_autonomous_improvement_cycle(args.root, sync_rest=not args.no_rest)
        print(
            f"[AUTO-IMPROVER v5.0 Ciclo #{rec['cycle']} | {rec['timestamp_utc']}] "
            f"+{rec['new_bars_synced']} velas | {rec['action_summary']}",
            flush=True,
        )
        if not args.watch or (time.monotonic() - start >= args.duration_sec):
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()

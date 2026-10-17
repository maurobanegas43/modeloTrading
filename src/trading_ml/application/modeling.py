from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..domain.backtest import simulate
from ..domain.config import StrategyConfig
from ..domain.features import FEATURES, feature_frame, with_labels
from ..domain.models import RegimeRouterEnsemble, _new_champion


def _accuracy(y: np.ndarray, probs: np.ndarray) -> dict:
    if len(y) == 0:
        return {"samples": 0, "positive_label_rate": 0.0, "brier": 0.0, "log_loss": 0.0, "roc_auc": None}
    probs_clipped = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    result = {
        "samples": int(len(y)),
        "positive_label_rate": round(float(np.mean(y)), 4),
        "brier": round(float(brier_score_loss(y, probs_clipped)), 5),
        "log_loss": round(float(log_loss(y, probs_clipped, labels=[0, 1])), 5),
    }
    base_rate = float(np.mean(y))
    baseline_brier = base_rate * (1.0 - base_rate)
    result["brier_skill_score"] = round(1.0 - result["brier"] / baseline_brier, 5) if baseline_brier > 0 else None
    result["roc_auc"] = round(float(roc_auc_score(y, probs_clipped)), 5) if len(np.unique(y)) == 2 else None
    return result


def _buy_hold_return_pct(data: pd.DataFrame, cfg: StrategyConfig) -> float:
    if len(data) == 0:
        return 0.0
    entry = float(data.iloc[0]["open"]) * (1 + cfg.execution_bps / 10_000)
    exit_px = float(data.iloc[-1]["close"]) * (1 - cfg.execution_bps / 10_000)
    return round(100 * (exit_px * (1 - cfg.fee_rate) / (entry * (1 + cfg.fee_rate)) - 1), 4)


def _predict_frame_probs(model, frame: pd.DataFrame, default_prob: float = 0.50) -> np.ndarray:
    probs = np.full(len(frame), np.nan, dtype=float)
    valid_mask = frame[FEATURES].notna().all(axis=1).to_numpy(dtype=bool)
    if not valid_mask.any():
        return probs
    if model is None:
        probs[valid_mask] = default_prob
        return probs
    X = frame.loc[valid_mask, FEATURES].to_numpy(dtype=float)
    probs[valid_mask] = model.predict_proba(X)[:, 1]
    return probs


def run_walk_forward(df: pd.DataFrame, cfg: StrategyConfig, n_folds: int = 4) -> dict:
    """Purged expanding-window walk-forward validation across chronological folds."""
    cfg.validate()
    full_frame = feature_frame(df, cfg)
    labeled = with_labels(df, cfg)
    n = len(full_frame)
    open_times = full_frame["open_time"].to_numpy(dtype=np.int64)

    anchor = int(n * 0.50)
    fold_size = (n - anchor) // max(1, n_folds)
    folds_report = []
    compounded_equity = 1.0
    bh_compounded = 1.0
    total_trades = 0
    winning_trades = 0

    for fold_idx in range(n_folds):
        f_start = anchor + fold_idx * fold_size
        f_end = n if fold_idx == n_folds - 1 else anchor + (fold_idx + 1) * fold_size
        t_start = int(open_times[f_start])
        t_end = int(open_times[f_end - 1])

        train_fold = labeled[labeled["exit_time"] < t_start]
        # A classification label is only eligible when its full exit is inside this fold.
        # Otherwise its target uses prices from the next fold, contaminating fold metrics.
        test_fold_signals = labeled[
            (labeled["open_time"] >= t_start)
            & (labeled["open_time"] <= t_end)
            & (labeled["exit_time"] <= t_end)
        ]
        test_fold_bars = full_frame.iloc[f_start:f_end].copy()

        model, _ = _new_champion()
        if len(train_fold) >= 6 and train_fold["target"].nunique() == 2:
            model.fit(train_fold[FEATURES].to_numpy(dtype=float), train_fold["target"].to_numpy(dtype=int))
            fold_probs = _predict_frame_probs(model, test_fold_bars)
            sig_probs = model.predict_proba(test_fold_signals[FEATURES].to_numpy(dtype=float))[:, 1] if len(test_fold_signals) else np.array([])
        else:
            fold_probs = _predict_frame_probs(None, test_fold_bars, default_prob=0.50)
            sig_probs = np.full(len(test_fold_signals), 0.50)

        bt = simulate(test_fold_bars, fold_probs, cfg.threshold, cfg)
        cls_metrics = _accuracy(test_fold_signals["target"].to_numpy(dtype=int), sig_probs) if len(test_fold_signals) else {}
        bh_ret = _buy_hold_return_pct(test_fold_bars, cfg)

        compounded_equity *= (1.0 + bt["total_return_pct"] / 100.0)
        bh_compounded *= (1.0 + bh_ret / 100.0)
        total_trades += bt["trades"]
        winning_trades += sum(1 for t in bt["closed_trades"] if t["pnl"] > 0)

        folds_report.append({
            "fold": fold_idx + 1,
            "train_signals_purged": int(len(train_fold)),
            "test_signals": int(len(test_fold_signals)),
            "from_utc": datetime.fromtimestamp(t_start / 1000, tz=timezone.utc).isoformat(),
            "to_utc": datetime.fromtimestamp(t_end / 1000, tz=timezone.utc).isoformat(),
            "roc_auc": cls_metrics.get("roc_auc"),
            "trades": bt["trades"],
            "return_pct": bt["total_return_pct"],
            "max_drawdown_pct": bt["max_drawdown_pct"],
            "win_rate": bt["win_rate"],
            "profit_factor": bt["profit_factor"],
            "buy_hold_return_pct": bh_ret,
        })

    summary = {
        "method": f"Purged Expanding-Window Walk-Forward v5.0 ({n_folds} folds)",
        "threshold": cfg.threshold,
        "folds": folds_report,
        "aggregate": {
            "total_oos_trades": total_trades,
            "winning_oos_trades": winning_trades,
            "oos_win_rate": round(winning_trades / total_trades, 4) if total_trades else 0.0,
            "oos_compounded_return_pct": round((compounded_equity - 1.0) * 100.0, 4),
            "buy_hold_compounded_return_pct": round((bh_compounded - 1.0) * 100.0, 4),
        },
    }
    return summary


def train_model(df: pd.DataFrame, cfg: StrategyConfig) -> tuple[dict, dict]:
    """Build a trained bundle and its OOS report without writing files."""
    cfg.validate()
    full_frame = feature_frame(df, cfg)
    labeled = with_labels(df, cfg)

    n = len(full_frame)
    a = int(n * 0.70)
    b = int(n * 0.85)

    train_split = full_frame.iloc[:a].copy()
    val_split = full_frame.iloc[a:b].copy()
    test_split = full_frame.iloc[b:].copy()

    t_a = int(full_frame["open_time"].iloc[a])
    t_b = int(full_frame["open_time"].iloc[b])

    train_labeled = labeled[labeled["exit_time"] < t_a].copy()
    val_labeled = labeled[(labeled["open_time"] >= t_a) & (labeled["exit_time"] < t_b)].copy()
    test_labeled = labeled[labeled["open_time"] >= t_b].copy()

    oos_champion, name = _new_champion()
    can_fit_train = len(train_labeled) >= 6 and train_labeled["target"].nunique() == 2

    if can_fit_train:
        oos_champion.fit(train_labeled[FEATURES].to_numpy(dtype=float), train_labeled["target"].to_numpy(dtype=int))
        val_raw_probs = _predict_frame_probs(oos_champion, val_split)
        if abs(cfg.threshold - 0.35) > 1e-6:
            threshold = float(cfg.threshold)
        else:
            candidate_thresholds = [0.38, 0.40, 0.42]
            best_thr = 0.40
            best_val_score = -1e9
            for cand_thr in candidate_thresholds:
                cand_bt = simulate(val_split, val_raw_probs, cand_thr, cfg)
                if cand_bt["trades"] >= 1:
                    score = cand_bt["chow_utility_score"]
                    if score > best_val_score:
                        best_val_score = score
                        best_thr = cand_thr
            threshold = float(best_thr)

        train_probs = _predict_frame_probs(oos_champion, train_split)
        val_probs = val_raw_probs
        test_probs = _predict_frame_probs(oos_champion, test_split)
        val_eval_probs = oos_champion.predict_proba(val_labeled[FEATURES].to_numpy(dtype=float))[:, 1] if len(val_labeled) else np.array([])
        test_eval_probs = oos_champion.predict_proba(test_labeled[FEATURES].to_numpy(dtype=float))[:, 1] if len(test_labeled) else np.array([])
    else:
        threshold = float(cfg.threshold)
        train_probs = _predict_frame_probs(None, train_split, default_prob=0.55)
        val_probs = _predict_frame_probs(None, val_split, default_prob=0.55)
        test_probs = _predict_frame_probs(None, test_split, default_prob=0.55)
        val_eval_probs = np.full(len(val_labeled), 0.55)
        test_eval_probs = np.full(len(test_labeled), 0.55)

    train_bt = simulate(train_split, train_probs, threshold, cfg)
    val_bt = simulate(val_split, val_probs, threshold, cfg)
    test_bt = simulate(test_split, test_probs, threshold, cfg)

    oos_split = full_frame.iloc[a:].copy()
    oos_probs = _predict_frame_probs(oos_champion if can_fit_train else None, oos_split, default_prob=0.55)
    oos_bt = simulate(oos_split, oos_probs, threshold, cfg)
    oos_valid = oos_split["valid_features"].to_numpy(dtype=bool)
    oos_triggers = oos_split["hourly_trigger"].to_numpy(dtype=bool) & oos_valid
    oos_over_threshold = np.isfinite(oos_probs) & (oos_probs >= threshold) & oos_valid
    oos_signal_opportunities = oos_triggers & (oos_probs >= threshold)
    regime_values = oos_split.loc[oos_valid, "regime_code"].value_counts().to_dict()

    full_probs = _predict_frame_probs(oos_champion if can_fit_train else None, full_frame, default_prob=0.55)
    full_bt = simulate(full_frame, full_probs, threshold, cfg)

    wf_summary = run_walk_forward(df, StrategyConfig(**{**asdict(cfg), "threshold": threshold}), n_folds=4)

    wf_is_robust = bool(
        wf_summary["aggregate"]["total_oos_trades"] >= 6
        and wf_summary["aggregate"]["oos_compounded_return_pct"] > 0.0
    )
    test_is_risk_bounded = bool(
        test_bt["trades"] >= 3
        and test_bt["max_drawdown_pct"] <= cfg.daily_loss_limit * 100.0
    )
    approved = bool(
        test_is_risk_bounded
        and val_bt["total_return_pct"] >= 0.0
        and oos_bt["total_return_pct"] > 0.0
        and (oos_bt["profit_factor"] is not None and oos_bt["profit_factor"] > 1.2)
        and wf_is_robust
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": cfg.symbol,
        "strategy": "Regularized regime-router + VPIN/BOCPD risk filters + ACI (v5.0)",
        "costs": {"fee_rate_per_side": cfg.fee_rate, "execution_bps_per_side": cfg.execution_bps},
        "model": name,
        "split": {
            "total_bars_1m": len(full_frame),
            "train_bars": len(train_split),
            "val_bars": len(val_split),
            "test_bars": len(test_split),
            "total_labeled_signals": len(labeled),
            "train_signals_purged": len(train_labeled),
            "val_signals_purged": len(val_labeled),
            "test_signals": len(test_labeled),
            "val_from_utc": datetime.fromtimestamp(t_a / 1000, tz=timezone.utc).isoformat(),
            "test_from_utc": datetime.fromtimestamp(t_b / 1000, tz=timezone.utc).isoformat(),
        },
        "threshold": threshold,
        "validation_classification": _accuracy(val_labeled["target"].to_numpy(dtype=int), val_eval_probs) if len(val_labeled) else {},
        "test_classification": _accuracy(test_labeled["target"].to_numpy(dtype=int), test_eval_probs) if len(test_labeled) else {},
        "train_trading": train_bt,
        "validation_trading": val_bt,
        "test_trading": test_bt,
        "out_of_sample_combined_trading": oos_bt,
        "full_period_trading": full_bt,
        "walk_forward_summary": wf_summary["aggregate"],
        "signal_diagnostics": {
            "oos_valid_bars": int(oos_valid.sum()),
            "oos_probabilities_at_or_above_threshold": int(oos_over_threshold.sum()),
            "oos_entry_triggers": int(oos_triggers.sum()),
            "oos_bars_meeting_trigger_and_probability": int(oos_signal_opportunities.sum()),
            "oos_regime_bar_counts": {str(int(k)): int(v) for k, v in regime_values.items()},
            "oos_simulated_trades": int(oos_bt["trades"]),
        },
        "paper_gate_checks": {
            "test_at_least_3_trades_and_drawdown_within_daily_limit": test_is_risk_bounded,
            "test_return_positive_diagnostic_only": bool(test_bt["total_return_pct"] > 0.0),
            "validation_return_nonnegative": bool(val_bt["total_return_pct"] >= 0.0),
            "combined_oos_return_positive": bool(oos_bt["total_return_pct"] > 0.0),
            "combined_oos_profit_factor_above_1_2": bool(oos_bt["profit_factor"] is not None and oos_bt["profit_factor"] > 1.2),
            "walk_forward_at_least_6_trades_and_positive": wf_is_robust,
        },
        "baselines": {
            "val_buy_and_hold_return_pct": _buy_hold_return_pct(val_split, cfg),
            "test_buy_and_hold_return_pct": _buy_hold_return_pct(test_split, cfg),
            "oos_buy_and_hold_return_pct": _buy_hold_return_pct(oos_split, cfg),
            "full_buy_and_hold_return_pct": _buy_hold_return_pct(full_frame, cfg),
        },
        "approved_for_default_paper_signals": approved,
    }

    live_champion, _ = _new_champion()
    if len(labeled) >= 6 and labeled["target"].nunique() == 2:
        live_champion.fit(labeled[FEATURES].to_numpy(dtype=float), labeled["target"].to_numpy(dtype=int))
    else:
        live_champion = oos_champion

    train_valid_X = train_split[FEATURES].dropna().to_numpy(dtype=float)
    scaler = StandardScaler().fit(train_valid_X if len(train_valid_X) > 0 else full_frame[FEATURES].dropna().to_numpy(dtype=float))
    online = SGDClassifier(loss="log_loss", alpha=0.01, learning_rate="optimal", random_state=42)
    if len(train_labeled) >= 2 and train_labeled["target"].nunique() == 2:
        X_sgd = scaler.transform(train_labeled[FEATURES].to_numpy(dtype=float))
        y_sgd = train_labeled["target"].to_numpy(dtype=int)
        online.partial_fit(X_sgd, y_sgd, classes=np.array([0, 1]))
    elif len(labeled) >= 2 and labeled["target"].nunique() == 2:
        # This is the deployment challenger, so it may learn from all labels available
        # at the training cutoff. Never fabricate labels to bootstrap it.
        X_sgd = scaler.transform(labeled[FEATURES].to_numpy(dtype=float))
        y_sgd = labeled["target"].to_numpy(dtype=int)
        online.partial_fit(X_sgd, y_sgd, classes=np.array([0, 1]))
    else:
        online = None

    bundle = {
        "champion": live_champion,
        "oos_champion": oos_champion,
        "online": online,
        "scaler": scaler,
        "features": FEATURES,
        "config": asdict(cfg),
        "threshold": float(threshold),
        "oos_start_open_time": t_a,
        "test_start_open_time": t_b,
        "paper_gate": approved,
        "last_training_utc": report["created_utc"],
    }
    return bundle, report


def predict_pair(bundle: dict, features: np.ndarray, use_oos_model: bool = False) -> tuple[float, float]:
    x = np.nan_to_num(features.reshape(1, -1), nan=0.0)
    model_key = "oos_champion" if (use_oos_model and "oos_champion" in bundle) else "champion"
    champion = float(bundle[model_key].predict_proba(x)[0, 1])
    online_model = bundle.get("online")
    online = float(online_model.predict_proba(bundle["scaler"].transform(x))[0, 1]) if online_model is not None else champion
    return champion, online


def predict_detailed(bundle: dict, features: np.ndarray, use_oos_model: bool = False) -> dict:
    """Return full MacroHFT sub-agent breakdown + online SGD challenger probability."""
    x = np.nan_to_num(features.reshape(1, -1), nan=0.0)
    model_key = "oos_champion" if (use_oos_model and "oos_champion" in bundle) else "champion"
    model = bundle[model_key]
    online_model = bundle.get("online")
    online = float(online_model.predict_proba(bundle["scaler"].transform(x))[0, 1]) if online_model is not None else float(model.predict_proba(x)[0, 1])
    if hasattr(model, "predict_regime_breakdown"):
        bd = model.predict_regime_breakdown(x)
        return {
            "champion_p": float(bd["total"][0]),
            "anchor_p": float(bd["anchor"][0]),
            "calm_agent_p": float(bd["calm_agent"][0]),
            "volatile_agent_p": float(bd["volatile_agent"][0]),
            "volatile_gate_weight": float(bd["volatile_gate_weight"][0]),
            "challenger_p": online,
        }
    p_champ = float(model.predict_proba(x)[0, 1])
    return {
        "champion_p": p_champ,
        "anchor_p": p_champ,
        "calm_agent_p": p_champ,
        "volatile_agent_p": p_champ,
        "volatile_gate_weight": 0.5,
        "challenger_p": online,
    }


def learn_mature_outcome(bundle: dict, features: np.ndarray, target: int) -> None:
    if bundle.get("online") is None:
        return
    x_clean = np.nan_to_num(features.reshape(1, -1), nan=0.0)
    x_scaled = bundle["scaler"].transform(x_clean)
    bundle["online"].partial_fit(x_scaled, np.array([target], dtype=int))

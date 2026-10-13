from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .backtest import simulate
from .config import StrategyConfig
from .features import FEATURES, with_labels


def _new_champion():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=220, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            tree_method="hist", eval_metric="logloss", n_jobs=2, random_state=42,
        ), "XGBoost"
    except ImportError:
        return HistGradientBoostingClassifier(
            max_iter=170, max_leaf_nodes=15, l2_regularization=5.0,
            learning_rate=0.05, random_state=42,
        ), "sklearn HistGradientBoosting (instalá .[xgboost] para XGBoost)"


def _accuracy(y: np.ndarray, probs: np.ndarray) -> dict:
    result = {
        "samples": int(len(y)),
        "positive_label_rate": round(float(np.mean(y)), 4),
        "brier": round(float(brier_score_loss(y, probs)), 5),
        "log_loss": round(float(log_loss(y, probs, labels=[0, 1])), 5),
    }
    result["roc_auc"] = round(float(roc_auc_score(y, probs)), 5) if len(np.unique(y)) == 2 else None
    return result


def _buy_hold_return_pct(data: pd.DataFrame, cfg: StrategyConfig) -> float:
    """Full-allocation buy-and-hold reference; not same risk as 25%-sized trades."""
    entry = float(data.iloc[0]["open"]) * (1 + cfg.execution_bps / 10_000)
    exit_px = float(data.iloc[-1]["close"]) * (1 - cfg.execution_bps / 10_000)
    return round(100 * (exit_px * (1 - cfg.fee_rate) / (entry * (1 + cfg.fee_rate)) - 1), 4)


def train_and_report(df: pd.DataFrame, cfg: StrategyConfig, model_path: Path, report_path: Path) -> dict:
    cfg.validate()
    labeled = with_labels(df, cfg)
    if len(labeled) < 5_000:
        raise ValueError(f"Insuficiente historial continuo: {len(labeled)} ejemplos; se requieren al menos 5000")
    n = len(labeled)
    a, b = int(n * 0.70), int(n * 0.85)
    # Gap ensures that a sample's forward label cannot reach into the next split.
    train = labeled.iloc[:a - cfg.horizon_bars]
    val = labeled.iloc[a:b - cfg.horizon_bars]
    test = labeled.iloc[b:]
    if min(len(train), len(val), len(test)) < 200:
        raise ValueError("Divisiones temporales demasiado pequeñas")

    champion, name = _new_champion()
    champion.fit(train[FEATURES].to_numpy(), train["target"].to_numpy())
    val_probs = champion.predict_proba(val[FEATURES].to_numpy())[:, 1]
    test_probs = champion.predict_proba(test[FEATURES].to_numpy())[:, 1]

    # Tune ONLY on validation data. No peeking at the hold-out test interval.
    trials = {}
    viable = []
    for threshold in (0.52, 0.55, 0.58, 0.60, 0.63, 0.66, 0.70):
        bt = simulate(val, val_probs, threshold, cfg)
        trials[str(threshold)] = {k: v for k, v in bt.items() if k != "closed_trades"}
        if bt["trades"] >= 15 and bt["total_return_pct"] > 0 and bt["max_drawdown_pct"] <= 100 * cfg.max_drawdown:
            viable.append((bt["total_return_pct"] - bt["max_drawdown_pct"] / 2, threshold))

    threshold = max(viable)[1] if viable else cfg.threshold
    val_bt = simulate(val, val_probs, threshold, cfg)
    test_bt = simulate(test, test_probs, threshold, cfg)
    approved = bool(viable and test_bt["trades"] >= 15 and test_bt["total_return_pct"] > 0)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": cfg.symbol, "horizon_bars": cfg.horizon_bars,
        "costs": {"fee_rate_per_side": cfg.fee_rate, "execution_bps_per_side": cfg.execution_bps},
        "model": name,
        "split": {"train": len(train), "validation": len(val), "test": len(test),
                  "purge_bars": cfg.horizon_bars,
                  "test_from_utc": datetime.fromtimestamp(int(test.iloc[0]["open_time"]) / 1000, tz=timezone.utc).isoformat()},
        "threshold": threshold,
        "validation_threshold_trials": trials,
        "validation_classification": _accuracy(val["target"].to_numpy(), val_probs),
        "test_classification": _accuracy(test["target"].to_numpy(), test_probs),
        "validation_trading": val_bt,
        "test_trading": test_bt,
        "baselines": {
            "no_trading_return_pct": 0.0,
            "test_full_allocation_buy_and_hold_return_pct": _buy_hold_return_pct(test, cfg),
            "note": "Buy-and-hold usa exposición del 100%; la estrategia asigna 25% por entrada.",
        },
        "approved_for_default_paper_signals": approved,
        "note": "Una evaluación rentable NO autoriza trading real; el modelo final se reentrena usando el historial disponible.",
    }

    # After measuring an untouched test interval, refit for subsequent *future*
    # simulation only. The report refers to the earlier train-only model.
    champion_final, _ = _new_champion()
    champion_final.fit(labeled[FEATURES].to_numpy(), labeled["target"].to_numpy())
    scaler = StandardScaler().fit(labeled[FEATURES].to_numpy())
    online = SGDClassifier(loss="log_loss", alpha=0.001, random_state=42)
    online.partial_fit(scaler.transform(labeled[FEATURES].to_numpy()), labeled["target"].to_numpy(), classes=np.array([0, 1]))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "champion": champion_final, "online": online, "scaler": scaler,
        "features": FEATURES, "config": asdict(cfg), "threshold": float(threshold),
        "paper_gate": approved, "last_training_utc": report["created_utc"],
    }, model_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_bundle(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}; ejecutá 'train' primero")
    # joblib/pickle is NOT safe for untrusted downloaded model files.
    bundle = joblib.load(path)
    if bundle.get("features") != FEATURES:
        raise ValueError("Features incompatibles; reentrená el modelo")
    StrategyConfig(**bundle["config"]).validate()
    return bundle


def predict_pair(bundle: dict, features: np.ndarray) -> tuple[float, float]:
    x = features.reshape(1, -1)
    champion = float(bundle["champion"].predict_proba(x)[0, 1])
    online = float(bundle["online"].predict_proba(bundle["scaler"].transform(x))[0, 1])
    return champion, online


def learn_mature_outcome(bundle: dict, features: np.ndarray, target: int) -> None:
    # The challenger learns, the approved champion stays frozen.
    x = bundle["scaler"].transform(features.reshape(1, -1))
    bundle["online"].partial_fit(x, np.array([target], dtype=int))


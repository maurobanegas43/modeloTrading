"""Entrenamiento y evaluación temporal para USDⓈ-M perpetual futures."""

from __future__ import annotations

from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from ..domain.config import FuturesConfig, StrategyConfig
from ..domain.features import feature_frame, interval_decision_mask, with_futures_directional_labels
from ..domain.futures_features import FUTURES_FEATURES, attach_futures_context
from ..domain.futures_backtest import probabilities_for_frame, simulate_futures
from ..domain.models import DirectionalFuturesModel


def _class_metrics(samples: pd.DataFrame, probabilities: np.ndarray) -> dict:
    if samples.empty:
        return {"samples": 0, "accuracy": None, "macro_f1": None, "log_loss": None, "confusion_matrix": []}
    truth = samples["target"].to_numpy(dtype=int)
    predicted = np.argmax(probabilities, axis=1)
    matrix = np.zeros((3, 3), dtype=int)
    for actual, guess in zip(truth, predicted):
        matrix[int(actual), int(guess)] += 1
    return {
        "samples": int(len(truth)),
        "accuracy": round(float(accuracy_score(truth, predicted)), 5),
        "always_flat_baseline_accuracy": round(float(np.mean(truth == 1)), 5),
        "accuracy_minus_always_flat_baseline": round(float(accuracy_score(truth, predicted) - np.mean(truth == 1)), 5),
        "macro_f1": round(float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)), 5),
        "log_loss": round(float(log_loss(truth, probabilities, labels=[0, 1, 2])), 5),
        "class_counts": {"short": int((truth == 0).sum()), "flat": int((truth == 1).sum()), "long": int((truth == 2).sum())},
        "confusion_matrix_short_flat_long": matrix.tolist(),
    }


def _decision_rows(frame: pd.DataFrame, cfg: FuturesConfig) -> pd.DataFrame:
    return frame.loc[interval_decision_mask(frame, cfg.decision_interval_minutes)].copy()


def _fit(model: DirectionalFuturesModel, samples: pd.DataFrame) -> DirectionalFuturesModel:
    return model.fit(samples[FUTURES_FEATURES].to_numpy(dtype=float), samples["target"].to_numpy(dtype=int))


def _walk_forward(frame: pd.DataFrame, samples: pd.DataFrame, funding: pd.DataFrame | None, cfg: FuturesConfig, folds: int = 4) -> dict:
    n = len(frame)
    anchor = int(n * 0.50)
    fold_size = max(1, (n - anchor) // folds)
    reports: list[dict] = []
    compound = 1.0
    trades = 0
    for fold in range(folds):
        start = anchor + fold * fold_size
        end = n if fold == folds - 1 else min(n, anchor + (fold + 1) * fold_size)
        if start >= end:
            continue
        start_time = int(frame.iloc[start]["open_time"])
        end_time = int(frame.iloc[end - 1]["open_time"])
        train = samples.loc[samples["exit_time"] < start_time]
        test_samples = samples.loc[
            (samples["open_time"] >= start_time)
            & (samples["open_time"] <= end_time)
            & (samples["exit_time"] <= end_time)
        ]
        test_frame = frame.iloc[start:end].copy()
        class_counts = np.bincount(train["target"].to_numpy(dtype=int), minlength=3) if len(train) else np.zeros(3, dtype=int)
        if np.any(class_counts < 12):
            reports.append({"fold": fold + 1, "status": "omitido_por_muestras_insuficientes", "train_samples": len(train), "train_class_counts": class_counts.tolist()})
            continue
        model = _fit(DirectionalFuturesModel(), train)
        rows = _decision_rows(test_frame, cfg)
        probs = probabilities_for_frame(model, test_frame, rows, FUTURES_FEATURES)
        bt = simulate_futures(test_frame, probs, funding, cfg)
        compound *= 1.0 + bt["total_return_pct"] / 100.0
        trades += bt["trades"]
        row: dict = {
            "fold": fold + 1,
            "status": "evaluado",
            "train_samples": int(len(train)),
            "test_labeled_samples": int(len(test_samples)),
            "from_utc": datetime.fromtimestamp(start_time / 1000, tz=timezone.utc).isoformat(),
            "to_utc": datetime.fromtimestamp(end_time / 1000, tz=timezone.utc).isoformat(),
            "trading": {key: value for key, value in bt.items() if key != "closed_trades"},
        }
        if len(test_samples):
            actual_probs = model.predict_proba(test_samples[FUTURES_FEATURES].to_numpy(dtype=float))
            row["classification"] = _class_metrics(test_samples, actual_probs)
        reports.append(row)
    return {
        "method": "walk-forward expansivo purgado, 4 folds cronológicos, política fijada antes del test",
        "folds": reports,
        "total_oos_trades": int(trades),
        "oos_compounded_return_pct": round((compound - 1) * 100, 5),
    }


def train_futures_model(
    candles: pd.DataFrame,
    funding: pd.DataFrame | None = None,
    cfg: FuturesConfig | None = None,
) -> tuple[dict, dict]:
    """Ajusta el candidato, mantiene test ciego y devuelve bundle/reporte sin persistirlos."""
    training_started = time.perf_counter()
    cfg = cfg or FuturesConfig()
    cfg.validate()
    strategy_cfg = StrategyConfig(symbol=cfg.symbol)
    frame = attach_futures_context(feature_frame(candles, strategy_cfg), funding)
    samples = attach_futures_context(with_futures_directional_labels(candles, cfg, funding), funding)
    if samples.empty:
        raise ValueError("No hay muestras futures válidas; verifica continuidad y warmup de las velas")

    n = len(frame)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_time = int(frame.iloc[train_end]["open_time"])
    val_end_time = int(frame.iloc[val_end]["open_time"])
    test_end_time = int(frame.iloc[-1]["open_time"])
    train = samples.loc[(samples["open_time"] < train_time) & (samples["exit_time"] < train_time)]
    validation = samples.loc[
        (samples["open_time"] >= train_time)
        & (samples["open_time"] < val_end_time)
        & (samples["exit_time"] < val_end_time)
    ]
    train_plus_validation = samples.loc[
        (samples["open_time"] < val_end_time)
        & (samples["exit_time"] < val_end_time)
    ]
    test = samples.loc[
        (samples["open_time"] >= val_end_time)
        & (samples["exit_time"] <= test_end_time)
    ]
    train_counts = np.bincount(train["target"].to_numpy(dtype=int), minlength=3)
    if np.any(train_counts < 12):
        raise ValueError(f"Muestras insuficientes para aprender SHORT/FLAT/LONG en train: {train_counts.tolist()}")

    # Política congelada antes de abrir el tramo ciego: argmax entre SHORT/FLAT/LONG,
    # con un único confidence floor para las clases direccionales.
    threshold = cfg.probability_threshold
    validation_model = _fit(DirectionalFuturesModel(), train)
    validation_rows = _decision_rows(frame.iloc[train_end:val_end].copy(), cfg)
    validation_frame = frame.iloc[train_end:val_end].copy()
    validation_probs = probabilities_for_frame(validation_model, validation_frame, validation_rows, FUTURES_FEATURES)
    validation_bt = simulate_futures(validation_frame, validation_probs, funding, cfg, threshold)
    threshold_sensitivity = []
    for candidate_threshold in (0.40, 0.45, 0.50):
        candidate_bt = validation_bt if abs(candidate_threshold - threshold) < 1e-12 else simulate_futures(
            validation_frame, validation_probs, funding, cfg, candidate_threshold
        )
        threshold_sensitivity.append({
            "confidence_floor": candidate_threshold,
            "closed_trades": candidate_bt["trades"],
            "return_pct": candidate_bt["total_return_pct"],
            "max_drawdown_pct": candidate_bt["max_drawdown_pct"],
            "profit_factor": candidate_bt["profit_factor"],
            "action_counts": candidate_bt["action_counts"],
        })
    validation_cls_probs = validation_model.predict_proba(validation[FUTURES_FEATURES].to_numpy(dtype=float)) if len(validation) else np.empty((0, 3))

    if np.any(np.bincount(train_plus_validation["target"].to_numpy(dtype=int), minlength=3) < 12):
        raise ValueError("Muestras insuficientes al ampliar train con validación")
    holdout_model = _fit(DirectionalFuturesModel(), train_plus_validation)
    test_frame = frame.iloc[val_end:].copy()
    test_rows = _decision_rows(test_frame, cfg)
    test_probs = probabilities_for_frame(holdout_model, test_frame, test_rows, FUTURES_FEATURES)
    test_bt = simulate_futures(test_frame, test_probs, funding, cfg, threshold)
    test_cls_probs = holdout_model.predict_proba(test[FUTURES_FEATURES].to_numpy(dtype=float)) if len(test) else np.empty((0, 3))

    walk = _walk_forward(frame, samples, funding, cfg)
    final_model = _fit(DirectionalFuturesModel(), samples)
    bundle = {
        "format": "binance-usdm-directional-paper-v2",
        "model": final_model,
        "features": FUTURES_FEATURES,
        "config": cfg.__dict__.copy(),
        "trained_through_open_time": int(frame["open_time"].iloc[-1]),
        "trained_through_utc": datetime.fromtimestamp(
            int(frame["open_time"].iloc[-1]) / 1000, tz=timezone.utc
        ).isoformat(),
        "decision_policy": {
            "classes": {"0": "SHORT", "1": "FLAT", "2": "LONG"},
            "selection": "argmax",
            "min_directional_probability": threshold,
            "entry_timeframe_minutes": cfg.decision_interval_minutes,
            "context_timeframe_minutes": 60,
            "context_role": "soft model features; never a hard entry gate",
            "holding_horizon_minutes": cfg.holding_horizon_minutes,
            "execution": "siguiente apertura de vela 1m posterior al cierre de decisión",
        },
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "market": "Binance USDⓈ-M BTCUSDT perpetual",
        "strategy": "clasificación direccional SHORT/FLAT/LONG con costes y funding netos",
        "model": "HistGradientBoosting + ExtraTrees + LogisticRegression (soft voting, regularizado)",
        "model_capacity": {
            "type": "soft-voting ensemble; 3 classes",
            "hist_gradient_boosting": {
                "iterations": 100,
                "max_leaf_nodes_per_tree": 7,
                "trees_per_iteration": 3,
                "min_samples_leaf": 18,
                "max_terminal_leaves_theoretical": 2_100,
            },
            "extra_trees": {
                "estimators": 180,
                "max_depth": 5,
                "max_terminal_leaves_per_tree_theoretical": 32,
                "min_samples_leaf": 14,
                "max_terminal_leaves_theoretical": 5_760,
            },
            "logistic_regression": {"classes": 3, "max_iter": 700, "C": 0.08},
            "total_tree_terminal_leaves_theoretical": 7_860,
            "actual_leaves_depend_on_training_data": True,
        },
        "feature_count": len(FUTURES_FEATURES),
        "features": FUTURES_FEATURES,
        "policy": bundle["decision_policy"],
        "assumptions": {
            "fee_rate_per_side": cfg.fee_rate,
            "execution_bps_per_side": cfg.execution_bps,
            "leverage": cfg.leverage,
            "max_margin_fraction": cfg.max_margin_fraction,
            "holding_horizon_minutes": cfg.holding_horizon_minutes,
            "minimum_net_edge_bps_for_directional_label": cfg.min_net_edge_bps,
            "funding_rows": int(len(funding)) if funding is not None else 0,
            "mark_price_available": bool({"mark_high", "mark_low", "mark_close"}.issubset(candles.columns)),
        },
        "data": {
            "bars_1m": int(len(frame)),
            "labeled_decision_samples": int(len(samples)),
            "decision_interval_minutes": cfg.decision_interval_minutes,
            "holding_horizon_minutes": cfg.holding_horizon_minutes,
            "label_counts_short_flat_long": np.bincount(samples["target"].to_numpy(dtype=int), minlength=3).tolist(),
            "train_samples_purged": int(len(train)),
            "validation_samples_purged": int(len(validation)),
            "test_samples": int(len(test)),
            "train_from_utc": datetime.fromtimestamp(int(frame.iloc[0]["open_time"]) / 1000, tz=timezone.utc).isoformat(),
            "validation_from_utc": datetime.fromtimestamp(train_time / 1000, tz=timezone.utc).isoformat(),
            "test_from_utc": datetime.fromtimestamp(val_end_time / 1000, tz=timezone.utc).isoformat(),
            "data_through_utc": datetime.fromtimestamp(test_end_time / 1000, tz=timezone.utc).isoformat(),
        },
        "validation_classification": _class_metrics(validation, validation_cls_probs),
        "validation_trading": {key: value for key, value in validation_bt.items() if key != "closed_trades"},
        "validation_threshold_sensitivity": threshold_sensitivity,
        "blind_test_classification": _class_metrics(test, test_cls_probs),
        "blind_test_trading": {key: value for key, value in test_bt.items() if key != "closed_trades"},
        "walk_forward": walk,
        "interpretation": "Resultados retrospectivos de paper; no prueban rentabilidad futura ni ejecución real.",
    }
    report["training_pipeline_seconds"] = round(time.perf_counter() - training_started, 3)
    return bundle, report


def evaluate_futures_bundle(bundle: dict, candles: pd.DataFrame, funding: pd.DataFrame | None = None) -> dict:
    """Evalúa solo datos que llegaron después del entrenamiento, evitando in-sample."""
    cfg = FuturesConfig(**bundle["config"])
    frame = attach_futures_context(feature_frame(candles, StrategyConfig(symbol=cfg.symbol)), funding)
    frame = frame.loc[frame["open_time"] > int(bundle["trained_through_open_time"])].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No hay velas posteriores al entrenamiento; descarga datos futuros antes del replay")
    rows = _decision_rows(frame, cfg)
    probs = probabilities_for_frame(bundle["model"], frame, rows, bundle["features"])
    bt = simulate_futures(frame, probs, funding, cfg, float(bundle["decision_policy"]["min_directional_probability"]))
    return {key: value for key, value in bt.items() if key != "closed_trades"}

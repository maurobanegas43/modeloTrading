from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from .features import FEATURES


def _make_sub_agent(l2_reg: float = 8.0, max_leaf_nodes: int = 5, c_lr: float = 0.10) -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_iter=60,
                    learning_rate=0.03,
                    max_leaf_nodes=min(max_leaf_nodes, 4),
                    min_samples_leaf=15,
                    l2_regularization=l2_reg,
                    random_state=42,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=120,
                    max_depth=2,
                    min_samples_leaf=10,
                    random_state=42,
                ),
            ),
            (
                "lr",
                LogisticRegression(C=c_lr, max_iter=500, random_state=42),
            ),
        ],
        voting="soft",
    )


class RegimeRouterEnsemble:
    """MacroHFT-inspired (Zong et al., KDD 2024) Context-Aware Regime Routing Ensemble.

    Decomposes training dynamics into two regime-specialized sub-agents (Calm/Normal Volatility
    vs. High-Volatility / Elevated VPIN Order-Flow Toxicity) plus a global anchor ensemble,
    and blends their probabilities via a contextual sigmoid meta-router over ATR, VPIN, and BOCPD.
    """

    def __init__(self) -> None:
        self.scaler = RobustScaler()
        self.anchor_agent = _make_sub_agent(l2_reg=8.0, max_leaf_nodes=5, c_lr=0.10)
        self.calm_agent = _make_sub_agent(l2_reg=6.0, max_leaf_nodes=6, c_lr=0.15)
        self.volatile_agent = _make_sub_agent(l2_reg=12.0, max_leaf_nodes=4, c_lr=0.06)
        self.atr_idx = FEATURES.index("atr_pct_1h")
        self.vpin_idx = FEATURES.index("vpin_1h")
        self.bocpd_idx = FEATURES.index("bocpd_shock_1h")
        self.atr_median = 0.0040
        self.atr_scale = 0.0015
        self.vpin_median = 0.15
        self.vpin_scale = 0.08

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegimeRouterEnsemble":
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        y_arr = np.asarray(y, dtype=int)

        atr_vals = X_arr[:, self.atr_idx]
        vpin_vals = X_arr[:, self.vpin_idx]
        self.atr_median = float(np.median(atr_vals)) if len(atr_vals) else 0.0040
        self.atr_scale = float(max(1e-4, np.std(atr_vals))) if len(atr_vals) else 0.0015
        self.vpin_median = float(np.median(vpin_vals)) if len(vpin_vals) else 0.15
        self.vpin_scale = float(max(1e-3, np.std(vpin_vals))) if len(vpin_vals) else 0.08

        is_high_vol = (atr_vals > self.atr_median) | (vpin_vals > self.vpin_median)
        calm_weights = np.where(~is_high_vol, 1.65, 0.65)
        vol_weights = np.where(is_high_vol, 1.65, 0.65)

        X_scaled = self.scaler.fit_transform(X_arr)
        self.anchor_agent.fit(X_scaled, y_arr)
        self.calm_agent.fit(X_scaled, y_arr, sample_weight=calm_weights)
        self.volatile_agent.fit(X_scaled, y_arr, sample_weight=vol_weights)
        return self

    def _gate_weight(self, X_arr: np.ndarray) -> np.ndarray:
        z_atr = (X_arr[:, self.atr_idx] - self.atr_median) / self.atr_scale
        z_vpin = (X_arr[:, self.vpin_idx] - self.vpin_median) / self.vpin_scale
        bocpd = X_arr[:, self.bocpd_idx]
        logit = np.clip(0.9 * z_atr + 0.5 * z_vpin + 1.5 * bocpd, -8.0, 8.0)
        return 1.0 / (1.0 + np.exp(-logit))

    def predict_regime_breakdown(self, X: np.ndarray) -> dict[str, np.ndarray]:
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_scaled = self.scaler.transform(X_arr)
        p_anchor = self.anchor_agent.predict_proba(X_scaled)[:, 1]
        p_calm = self.calm_agent.predict_proba(X_scaled)[:, 1]
        p_vol = self.volatile_agent.predict_proba(X_scaled)[:, 1]
        gate = self._gate_weight(X_arr)
        p_routed = (1.0 - gate) * p_calm + gate * p_vol
        p_total = 0.60 * p_anchor + 0.40 * p_routed
        return {
            "total": p_total,
            "anchor": p_anchor,
            "calm_agent": p_calm,
            "volatile_agent": p_vol,
            "volatile_gate_weight": gate,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        bd = self.predict_regime_breakdown(X)
        p1 = np.clip(bd["total"], 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p1, p1])


def _new_champion():
    return (
        RegimeRouterEnsemble(),
        "Regularized_RegimeRouter (HGB + ExtraTrees + Logistic; low-sample v5.0)",
    )


class DirectionalFuturesModel:
    """Clasificador supervisado de tres acciones: SHORT, FLAT y LONG.

    Las probabilidades expresan resultados históricos netos por dirección; la decisión
    queda separada para que el simulador pueda abstenerse si no ve ventaja suficiente.
    """

    CLASSES = np.array([0, 1, 2], dtype=int)  # SHORT, FLAT, LONG

    def __init__(self) -> None:
        self.scaler = RobustScaler()
        self.model = VotingClassifier(
            estimators=[
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_iter=100,
                        learning_rate=0.035,
                        max_leaf_nodes=7,
                        min_samples_leaf=18,
                        l2_regularization=8.0,
                        random_state=42,
                    ),
                ),
                (
                    "et",
                    ExtraTreesClassifier(
                        n_estimators=180,
                        max_depth=5,
                        min_samples_leaf=14,
                        max_features=0.8,
                        random_state=42,
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(C=0.08, max_iter=700, random_state=42),
                ),
            ],
            voting="soft",
        )
        self.classes_ = self.CLASSES.copy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DirectionalFuturesModel":
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        y_arr = np.asarray(y, dtype=int)
        counts = np.bincount(y_arr, minlength=3)
        if X_arr.ndim != 2 or len(X_arr) != len(y_arr) or np.any(counts < 12):
            raise ValueError(f"Se requieren al menos 12 ejemplos de cada clase SHORT/FLAT/LONG; conteos={counts.tolist()}")
        self.scaler.fit(X_arr)
        X_scaled = self.scaler.transform(X_arr)
        # La etiqueta FLAT suele ser mayoritaria. Una ponderación inversa plena
        # sobrecorrige las clases direccionales; la raíz cuadrada conserva señal
        # de SHORT/LONG sin borrar la prevalencia real de FLAT.
        class_weights = np.sqrt(len(y_arr) / (3.0 * np.maximum(counts, 1)))
        class_weights /= float(np.mean(class_weights[y_arr]))
        weights = np.clip(class_weights[y_arr], 0.75, 1.35)
        self.model.fit(X_scaled, y_arr, sample_weight=weights)
        self.classes_ = np.asarray(self.model.classes_, dtype=int)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        raw = self.model.predict_proba(self.scaler.transform(X_arr))
        aligned = np.zeros((len(X_arr), 3), dtype=float)
        for column, label in enumerate(self.classes_):
            aligned[:, int(label)] = raw[:, column]
        return aligned

    def predict_actions(
        self,
        X: np.ndarray,
        probability_threshold: float = 0.40,
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_proba(X)
        actions = np.zeros(len(probabilities), dtype=int)  # 0 = FLAT
        reasons: list[str] = []
        for idx, (p_short, p_flat, p_long) in enumerate(probabilities):
            winner = int(np.argmax((p_short, p_flat, p_long)))
            if winner == 1:
                reasons.append("FLAT es la clase más probable")
            elif max(p_short, p_long) < probability_threshold:
                reasons.append("confianza direccional debajo del mínimo")
            else:
                actions[idx] = winner
                reasons.append("LONG más probable" if winner == 2 else "SHORT más probable")
        return actions, np.asarray(reasons, dtype=object)


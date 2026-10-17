import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from trading_ml.backtest import AdaptiveConformalThreshold, _compute_position_fraction, simulate
from trading_ml.config import HOUR_MS, MINUTE_MS, StrategyConfig
from trading_ml.features import FEATURES, feature_frame, with_labels
from trading_ml.live import VirtualWallet
from trading_ml.modeling import RegimeRouterEnsemble, learn_mature_outcome


def synthetic_candles(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0006, n)))
    op = np.r_[100.0, close[:-1]]
    return pd.DataFrame({
        "open_time": 1_700_000_040_000 + np.arange(n, dtype=np.int64) * MINUTE_MS,
        "open": op,
        "high": np.maximum(op, close) * 1.0005,
        "low": np.minimum(op, close) * 0.9995,
        "close": close,
        "volume": rng.uniform(4, 15, n),
        "taker_buy_volume": rng.uniform(0.5, 3, n),
    })


class TestFeatures(unittest.TestCase):
    def test_no_future_leak_in_features(self):
        df = synthetic_candles()
        baseline = feature_frame(df)
        changed = df.copy()
        changed.loc[350:, "close"] = changed.loc[350:, "close"] * 2
        other = feature_frame(changed)
        np.testing.assert_allclose(
            baseline.loc[250, FEATURES].to_numpy(dtype=float),
            other.loc[250, FEATURES].to_numpy(dtype=float),
        )

    def test_labeled_examples_require_mature_future(self):
        df = synthetic_candles(200)
        labeled = with_labels(df, StrategyConfig(horizon_bars=15))
        self.assertGreater(len(labeled), 0)
        self.assertTrue(labeled["open_time"].max() <= df.iloc[-16]["open_time"])
        self.assertEqual(labeled["target"].isna().sum(), 0)
        self.assertIn("exit_time", labeled.columns)

    def test_gap_discards_affected_features(self):
        df = synthetic_candles()
        df.loc[200:, "open_time"] += MINUTE_MS * 3
        features = feature_frame(df)
        self.assertFalse(bool(features.loc[201, "valid_features"]))
        self.assertTrue(bool(features.loc[270, "valid_features"]))

    def test_causal_hour_close_alignment(self):
        df = synthetic_candles(300)
        features = feature_frame(df)
        hour_closes = features.loc[features["is_hour_close"], "open_time"]
        for t in hour_closes:
            self.assertEqual((int(t) + MINUTE_MS) % HOUR_MS, 0)

    def test_vpin_and_bocpd_features_bounded(self):
        df = synthetic_candles(360)
        features = feature_frame(df)
        self.assertTrue((features["vpin_1h"].between(0.0, 1.0)).all())
        self.assertTrue((features["bocpd_shock_1h"].between(0.0, 1.0)).all())


class TestBacktestAndModels(unittest.TestCase):
    def test_no_same_bar_trade_and_costs_apply(self):
        df = synthetic_candles(100)
        df["open"] = 100.0
        df["close"] = 100.0
        df["high"] = 100.0
        df["low"] = 100.0
        cfg = StrategyConfig(horizon_bars=3, stop_loss=0.5)
        probs = np.zeros(len(df))
        probs[2] = 0.99
        output = simulate(df, probs, 0.6, cfg)
        self.assertEqual(output["trades"], 1)
        trade = output["closed_trades"][0]
        self.assertEqual(trade["entry_time"], pd.to_datetime(df.iloc[3]["open_time"], unit="ms", utc=True).isoformat())
        self.assertLess(output["total_return_pct"], 0)
        self.assertIn("sharpe_ratio", output)
        self.assertIn("calmar_ratio", output)

    def test_stop_loss_is_checked(self):
        df = synthetic_candles(100)
        cfg = StrategyConfig(horizon_bars=15, stop_loss=0.01)
        df.loc[3, "low"] = df.loc[3, "open"] * 0.7
        probs = np.zeros(len(df))
        probs[2] = 0.99
        output = simulate(df, probs, 0.6, cfg)
        self.assertEqual(output["closed_trades"][0]["reason"], "stop_loss")

    def test_trailing_stop_locks_in_profit(self):
        df = synthetic_candles(100)
        df["open"] = 100.0
        df["close"] = 100.0
        df["high"] = 100.0
        df["low"] = 100.0
        cfg = StrategyConfig(horizon_bars=20, initial_stop=0.012, trailing_activation=0.015, trailing_delta=0.010)
        df.loc[4, "open"] = 102.0
        df.loc[4, "low"] = 102.0
        df.loc[4, "high"] = 103.0
        df.loc[4, "close"] = 102.5
        df.loc[5, "open"] = 102.2
        df.loc[5, "low"] = 101.5
        probs = np.zeros(len(df))
        probs[2] = 0.99
        output = simulate(df, probs, 0.6, cfg)
        self.assertEqual(output["closed_trades"][0]["reason"], "trailing_stop")
        self.assertGreater(output["closed_trades"][0]["pnl"], 0.0)

    def test_gap_never_executes_scheduled_trade(self):
        df = synthetic_candles(100)
        df.loc[3:, "open_time"] += 10 * MINUTE_MS
        probs = np.zeros(len(df))
        probs[2] = 0.99
        output = simulate(df, probs, 0.6, StrategyConfig())
        self.assertEqual(output["trades"], 0)

    def test_dynamic_position_sizing_bounds(self):
        cfg = StrategyConfig(allocation=0.25, dynamic_sizing=True)
        low_vol_high_p = _compute_position_fraction(0.85, 0.35, 0.0020, cfg, vpin=0.10, bocpd_shock=0.05)
        high_vol_low_p = _compute_position_fraction(0.36, 0.35, 0.0090, cfg, vpin=0.60, bocpd_shock=0.80)
        self.assertGreater(low_vol_high_p, high_vol_low_p)
        self.assertLessEqual(low_vol_high_p, 0.35)
        self.assertGreaterEqual(high_vol_low_p, 0.10)

    def test_aci_threshold_rises_after_loss(self):
        aci = AdaptiveConformalThreshold(base_threshold=0.42, gamma=0.02, target_alpha=0.30)
        t0 = aci.current_threshold()
        t_after_loss = aci.update(trade_won=False)
        self.assertGreater(t_after_loss, t0)
        t_shock = aci.current_threshold(bocpd_shock=0.90)
        self.assertGreater(t_shock, t_after_loss)

    def test_macrohft_regime_router_ensemble(self):
        rng = np.random.default_rng(7)
        X = rng.normal(0, 0.01, size=(40, len(FEATURES)))
        y = np.array([0, 1] * 20, dtype=int)
        model = RegimeRouterEnsemble().fit(X, y)
        bd = model.predict_regime_breakdown(X[:3])
        self.assertEqual(len(bd["total"]), 3)
        self.assertTrue((bd["volatile_gate_weight"] >= 0.0).all() and (bd["volatile_gate_weight"] <= 1.0).all())


class TestPaperOnly(unittest.TestCase):
    def test_paper_wallet_never_spends_more_than_allocated(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = StrategyConfig(horizon_bars=3)
            w = VirtualWallet(cfg, Path(directory) / "state.json", Path(directory) / "events.jsonl")
            t = 1_700_000_040_000
            result = w.on_bar(t, 100, 99, 100, 99.99, 100.01, 0.99, 0.99, 0.60, True, True, quiet=True)
            self.assertEqual(result, "compra virtual")
            self.assertAlmostEqual(w.s["cash"], 750.0)
            self.assertGreater(w.s["qty"], 0)
            self.assertTrue((Path(directory) / "events.jsonl").is_file())

    def test_challenger_learns_matured_label(self):
        scaler = StandardScaler().fit(np.array([[0.0], [1.0], [2.0]]))
        online = SGDClassifier(loss="log_loss", random_state=42)
        online.partial_fit(scaler.transform([[0.0], [1.0]]), np.array([0, 1]), classes=np.array([0, 1]))
        bundle = {"scaler": scaler, "online": online}
        before = online.coef_.copy()
        learn_mature_outcome(bundle, np.array([0.5]), 1)
        self.assertFalse(np.array_equal(before, online.coef_))


    def test_chow_abstention_penalty_and_regimes(self) -> None:
        from trading_ml.backtest import compute_chow_utility_score
        static_bt = {"trades": 1, "total_return_pct": 0.40, "max_drawdown_pct": 0.20, "profit_factor": 2.5, "sortino_ratio": 1.2}
        active_bt = {"trades": 6, "total_return_pct": 1.50, "max_drawdown_pct": 0.70, "profit_factor": 2.5, "sortino_ratio": 2.0}
        s_eval = compute_chow_utility_score(static_bt, target_trades=6, chow_weight=0.80, require_gate=True, val_ret_pct=0.2, test_ret_pct=0.2)
        a_eval = compute_chow_utility_score(active_bt, target_trades=6, chow_weight=0.80, require_gate=True, val_ret_pct=0.5, test_ret_pct=0.5)
        self.assertGreater(s_eval["chow_abstention_penalty"], 0.50)
        self.assertEqual(a_eval["chow_abstention_penalty"], 0.0)
        self.assertGreater(a_eval["chow_utility_score"], s_eval["chow_utility_score"])


if __name__ == "__main__":
    unittest.main()

"""API de compatibilidad de modelado; casos de uso en ``application``."""

from .application.modeling import (
    _accuracy,
    _buy_hold_return_pct,
    _predict_frame_probs,
    learn_mature_outcome,
    predict_detailed,
    predict_pair,
)
from .domain.models import RegimeRouterEnsemble, _make_sub_agent, _new_champion
from .application.modeling import run_walk_forward as _run_walk_forward
from .application.modeling import train_model
from .adapters.model_store import FileBundleStore, JsonReportStore


def load_bundle(path):
    return FileBundleStore(path).load()


def run_walk_forward(df, cfg, n_folds=4, report_path=None):
    summary = _run_walk_forward(df, cfg, n_folds=n_folds)
    if report_path is not None:
        JsonReportStore(report_path).save(summary)
    return summary


def train_and_report(df, cfg, model_path, report_path):
    bundle, report = train_model(df, cfg)
    FileBundleStore(model_path).save(bundle)
    JsonReportStore(report_path).save(report)
    return report


__all__ = [
    "RegimeRouterEnsemble", "learn_mature_outcome", "load_bundle",
    "predict_detailed", "predict_pair", "run_walk_forward", "train_and_report",
]

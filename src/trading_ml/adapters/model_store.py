"""Persistencia local de modelos y reportes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

from ..domain.config import StrategyConfig
from ..domain.features import FEATURES
from ..domain.config import FuturesConfig
from ..domain.models import DirectionalFuturesModel
from ..domain.futures_features import FUTURES_FEATURES


class FileBundleStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def save(self, bundle: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            joblib.dump(bundle, temp)
            os.replace(temp, self.path)
        finally:
            temp.unlink(missing_ok=True)

    def load(self) -> dict:
        if not self.path.is_file():
            raise FileNotFoundError(f"No existe {self.path}; ejecutá 'train' primero")
        bundle = joblib.load(self.path)
        if bundle.get("features") != FEATURES:
            raise ValueError("Features incompatibles; reentrená el modelo")
        StrategyConfig(**bundle["config"]).validate()
        return bundle

    def modified_at(self) -> float:
        return self.path.stat().st_mtime if self.path.is_file() else 0.0


class FuturesBundleStore:
    """Persistencia separada para el bundle USDⓈ-M LONG/FLAT/SHORT."""

    FORMAT = "binance-usdm-directional-paper-v2"

    def __init__(self, path: Path):
        self.path = Path(path)

    def save(self, bundle: dict) -> None:
        if bundle.get("format") != self.FORMAT:
            raise ValueError("Formato de bundle USDⓈ-M incompatible")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            joblib.dump(bundle, temp)
            os.replace(temp, self.path)
        finally:
            temp.unlink(missing_ok=True)

    def load(self) -> dict:
        if not self.path.is_file():
            raise FileNotFoundError(f"No existe {self.path}; ejecutá 'train-futures' primero")
        bundle = joblib.load(self.path)
        if bundle.get("format") != self.FORMAT or bundle.get("features") != FUTURES_FEATURES:
            raise ValueError("Bundle USDⓈ-M incompatible; reentrená el modelo")
        if not isinstance(bundle.get("model"), DirectionalFuturesModel):
            raise ValueError("El bundle no contiene un modelo LONG/FLAT/SHORT")
        if "trained_through_open_time" not in bundle:
            raise ValueError("Falta el corte temporal de entrenamiento; el bundle no puede evaluarse sin riesgo de leakage")
        FuturesConfig(**bundle["config"]).validate()
        return bundle


class JsonReportStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def save(self, report: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, self.path)
        finally:
            temp.unlink(missing_ok=True)

"""Adaptador JSONL/JSON para el wallet paper y el estado de sesión."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..domain.config import StrategyConfig
from ..domain.paper_wallet import PaperWallet


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")


class FileBackedPaperWallet(PaperWallet):
    """Combina reglas puras del wallet con su persistencia local."""

    def __init__(self, cfg: StrategyConfig, path: Path, events: Path, reset: bool = False):
        self.path, self.events = Path(path), Path(events)
        state = None
        if not reset and self.path.is_file():
            state = json.loads(self.path.read_text(encoding="utf-8"))
        super().__init__(cfg, state=state, event_sink=lambda event: append_jsonl(self.events, event))

    def on_bar(self, *args, persist: bool = True, **kwargs) -> str:
        note = super().on_bar(*args, **kwargs)
        durable_actions = {
            "compra virtual", "trailing stop virtual", "stop virtual",
            "salida por cambio drástico de tendencia", "cierre por cambio de tendencia",
            "cierre por horizonte", "límite de pérdida diaria", "bloqueo por drawdown",
        }
        if persist or note in durable_actions or note == "cotización desactualizada":
            self.persist()
        return note

    def persist(self) -> None:
        atomic_write_json(self.path, self.s)

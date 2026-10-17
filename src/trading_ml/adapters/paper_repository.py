"""Repositorio local de estado, eventos, predicciones y checkpoints paper."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from .paper_files import append_jsonl, atomic_write_json


class JsonPaperRepository:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @property
    def state_path(self) -> Path:
        return self.directory / "state.json"

    @property
    def events_path(self) -> Path:
        return self.directory / "events.jsonl"

    @property
    def predictions_path(self) -> Path:
        return self.directory / "predictions.jsonl"

    @property
    def checkpoint_path(self) -> Path:
        return self.directory / "online_checkpoint.joblib"

    def load_state(self) -> dict | None:
        if not self.state_path.is_file():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save_state(self, state: dict) -> None:
        atomic_write_json(self.state_path, state)

    def append_event(self, event: dict) -> None:
        append_jsonl(self.events_path, event)

    def append_prediction(self, prediction: dict) -> None:
        append_jsonl(self.predictions_path, prediction)

    def clear_replay(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in (self.state_path, self.events_path, self.predictions_path, self.checkpoint_path):
            path.unlink(missing_ok=True)

    def write_predictions(self, predictions: list[dict]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = self.predictions_path.with_suffix(self.predictions_path.suffix + ".tmp")
        try:
            with temp.open("w", encoding="utf-8") as stream:
                for prediction in predictions:
                    stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            temp.replace(self.predictions_path)
        finally:
            temp.unlink(missing_ok=True)

    def load_checkpoint(self) -> dict | None:
        return joblib.load(self.checkpoint_path) if self.checkpoint_path.is_file() else None

    def save_checkpoint(self, checkpoint: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        try:
            joblib.dump(checkpoint, temp)
            temp.replace(self.checkpoint_path)
        finally:
            temp.unlink(missing_ok=True)

    def save_status(self, status: dict) -> None:
        atomic_write_json(self.directory / "live_status.json", status)

    def save_summary(self, summary: dict) -> None:
        atomic_write_json(self.directory / "live_summary_4h.json", summary)

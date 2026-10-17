from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path, max_lines: int = 500) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[-max_lines:]:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except Exception:
        pass
    return rows


def build_dashboard_payload(root: Path) -> dict:
    live_dir = root / "paper_futures_live"
    reports_dir = root / "reports"

    live_status = _read_json(live_dir / "live_status.json", {}) or {}
    live_preds = _read_jsonl(live_dir / "predictions.jsonl", max_lines=300)
    live_events = _read_jsonl(live_dir / "events.jsonl", max_lines=200)
    live_state = _read_json(live_dir / "state.json", {}) or {}
    evaluation = _read_json(reports_dir / "futures_long_short_evaluation.json", {}) or {}

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "version": "6.1",
        "active_model": {
            "name": evaluation.get("model", "USDⓈ-M LONG / FLAT / SHORT"),
            "feature_count": evaluation.get("feature_count", 21),
            "capacity": evaluation.get("model_capacity", {}),
            "training_pipeline_seconds": evaluation.get("training_pipeline_seconds"),
            "market": evaluation.get("market", "Binance USDⓈ-M BTCUSDT perpetual"),
            "policy": evaluation.get("policy", {}),
            "data": evaluation.get("data", {}),
            "validation_trading": evaluation.get("validation_trading", {}),
            "blind_test_trading": evaluation.get("blind_test_trading", {}),
            "walk_forward": evaluation.get("walk_forward", {}),
            "validation_classification": evaluation.get("validation_classification", {}),
            "blind_test_classification": evaluation.get("blind_test_classification", {}),
            "interpretation": evaluation.get("interpretation", ""),
        },
        "live": {
            "status": live_status,
            "state": live_state,
            "predictions": live_preds,
            "events": live_events,
        },
        "evaluation": evaluation,
    }


def write_live_js(root: Path) -> dict:
    payload = build_dashboard_payload(root)
    out_paths = [root / "paper_futures_live" / "live_data.js"]
    js_content = "window.TRADING_DASHBOARD_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n"
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".js.tmp")
        tmp.write_text(js_content, encoding="utf-8")
        os.replace(tmp, out_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Sincroniza USDⓈ-M paper data con el dashboard activo")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--watch", action="store_true", help="Actualizar continuamente cada N segundos")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--duration-sec", type=int, default=14500)
    args = parser.parse_args()

    start = time.monotonic()
    while True:
        write_live_js(args.root.resolve())
        if not args.watch or (time.monotonic() - start >= args.duration_sec):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

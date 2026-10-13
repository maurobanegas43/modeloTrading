from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import StrategyConfig
from .data import download_history, read_history
from .live import run_live
from .modeling import train_and_report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="binance-ml", description="Investigación ML de Binance Spot, SOLO PAPER. Nunca envía órdenes reales."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="Descargar velas 1m de Binance (datos públicos)")
    dl.add_argument("--days", type=int, default=90)
    dl.add_argument("--symbol", default="BTCUSDT")
    dl.add_argument("--out", type=Path, default=Path("data/BTCUSDT_1m.csv"))

    tr = sub.add_parser("train", help="Entrenar, validar y crear reporte de backtesting")
    tr.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1m.csv"))
    tr.add_argument("--model", type=Path, default=Path("models/bundle.joblib"))
    tr.add_argument("--report", type=Path, default=Path("reports/evaluation.json"))
    tr.add_argument("--symbol", default="BTCUSDT")
    tr.add_argument("--horizon", type=int, default=15)
    tr.add_argument("--fee", type=float, default=0.001)
    tr.add_argument("--execution-bps", type=float, default=5.0)

    live = sub.add_parser("paper-live", help="Leer Binance en vivo; aprender y operar SOLO saldo ficticio")
    live.add_argument("--model", type=Path, default=Path("models/bundle.joblib"))
    live.add_argument("--paper-dir", type=Path, default=Path("paper"))
    live.add_argument("--allow-unvalidated-paper", action="store_true",
                      help="Permitir compras VIRTUALES aunque el backtest no haya sido aprobado")

    rp = sub.add_parser("report", help="Mostrar las métricas más relevantes")
    rp.add_argument("--file", type=Path, default=Path("reports/evaluation.json"))

    args = parser.parse_args()
    if args.command == "download":
        cfg = StrategyConfig(symbol=args.symbol)
        cfg.validate()
        print(f"Descargando {args.days} días de {args.symbol}, velas de 1 minuto...")
        df = download_history(args.symbol, args.days)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"{len(df)} velas guardadas en {args.out}")
    elif args.command == "train":
        cfg = StrategyConfig(symbol=args.symbol, horizon_bars=args.horizon,
                             fee_rate=args.fee, execution_bps=args.execution_bps)
        df = read_history(args.csv)
        report = train_and_report(df, cfg, args.model, args.report)
        print(f"Modelo: {report['model']} | umbral {report['threshold']:.2f}")
        print(f"Validación: {report['validation_trading']['total_return_pct']:+.3f}% "
              f"({report['validation_trading']['trades']} operaciones)")
        print(f"TEST CIEGO: {report['test_trading']['total_return_pct']:+.3f}% "
              f"({report['test_trading']['trades']} operaciones)")
        print(f"Habilitado paper predeterminado: {report['approved_for_default_paper_signals']}")
        print(f"Reporte completo: {args.report} | Modelo: {args.model}")
    elif args.command == "paper-live":
        try:
            asyncio.run(run_live(args.model, args.paper_dir, args.allow_unvalidated_paper))
        except KeyboardInterrupt:
            print("Paper trading interrumpido por el usuario.")
    elif args.command == "report":
        data = json.loads(args.file.read_text(encoding="utf-8"))
        print(json.dumps({
            "model": data["model"], "threshold": data["threshold"],
            "validation": {k: v for k, v in data["validation_trading"].items() if k != "closed_trades"},
            "test": {k: v for k, v in data["test_trading"].items() if k != "closed_trades"},
            "approved_for_default_paper_signals": data["approved_for_default_paper_signals"],
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


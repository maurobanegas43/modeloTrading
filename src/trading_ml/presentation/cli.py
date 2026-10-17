from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from ..domain.config import FuturesConfig, StrategyConfig
from ..adapters.binance_market_data import BinanceMarketData
from ..adapters.binance_futures_stream import BinancePublicFuturesMarketStream
from ..adapters.futures_market_data import (
    attach_mark_prices,
    download_usdm_funding_history,
    download_usdm_history,
    download_usdm_mark_history,
    read_usdm_funding,
    read_usdm_history,
)
from ..adapters.model_store import FileBundleStore, FuturesBundleStore, JsonReportStore
from ..adapters.paper_repository import JsonPaperRepository
from ..application.paper_trading import run_replay_simulation
from ..application.modeling import run_walk_forward, train_model
from ..application.futures_modeling import evaluate_futures_bundle, train_futures_model
from ..application.futures_live_trading import run_live_futures


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="binance-ml",
        description="Investigación cuantitativa en Binance Spot y USDⓈ-M Futures, SOLO PAPER. Nunca envía órdenes reales.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Descargar velas 1m de Binance con caché incremental (datos públicos)")
    dl.add_argument("--days", type=int, default=90)
    dl.add_argument("--symbol", default="BTCUSDT")
    dl.add_argument("--out", type=Path, default=Path("data/BTCUSDT_1m.csv"))

    tr = sub.add_parser("train", help="Entrenar con purga temporal, validar y generar reporte fuera de muestra")
    tr.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1m.csv"))
    tr.add_argument("--model", type=Path, default=Path("models/bundle.joblib"))
    tr.add_argument("--report", type=Path, default=Path("reports/evaluation.json"))
    tr.add_argument("--symbol", default="BTCUSDT")
    tr.add_argument("--horizon", type=int, default=1440)
    tr.add_argument("--initial-stop", type=float, default=0.012)
    tr.add_argument("--trailing-activation", type=float, default=0.015)
    tr.add_argument("--trailing-delta", type=float, default=0.010)
    tr.add_argument("--fee", type=float, default=0.001)
    tr.add_argument("--execution-bps", type=float, default=3.0)
    tr.add_argument("--threshold", type=float, default=0.35)

    wf = sub.add_parser("walk-forward", help="Ejecutar validación Purged Walk-Forward de ventana expansiva")
    wf.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1m.csv"))
    wf.add_argument("--model", type=Path, default=Path("models/bundle.joblib"),
                    help="Usa el umbral del bundle vigente si no se informa --threshold")
    wf.add_argument("--folds", type=int, default=4)
    wf.add_argument("--out", type=Path, default=Path("reports/walk_forward.json"))
    wf.add_argument("--symbol", default="BTCUSDT")
    wf.add_argument("--threshold", type=float, default=None)

    sim = sub.add_parser("paper-sim", help="Simular paso a paso paper trading sobre velas 1m de Binance")
    sim.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_1m.csv"))
    sim.add_argument("--model", type=Path, default=Path("models/bundle.joblib"))
    sim.add_argument("--paper-dir", type=Path, default=Path("paper_balanced_oos"))
    sim.add_argument("--n-bars", type=int, default=None,
                     help="Número de velas recientes a simular (default: todas)")
    sim.add_argument("--oos-only", action="store_true",
                     help="Simular únicamente el período fuera de muestra (Validación + Test Ciego)")
    sim.add_argument("--allow-unvalidated-paper", action="store_true",
                     help="Permitir compras VIRTUALES aunque el backtest no haya sido aprobado")

    dl_futures = sub.add_parser("download-futures", help="Descargar velas, mark price y funding públicos de USDⓈ-M")
    dl_futures.add_argument("--days", type=int, default=90)
    dl_futures.add_argument("--symbol", default="BTCUSDT")
    dl_futures.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_USDM_1m.csv"))
    dl_futures.add_argument("--funding-csv", type=Path, default=Path("data/BTCUSDT_USDM_funding.csv"))
    dl_futures.add_argument("--mark-csv", type=Path, default=Path("data/BTCUSDT_USDM_mark_1m.csv"))

    train_fut = sub.add_parser("train-futures", help="Entrenar LONG/FLAT/SHORT con test temporal ciego y costes netos")
    train_fut.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_USDM_1m.csv"))
    train_fut.add_argument("--funding-csv", type=Path, default=Path("data/BTCUSDT_USDM_funding.csv"))
    train_fut.add_argument("--mark-csv", type=Path, default=Path("data/BTCUSDT_USDM_mark_1m.csv"))
    train_fut.add_argument("--model", type=Path, default=Path("models/futures_long_short.joblib"))
    train_fut.add_argument("--report", type=Path, default=Path("reports/futures_long_short_evaluation.json"))
    train_fut.add_argument("--symbol", default="BTCUSDT")
    train_fut.add_argument("--fee", type=float, default=0.0005)
    train_fut.add_argument("--execution-bps", type=float, default=3.0)
    train_fut.add_argument("--leverage", type=float, default=1.0)
    train_fut.add_argument("--min-net-edge-bps", type=float, default=5.0)
    train_fut.add_argument("--probability-threshold", type=float, default=0.40)

    paper_fut = sub.add_parser("paper-futures-sim", help="Replay paper solo con velas posteriores al entrenamiento del bundle")
    paper_fut.add_argument("--csv", type=Path, default=Path("data/BTCUSDT_USDM_1m.csv"))
    paper_fut.add_argument("--funding-csv", type=Path, default=Path("data/BTCUSDT_USDM_funding.csv"))
    paper_fut.add_argument("--mark-csv", type=Path, default=Path("data/BTCUSDT_USDM_mark_1m.csv"))
    paper_fut.add_argument("--model", type=Path, default=Path("models/futures_long_short.joblib"))

    paper_fut_live = sub.add_parser(
        "paper-futures-live",
        help="Simulación USDⓈ-M en vivo con decisiones 15m y contexto 1h (solo paper)",
    )
    paper_fut_live.add_argument("--model", type=Path, default=Path("models/futures_long_short.joblib"))
    paper_fut_live.add_argument("--paper-dir", type=Path, default=Path("paper_futures_live"))
    paper_fut_live.add_argument("--history-csv", type=Path, default=Path("data/BTCUSDT_USDM_1m.csv"))
    paper_fut_live.add_argument("--funding-csv", type=Path, default=Path("data/BTCUSDT_USDM_funding.csv"))
    paper_fut_live.add_argument("--mark-csv", type=Path, default=Path("data/BTCUSDT_USDM_mark_1m.csv"))
    paper_fut_live.add_argument("--duration-sec", type=int, default=14_400)

    rp = sub.add_parser("report", help="Mostrar las métricas más relevantes")
    rp.add_argument("--file", type=Path, default=Path("reports/evaluation.json"))

    args = parser.parse_args()
    market_data = BinanceMarketData()
    if args.command == "download":
        cfg = StrategyConfig(symbol=args.symbol)
        cfg.validate()
        print(f"Descargando {args.days} días de {args.symbol}, velas de 1 minuto...")
        df = market_data.download_history(args.symbol, args.days, existing_path=args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"{len(df):,} velas guardadas en {args.out}")
    elif args.command == "download-futures":
        print(f"Descargando {args.days} días de {args.symbol} USDⓈ-M (perpetual)...", flush=True)
        candles = download_usdm_history(args.symbol, args.days)
        first_ms = int(candles["open_time"].iloc[0])
        last_ms = int(candles["open_time"].iloc[-1]) + 60_000
        funding = download_usdm_funding_history(args.symbol, first_ms, last_ms)
        mark = download_usdm_mark_history(args.symbol, args.days)
        for path in (args.csv, args.funding_csv, args.mark_csv):
            path.parent.mkdir(parents=True, exist_ok=True)
        candles.to_csv(args.csv, index=False)
        funding.to_csv(args.funding_csv, index=False)
        mark.to_csv(args.mark_csv, index=False)
        print(f"Velas: {len(candles):,} → {args.csv}")
        print(f"Funding: {len(funding):,} pagos → {args.funding_csv}")
        print(f"Mark price: {len(mark):,} velas → {args.mark_csv}")
    elif args.command == "train-futures":
        cfg = FuturesConfig(
            symbol=args.symbol,
            fee_rate=args.fee,
            execution_bps=args.execution_bps,
            leverage=args.leverage,
            min_net_edge_bps=args.min_net_edge_bps,
            probability_threshold=args.probability_threshold,
        )
        cfg.validate()
        candles = attach_mark_prices(read_usdm_history(args.csv), args.mark_csv if args.mark_csv.exists() else None)
        funding = read_usdm_funding(args.funding_csv if args.funding_csv.exists() else None)
        bundle, report = train_futures_model(candles, funding, cfg)
        FuturesBundleStore(args.model).save(bundle)
        JsonReportStore(args.report).save(report)
        print("\n============= ENTRENAMIENTO BTCUSDT USDⓈ-M =============")
        print(f"Muestras de decisión ({cfg.decision_interval_minutes}m): {report['data']['labeled_decision_samples']:,} | etiquetas S/F/L: {report['data']['label_counts_short_flat_long']}")
        print(f"Validación: {report['validation_trading']['trades']} operaciones | retorno {report['validation_trading']['total_return_pct']:+.3f}% | DD {report['validation_trading']['max_drawdown_pct']:.2f}% | acciones {report['validation_trading']['action_counts']}")
        print(f"  Clasificación: accuracy {report['validation_classification']['accuracy']:.3f} vs baseline FLAT {report['validation_classification']['always_flat_baseline_accuracy']:.3f}")
        print(f"TEST CIEGO: {report['blind_test_trading']['trades']} operaciones | retorno {report['blind_test_trading']['total_return_pct']:+.3f}% | DD {report['blind_test_trading']['max_drawdown_pct']:.2f}% | acciones {report['blind_test_trading']['action_counts']}")
        print(f"  Clasificación: accuracy {report['blind_test_classification']['accuracy']:.3f} vs baseline FLAT {report['blind_test_classification']['always_flat_baseline_accuracy']:.3f}")
        print(f"Walk-forward: {report['walk_forward']['total_oos_trades']} operaciones OOS | retorno compuesto {report['walk_forward']['oos_compounded_return_pct']:+.3f}%")
        print(f"Tiempo entrenamiento + validaciones: {report['training_pipeline_seconds']:.2f}s")
        print(f"Artefactos: {args.model} | {args.report}")
        print("El bundle es solo paper; las métricas no garantizan rentabilidad futura.")
        print("=========================================================\n")
    elif args.command == "paper-futures-sim":
        candles = attach_mark_prices(read_usdm_history(args.csv), args.mark_csv if args.mark_csv.exists() else None)
        funding = read_usdm_funding(args.funding_csv if args.funding_csv.exists() else None)
        bundle = FuturesBundleStore(args.model).load()
        print(json.dumps(evaluate_futures_bundle(bundle, candles, funding), indent=2, ensure_ascii=False))
    elif args.command == "paper-futures-live":
        bundle = FuturesBundleStore(args.model).load()
        cfg = FuturesConfig(**bundle["config"])

        async def start_futures_live() -> None:
            local = read_usdm_history(args.history_csv)
            try:
                recent = await asyncio.to_thread(download_usdm_history, cfg.symbol, 5)
                common = [column for column in local.columns if column in recent.columns]
                candles = (
                    pd.concat([local[common], recent[common]], ignore_index=True)
                    .sort_values("open_time")
                    .drop_duplicates("open_time", keep="last")
                    .iloc[-6500:]
                    .reset_index(drop=True)
                )
                print(f"Bootstrap USDⓈ-M sincronizado con REST: {len(candles):,} velas.", flush=True)
            except Exception as exc:
                candles = local.iloc[-6500:].copy().reset_index(drop=True)
                print(f"REST no disponible ({exc}); usando histórico local: {len(candles):,} velas.", flush=True)
            candles = attach_mark_prices(candles, args.mark_csv if args.mark_csv.exists() else None)
            funding = read_usdm_funding(args.funding_csv if args.funding_csv.exists() else None)
            await run_live_futures(
                bundle,
                JsonPaperRepository(args.paper_dir),
                BinancePublicFuturesMarketStream(),
                candles,
                funding,
                args.duration_sec,
            )

        try:
            asyncio.run(start_futures_live())
        except KeyboardInterrupt:
            print("Simulación futures paper interrumpida.")
    elif args.command == "train":
        cfg = StrategyConfig(
            symbol=args.symbol,
            horizon_bars=args.horizon,
            initial_stop=args.initial_stop,
            trailing_activation=args.trailing_activation,
            trailing_delta=args.trailing_delta,
            fee_rate=args.fee,
            execution_bps=args.execution_bps,
            threshold=args.threshold,
        )
        df = market_data.read_history(args.csv)
        bundle, report = train_model(df, cfg)
        FileBundleStore(args.model).save(bundle)
        JsonReportStore(args.report).save(report)
        print("\n==================== RESUMEN DE ENTRENAMIENTO v5.0 ====================")
        print(f"Estrategia: {report['strategy']}")
        print(f"Modelo:     {report['model']} | Umbral Óptimo (Val): {report['threshold']:.2f}")
        tr_bt = report["train_trading"]
        print(f"Train (70%):       Retorno {tr_bt['total_return_pct']:+.3f}% | Ops: {tr_bt['trades']} | WinRate: {tr_bt['win_rate']*100:.1f}% | PF: {tr_bt['profit_factor']} | MaxDD: {tr_bt['max_drawdown_pct']:.2f}%")
        val_bt = report["validation_trading"]
        print(f"Validación (15%):  Retorno {val_bt['total_return_pct']:+.3f}% | Ops: {val_bt['trades']} | WinRate: {val_bt['win_rate']*100:.1f}% | PF: {val_bt['profit_factor']} | MaxDD: {val_bt['max_drawdown_pct']:.2f}%")
        test_bt = report["test_trading"]
        print(f"TEST CIEGO (15%):  Retorno {test_bt['total_return_pct']:+.3f}% | Ops: {test_bt['trades']} | WinRate: {test_bt['win_rate']*100:.1f}% | PF: {test_bt['profit_factor']} | MaxDD: {test_bt['max_drawdown_pct']:.2f}%")
        oos_bt = report["out_of_sample_combined_trading"]
        print(f"TOTAL OOS (30%):   Retorno {oos_bt['total_return_pct']:+.3f}% | Ops: {oos_bt['trades']} | WinRate: {oos_bt['win_rate']*100:.1f}% | PF: {oos_bt['profit_factor']} | Sharpe: {oos_bt['sharpe_ratio']:.2f}")
        full_bt = report["full_period_trading"]
        print(f"PERÍODO COMPLETO:  Retorno {full_bt['total_return_pct']:+.3f}% | Ops: {full_bt['trades']} | WinRate: {full_bt['win_rate']*100:.1f}% | PF: {full_bt['profit_factor']} | Sharpe: {full_bt['sharpe_ratio']:.2f}")
        wf = report["walk_forward_summary"]
        print(f"Walk-Forward (4f): Retorno OOS {wf['oos_compounded_return_pct']:+.3f}% | Ops: {wf['total_oos_trades']} | WinRate: {wf['oos_win_rate']*100:.1f}%")
        print(f"Habilitado paper trading predeterminado (OOS Gate): {report['approved_for_default_paper_signals']}")
        print(f"Reporte guardado: {args.report} | Modelo guardado: {args.model}")
        print("=======================================================================\n")
    elif args.command == "walk-forward":
        threshold = args.threshold
        if threshold is None:
            try:
                threshold = FileBundleStore(args.model).load()["threshold"]
            except FileNotFoundError:
                threshold = StrategyConfig().threshold
        cfg = StrategyConfig(symbol=args.symbol, threshold=threshold)
        df = market_data.read_history(args.csv)
        summary = run_walk_forward(df, cfg, n_folds=args.folds)
        JsonReportStore(args.out).save(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.command == "paper-sim":
        run_replay_simulation(
            FileBundleStore(args.model).load(),
            market_data.read_history(args.csv),
            JsonPaperRepository(args.paper_dir),
            args.n_bars,
            args.allow_unvalidated_paper,
            args.oos_only,
        )
    elif args.command == "report":
        data = json.loads(args.file.read_text(encoding="utf-8"))
        out = {
            "model": data["model"],
            "threshold": data["threshold"],
            "strategy": data.get("strategy"),
            "validation": {k: v for k, v in data["validation_trading"].items() if k != "closed_trades"},
            "test": {k: v for k, v in data["test_trading"].items() if k != "closed_trades"},
            "out_of_sample_combined": {
                k: v for k, v in data.get("out_of_sample_combined_trading", {}).items() if k != "closed_trades"
            },
            "full_period": {k: v for k, v in data.get("full_period_trading", {}).items() if k != "closed_trades"},
            "walk_forward_summary": data.get("walk_forward_summary"),
            "approved_for_default_paper_signals": data["approved_for_default_paper_signals"],
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

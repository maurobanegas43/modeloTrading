"""Fachada compatible para crear los adaptadores de paper trading."""

from __future__ import annotations

from pathlib import Path

from .adapters.binance_market_data import BinanceMarketData
from .adapters.binance_stream import BinancePublicMarketStream
from .adapters.model_store import FileBundleStore
from .adapters.paper_files import FileBackedPaperWallet as VirtualWallet
from .adapters.paper_repository import JsonPaperRepository
from .application.paper_trading import run_live as _run_live
from .application.paper_trading import run_replay_simulation as _run_replay


def run_replay_simulation(
    model_path: Path,
    paper_dir: Path,
    csv_path: Path,
    n_bars: int | None = None,
    allow_unvalidated_paper: bool = False,
    oos_only: bool = False,
) -> dict:
    return _run_replay(
        FileBundleStore(model_path).load(),
        BinanceMarketData().read_history(csv_path),
        JsonPaperRepository(paper_dir),
        n_bars,
        allow_unvalidated_paper,
        oos_only,
    )


async def run_live(
    model_path: Path,
    paper_dir: Path,
    allow_unvalidated_paper: bool = False,
    max_duration_sec: int | None = None,
) -> None:
    store = FileBundleStore(model_path)
    bundle = store.load()
    symbol = bundle["config"]["symbol"]
    await _run_live(
        store,
        JsonPaperRepository(paper_dir),
        BinanceMarketData(),
        BinancePublicMarketStream(),
        Path("data") / f"{symbol}_1m.csv",
        allow_unvalidated_paper,
        max_duration_sec,
    )


__all__ = ["VirtualWallet", "run_live", "run_replay_simulation"]

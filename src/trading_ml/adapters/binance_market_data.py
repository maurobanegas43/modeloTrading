"""Adaptador de velas públicas de Binance y del histórico CSV local."""

from pathlib import Path

import pandas as pd

from .market_data import download_history, read_history, recent_closed_candles


class BinanceMarketData:
    download_history = staticmethod(download_history)
    read_history = staticmethod(read_history)
    recent_closed_candles = staticmethod(recent_closed_candles)

    def read_recent_local(self, path: Path, bars: int) -> pd.DataFrame:
        return self.read_history(path).iloc[-bars:].copy()

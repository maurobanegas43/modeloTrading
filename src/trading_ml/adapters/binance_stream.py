"""Adaptador del stream público bookTicker/kline de Binance."""

from __future__ import annotations

import json
from typing import AsyncIterator

from websockets.asyncio.client import connect


class BinancePublicMarketStream:
    async def events(self, symbol: str) -> AsyncIterator[dict]:
        uri = (
            "wss://stream.binance.com:9443/stream?streams="
            f"{symbol.lower()}@bookTicker/{symbol.lower()}@kline_1m"
        )
        async with connect(uri, ping_interval=30, ping_timeout=30, close_timeout=10, max_size=2**20) as websocket:
            async for message in websocket:
                yield json.loads(message)

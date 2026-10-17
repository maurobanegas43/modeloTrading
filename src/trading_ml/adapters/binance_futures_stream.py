"""Streams públicos USDⓈ-M para libro, velas de 1m y mark price/funding."""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

from websockets.asyncio.client import connect


BASE_URL = "wss://fstream.binance.com"
BOOK_TICKER_INTERVAL_SEC = 0.2


class BinancePublicFuturesMarketStream:
    async def _pump(self, uri: str, queue: asyncio.Queue, coalesce_book_ticker: bool = False) -> None:
        last_book_ticker = 0.0
        async with connect(uri, ping_interval=30, ping_timeout=30, close_timeout=10, max_size=2**20) as websocket:
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                except asyncio.TimeoutError:
                    # The outer iterator emits the heartbeat and checks each feed.
                    continue
                event = json.loads(message)
                payload = event.get("data", event)
                if coalesce_book_ticker and payload.get("e") == "bookTicker":
                    now = time.monotonic()
                    if now - last_book_ticker < BOOK_TICKER_INTERVAL_SEC:
                        continue
                    last_book_ticker = now
                await queue.put(event)

    async def events(self, symbol: str) -> AsyncIterator[dict]:
        normalized = symbol.lower()
        public_uri = f"{BASE_URL}/public/stream?streams={normalized}@bookTicker"
        market_uri = (
            f"{BASE_URL}/market/stream?streams="
            f"{normalized}@kline_1m/{normalized}@markPrice@1s"
        )
        queue: asyncio.Queue = asyncio.Queue()
        feeds = [
            asyncio.create_task(self._pump(public_uri, queue, coalesce_book_ticker=True)),
            asyncio.create_task(self._pump(market_uri, queue)),
        ]
        try:
            while True:
                item = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    [item, *feeds], timeout=10.0, return_when=asyncio.FIRST_COMPLETED
                )
                failed_feed = next((feed for feed in feeds if feed in done), None)
                if failed_feed is not None:
                    item.cancel()
                    await asyncio.gather(item, return_exceptions=True)
                    exception = failed_feed.exception()
                    if exception is not None:
                        raise exception
                    raise ConnectionError("stream USDⓈ-M finalizado")
                if item in done:
                    yield item.result()
                else:
                    item.cancel()
                    await asyncio.gather(item, return_exceptions=True)
                    yield {"_heartbeat": True}
        finally:
            for feed in feeds:
                feed.cancel()
            await asyncio.gather(*feeds, return_exceptions=True)

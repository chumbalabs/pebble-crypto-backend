# WebSocket handlers

from fastapi import APIRouter, WebSocket, HTTPException
from datetime import datetime, timezone
import asyncio
from app.services.binance import BinanceClient

router = APIRouter()

ALLOWED_INTERVALS = ["1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]

binance = BinanceClient()

@router.websocket("/ws/realtime/{symbol}")
async def websocket_realtime(websocket: WebSocket, symbol: str, interval: str = "1h"):
    await websocket.accept()
    try:
        if interval not in ALLOWED_INTERVALS:
            await websocket.send_json({
                "error": "Invalid interval",
                "detail": f"Allowed values: {', '.join(ALLOWED_INTERVALS)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            await websocket.close(code=1008)
            return
        while True:
            ohlcv = await binance.fetch_ohlcv(symbol, interval, limit=1)
            if ohlcv:
                await websocket.send_json({
                    "symbol": symbol,
                    "interval": interval,
                    "data": ohlcv[0],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            if interval.endswith('h'):
                sleep_time = 300
            elif interval in ['1d', '3d', '1w', '1M']:
                sleep_time = 900
            else:
                sleep_time = 300
            await asyncio.sleep(sleep_time)
    except Exception as e:
        await websocket.close(code=1011)

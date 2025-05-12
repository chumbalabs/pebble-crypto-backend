# Prediction endpoints

from fastapi import APIRouter, HTTPException, Request
from datetime import datetime, timezone
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.prediction.technical import predictor
from app.services.binance import BinanceClient

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
binance = BinanceClient()

ALLOWED_INTERVALS = ["1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]

@router.get("/predict/{symbol}", tags=["Predictions"])
@limiter.limit("30/minute")
async def predict_price(request: Request, symbol: str, interval: str = "1h"):
    try:
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Allowed values: {', '.join(ALLOWED_INTERVALS)}"
            )
        predictor.analysis_cache.clear()
        ohlcv = await binance.fetch_ohlcv(symbol, interval, limit=100)
        closes = [entry["close"] for entry in ohlcv]
        if len(closes) < 50:
            raise HTTPException(
                status_code=422,
                detail="Need at least 50 data points for analysis"
            )
        analysis = await predictor.analyze_market(closes, interval)
        analysis["metadata"]["interval"] = interval
        analysis["metadata"]["symbol"] = symbol
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Analysis failed: " + str(e)
        )

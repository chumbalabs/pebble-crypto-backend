from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Request, WebSocket, Depends, Body
from datetime import datetime, timezone
import os
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, List, Dict, Any
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from pydantic import BaseModel, Field
from app.services.binance import BinanceClient, SYMBOLS_CACHE, TICKER_CACHE
from app.core.prediction.technical import predictor
from app.services.metrics import MetricsTracker
from app.core.analysis.market_advisor import MarketAdvisor, MarketComparisonAnalyzer
from app.core.ai.agent import MarketAgent

ALLOWED_INTERVALS = ["1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
INTERVAL_HOURS = {
    "1h": 1, 
    "2h": 2, 
    "4h": 4, 
    "6h": 6, 
    "8h": 8, 
    "12h": 12, 
    "1d": 24, 
    "3d": 72, 
    "1w": 168, 
    "1M": 720
}
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CoolifyCryptoAPI")
metrics = MetricsTracker()
binance = BinanceClient()
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="pebble-crypto-api",
    description="Advanced Crypto Analytics & Predictions",
    version="0.3.0",
    docs_url="/docs",
    redoc_url=None
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class LoggingMiddleware:
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        logger.info(f"{request.method} {request.url} - Status: {response.status_code}")
        return response
app.middleware("http")(LoggingMiddleware())

# Endpoints
@app.get("/api/health", tags=["Health"])
@limiter.limit("100/minute")
async def health_check(request: Request):
    return {
        "name": "CryptoPredict Pro+",
        "status": "online",
        "version": "0.3.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/symbols", tags=["Market Data"])
@limiter.limit("30/minute")
async def get_active_symbols(request: Request, sort_by: Optional[str] = None, descending: bool = True):
    try:
        cache_key = "symbols"
        if not SYMBOLS_CACHE.get(cache_key):
            SYMBOLS_CACHE[cache_key] = binance.fetch_symbols()
        symbols = SYMBOLS_CACHE[cache_key]

        if sort_by == "volume":
            sort_cache_key = f"symbols_sorted_volume"
            if not SYMBOLS_CACHE.get(sort_cache_key):
                tickers = binance.fetch_tickers()
                ticker_map = {t['symbol']: t for t in tickers}
                sorted_symbols = sorted(
                    symbols,
                    key=lambda s: float(ticker_map.get(s, {}).get('quoteVolume', 0)),
                    reverse=descending
                )
                SYMBOLS_CACHE[sort_cache_key] = sorted_symbols
            return {
                "symbols": SYMBOLS_CACHE[sort_cache_key],
                "sorting": f"24h_quote_volume_{'desc' if descending else 'asc'}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        return {"symbols": symbols, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"Symbols error: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.websocket("/ws/realtime/{symbol}")
async def websocket_realtime(websocket: WebSocket, symbol: str, interval: str = "1h"):
    await websocket.accept()
    try:
        # Validate interval
        if interval not in ALLOWED_INTERVALS:
            await websocket.send_json({
                "error": "Invalid interval",
                "detail": f"Allowed values: {', '.join(ALLOWED_INTERVALS)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            await websocket.close(code=1008)
            return
            
        while True:
            # Await the async fetch_ohlcv call here
            ohlcv = await binance.fetch_ohlcv(symbol, interval, limit=1)
            if ohlcv:
                await websocket.send_json({
                    "symbol": symbol,
                    "interval": interval,
                    "data": ohlcv[0],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Adjust sleep time based on interval
            if interval.endswith('h'):
                # For hour intervals, sleep for 5 minutes
                sleep_time = 300
            elif interval in ['1d', '3d', '1w', '1M']:
                # For day/week/month intervals, sleep for 15 minutes
                sleep_time = 900
            else:
                sleep_time = 300
                
            await asyncio.sleep(sleep_time)
    except Exception as e:
        logger.error(f"WebSocket error ({symbol}): {str(e)}")
        await websocket.close(code=1011)

@app.get("/intraday/{symbol}", tags=["Market Data"])
@limiter.limit("30/minute")
async def get_intraday_data(request: Request, symbol: str, interval: str = "1h"):
    """
    Returns intraday data for the given symbol based on the specified interval for the current day.
    Data points are fetched from midnight (UTC) until the current time.
    """
    try:
        # Validate interval
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Allowed values: {', '.join(ALLOWED_INTERVALS)}"
            )
            
        now = datetime.now(timezone.utc)
        # Determine the start of the current day in UTC
        start_of_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        
        # Calculate how many intervals have elapsed since the start of the day
        # For intervals less than 1 hour, we need more data points
        interval_hours = INTERVAL_HOURS[interval]
        intervals_elapsed = int((now - start_of_day).total_seconds() / (interval_hours * 3600)) + 1
        
        # Limit to a reasonable number of candles
        limit = min(intervals_elapsed, 500)
        
        # Await the async call for OHLCV data
        data = await binance.fetch_ohlcv(symbol, interval, limit=limit)
        if not data:
            raise HTTPException(status_code=404, detail="No intraday data available")

        # Filter candles to ensure they are within the current day (if necessary)
        intraday = []
        for candle in data:
            candle_time = datetime.fromtimestamp(candle["timestamp"] / 1000, tz=timezone.utc)
            if candle_time >= start_of_day:
                intraday.append(candle)

        return {
            "symbol": symbol,
            "interval": interval,
            "intraday_data": intraday,
            "time_updated": now.isoformat(),
            "intervals_elapsed": intervals_elapsed,
            "candles_returned": len(intraday)
        }
    except Exception as e:
        logger.error(f"Intraday data error for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Intraday data retrieval failed")

@app.get("/predict/{symbol}", tags=["Predictions"])
@limiter.limit("30/minute")
async def predict_price(request: Request, symbol: str, interval: str = "1h"):
    try:
        # Debug logging for interval
        logger.info(f"Predict endpoint called with interval: {interval}")
        
        # Validate interval
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Allowed values: {', '.join(ALLOWED_INTERVALS)}"
            )
            
        # Clear the cache to ensure fresh analysis
        predictor.analysis_cache.clear()
        logger.info(f"Cache cleared for fresh analysis with interval: {interval}")
            
        # Await the async call for OHLCV data
        ohlcv = await binance.fetch_ohlcv(symbol, interval, limit=100)
        closes = [entry["close"] for entry in ohlcv]
        
        if len(closes) < 50:
            raise HTTPException(
                status_code=422,
                detail="Need at least 50 data points for analysis"
            )
            
        # Await the async analyze_market call
        analysis = await predictor.analyze_market(closes, interval)
        
        # Force the interval in the response
        analysis["metadata"]["interval"] = interval
        analysis["metadata"]["symbol"] = symbol
        
        return analysis
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed: " + str(e)
        )

@app.get("/historical/{symbol}", tags=["Market Data"])
@limiter.limit("20/minute")
async def get_historical_data(
    request: Request, 
    symbol: str, 
    interval: str = "1h", 
    limit: int = 100
):
    """
    Returns historical data for the given symbol and interval.
    Allows specifying the number of candles to retrieve.
    """
    try:
        # Validate interval
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Allowed values: {', '.join(ALLOWED_INTERVALS)}"
            )
            
        # Validate limit
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 1000"
            )
            
        # Await the async call for OHLCV data
        data = await binance.fetch_ohlcv(symbol, interval, limit=limit)
        if not data:
            raise HTTPException(status_code=404, detail="No historical data available")

        return {
            "symbol": symbol,
            "interval": interval,
            "historical_data": data,
            "time_updated": datetime.now(timezone.utc).isoformat(),
            "candles_returned": len(data)
        }
    except Exception as e:
        logger.error(f"Historical data error for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Historical data retrieval failed")

@app.get("/investment-advice/{symbol}", tags=["Analysis"])
@limiter.limit("20/minute")
async def get_investment_advice(
    request: Request, 
    symbol: str, 
    interval: str = "1h"
):
    """
    Returns detailed buy/sell advice with entry and exit points for the given symbol.
    """
    try:
        # Validate interval
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Allowed values: {', '.join(ALLOWED_INTERVALS)}"
            )
            
        # Fetch OHLCV data
        ohlcv = await binance.fetch_ohlcv(symbol, interval, limit=100)
        if not ohlcv:
            raise HTTPException(status_code=404, detail="No data available for this symbol")
            
        # Calculate technical indicators
        closes = [entry["close"] for entry in ohlcv]
        highs = [entry["high"] for entry in ohlcv]
        lows = [entry["low"] for entry in ohlcv]
        
        # Initialize technical data dictionary
        technical_data = {}
        
        # Calculate moving averages
        sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else sum(closes) / len(closes)
        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sum(closes) / len(closes)
        technical_data["sma_20"] = sma_20
        technical_data["sma_50"] = sma_50
        
        # Calculate RSI
        try:
            rsi = predictor._calculate_rsi()
            technical_data["rsi"] = rsi
        except Exception as e:
            logger.error(f"RSI calculation error: {str(e)}")
            technical_data["rsi"] = 50.0  # Neutral RSI fallback
            
        # Calculate Bollinger Bands
        from app.core.indicators.advanced import BollingerBands
        bb_indicator = BollingerBands()
        bb_data = bb_indicator.calculate(closes)
        bb_signal = bb_indicator.get_signal(closes)
        technical_data["bollinger_bands"] = {
            **bb_data,
            "signal": bb_signal
        }
        
        # Calculate ATR
        from app.core.indicators.advanced import AverageTrueRange
        atr_indicator = AverageTrueRange()
        atr_data = atr_indicator.calculate(highs, lows, closes)
        atr_signal = atr_indicator.get_signal(highs, lows, closes)
        atr_data = {
            **atr_data,
            "signal": atr_signal
        }
        
        # Prepare price data
        price_data = {
            "current_price": closes[-1],
            "price_change_24h": (closes[-1] - closes[-24]) / closes[-24] if len(closes) >= 24 else 0
        }
        
        # Generate advice
        advisor = MarketAdvisor()
        advice = advisor.generate_trading_advice(
            technical_data=technical_data,
            price_data=price_data,
            atr_data=atr_data
        )
        
        return {
            "symbol": symbol,
            "interval": interval,
            "advice": advice,
            "time_updated": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Investment advice error for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/compare-assets/{primary_symbol}", tags=["Analysis"])
@limiter.limit("15/minute")
async def compare_assets(
    request: Request, 
    primary_symbol: str, 
    comparison_symbols: str,
    time_period: str = "7d"
):
    """
    Compares a primary cryptocurrency with multiple other assets over a specified time period.
    Returns performance metrics and relative rankings.
    
    Parameters:
    - primary_symbol: Main cryptocurrency to analyze (e.g., "BTCUSDT")
    - comparison_symbols: Comma-separated list of symbols to compare against (e.g., "ETHUSDT,SOLUSDT,BNBUSDT")
    - time_period: Time period for comparison ("1d", "3d", "7d", "14d", "30d")
    """
    try:
        # Validate symbols format
        if not primary_symbol or not comparison_symbols:
            raise HTTPException(
                status_code=400,
                detail="Both primary_symbol and comparison_symbols must be provided"
            )
            
        # Parse comparison symbols
        comparison_assets = comparison_symbols.split(",")
        
        # Limit number of comparisons
        if len(comparison_assets) > 10:
            comparison_assets = comparison_assets[:10]
            logger.warning("Too many comparison assets requested. Limited to 10.")
            
        # Validate time period
        valid_periods = ["1d", "3d", "7d", "14d", "30d"]
        if time_period not in valid_periods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time period. Allowed values: {', '.join(valid_periods)}"
            )
            
        # Initialize the comparison analyzer
        analyzer = MarketComparisonAnalyzer(binance_client=binance)
        
        # Generate comparison data
        comparison_data = await analyzer.compare_assets(
            primary_symbol=primary_symbol,
            comparison_assets=comparison_assets,
            time_period=time_period
        )
        
        if "error" in comparison_data:
            raise HTTPException(
                status_code=404,
                detail=comparison_data["error"]
            )
            
        return {
            "comparison_data": comparison_data,
            "time_updated": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Asset comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

# Pydantic models for AI query endpoint
class CryptoQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about cryptocurrency markets")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context to enhance the AI response")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the current price of Bitcoin and should I buy it now?",
                "context": {"preferred_timeframe": "1d", "risk_tolerance": "moderate"}
            }
        }

class CryptoQueryResponse(BaseModel):
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="AI-generated response to the query")
    timestamp: str = Field(..., description="When the response was generated")
    supporting_data: Optional[Dict[str, Any]] = Field(default=None, description="Supporting data used to generate the response")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata about the query processing")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the current price of Bitcoin and should I buy it now?",
                "response": "Bitcoin (BTCUSDT) is currently trading at $50,123.45, up 2.3% in the last 24 hours. Technical indicators show a bullish trend with RSI at 58. Based on current volatility and market conditions, consider dollar-cost averaging rather than a single large purchase. Always do your own research and consider your risk tolerance before investing.",
                "timestamp": "2024-10-08T12:34:56.789Z",
                "supporting_data": {
                    "current_price": 50123.45,
                    "price_change_24h": 0.023,
                    "rsi": 58
                },
                "metadata": {
                    "symbol": "BTCUSDT",
                    "interval": "1d",
                    "data_sources": ["price_data", "technical_indicators", "ai_insights"]
                }
            }
        }

# Initialize the MarketAgent for AI-powered responses
market_agent = MarketAgent()

@app.post("/api/ask", response_model=CryptoQueryResponse, tags=["AI Assistant"])
@limiter.limit("60/minute")
async def process_crypto_query(
    request: Request,
    query_request: CryptoQueryRequest = Body(...),
):
    """
    Process a natural language query about cryptocurrency markets and return an AI-powered response.
    
    This endpoint can handle a wide variety of questions including but not limited to:
    - Price information (e.g., "What's the current price of ETH?")
    - Trend analysis (e.g., "What's the trend for Solana?")
    - Volatility assessment (e.g., "How volatile is LINK today?")
    - Investment advice (e.g., "Should I buy SOL now?")
    - Technical indicators (e.g., "What does the RSI say about Bitcoin?")
    - Market comparison (e.g., "How is ADA performing compared to DOT?")
    
    The AI will analyze relevant data sources and provide a comprehensive response with supporting information.
    """
    try:
        # Extract query and context
        query = query_request.query
        context = query_request.context or {}
        
        # Log the incoming query
        logger.info(f"Processing query: {query}")
        
        # Process query through the market agent
        result = await market_agent.process_query(query)
        
        # Apply any context-specific adjustments to the response
        if context:
            result = await _enhance_with_context(result, context)
        
        # Extract supporting data with multi-timeframe insights
        supporting_data = result.get("supporting_data", {})
        
        # Add multi-timeframe data if available
        if "multi_timeframe" in result:
            # Only include essential data from multi-timeframe analysis
            multi_tf_insights = {}
            for tf, tf_data in result.get("multi_timeframe", {}).items():
                if "trend" in tf_data:
                    multi_tf_insights[tf] = {
                        "trend": tf_data["trend"].get("description", "NEUTRAL"),
                        "rsi": tf_data.get("indicators", {}).get("rsi", 50),
                        "volatility": tf_data.get("volatility")
                    }
            
            # Add to supporting data if we have insights
            if multi_tf_insights:
                supporting_data["timeframe_analysis"] = multi_tf_insights
        
        # Return the structured response
        return CryptoQueryResponse(
            query=result["query"],
            response=result["response"],
            timestamp=result["timestamp"],
            supporting_data=supporting_data,
            metadata=result.get("metadata")
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process query: {str(e)}"
        )

async def _enhance_with_context(result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Apply context-specific adjustments to the AI response"""
    # Handle preferred timeframe if provided
    if "preferred_timeframe" in context and result.get("metadata", {}).get("symbol"):
        symbol = result["metadata"]["symbol"]
        timeframe = context["preferred_timeframe"]
        
        # Map common timeframe formats to API intervals
        timeframe_map = {
            "short": "1h",
            "medium": "4h",
            "long": "1d",
            "very_long": "1w"
        }
        interval = timeframe_map.get(timeframe, timeframe)
        
        # Only adjust if the interval is valid
        if interval in ALLOWED_INTERVALS:
            # Add timeframe-specific context to the response
            if "I've analyzed" not in result["response"]:
                result["response"] = f"I've analyzed {symbol} using {interval} timeframe data. {result['response']}"
    
    # Handle risk tolerance if provided
    if "risk_tolerance" in context:
        risk_tolerance = context["risk_tolerance"].lower()
        if "advice" in result.get("metadata", {}).get("intent", ""):
            # Add risk tolerance context if it's an advice-seeking query
            if risk_tolerance == "high":
                result["response"] += "\n\nNote: Based on your high risk tolerance, you might consider more aggressive entry/exit points than suggested above."
            elif risk_tolerance == "low":
                result["response"] += "\n\nNote: Given your low risk tolerance, consider using tighter stop losses and taking smaller positions than suggested above."
    
    return result

# Multi-Exchange Endpoints
@app.get("/api/exchanges/health", tags=["Multi-Exchange"])
@limiter.limit("30/minute")
async def get_exchange_health(request: Request):
    """
    Get health status of all registered cryptocurrency exchanges.
    
    Returns information about:
    - Exchange availability and response times
    - Number of healthy vs unhealthy exchanges
    - Last health check timestamps
    - Exchange configurations
    """
    try:
        health_data = await market_agent.get_exchange_health()
        return health_data
    except Exception as e:
        logger.error(f"Exchange health error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get exchange health: {str(e)}")

@app.post("/api/exchanges/best-prices", tags=["Multi-Exchange"])
@limiter.limit("20/minute")
async def find_best_prices(request: Request, symbols_request: Dict[str, List[str]] = Body(...)):
    """
    Find the best prices across all exchanges for multiple cryptocurrency symbols.
    
    Useful for:
    - Arbitrage opportunity detection
    - Price comparison across exchanges
    - Finding the best exchange for trading
    
    Request body should contain a "symbols" field with a list of trading symbols.
    Example: {"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}
    """
    try:
        if "symbols" not in symbols_request or not symbols_request["symbols"]:
            raise HTTPException(status_code=400, detail="Symbols list is required")
            
        if len(symbols_request["symbols"]) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed per request")
            
        # Find best prices across exchanges
        results = await market_agent.find_best_prices(symbols_request["symbols"])
        
        return results
    except Exception as e:
        logger.error(f"Best prices error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to find best prices: {str(e)}")

@app.get("/api/exchanges/coverage", tags=["Multi-Exchange"])
@limiter.limit("10/minute")
async def get_exchange_coverage(request: Request):
    """
    Get information about exchange coverage and capabilities.
    
    Returns:
    - List of supported exchanges
    - Exchange priorities and configurations
    - Estimated number of trading pairs per exchange
    - Exchange specialties (derivatives, spot, etc.)
    """
    try:
        coverage_info = {
            "status": "success",
            "exchanges": {
                "binance": {
                    "priority": 1,
                    "specialty": "Primary exchange with highest liquidity",
                    "estimated_pairs": 600,
                    "features": ["spot", "futures", "options"],
                    "rate_limit": "1200 requests/minute"
                },
                "kucoin": {
                    "priority": 2,
                    "specialty": "Early altcoin discovery and emerging tokens",
                    "estimated_pairs": 800,
                    "features": ["spot", "futures", "margin"],
                    "rate_limit": "100 requests/minute"
                },
                "bybit": {
                    "priority": 3,
                    "specialty": "Derivatives and Asian market focus",
                    "estimated_pairs": 400,
                    "features": ["spot", "derivatives", "funding_rates"],
                    "rate_limit": "120 requests/minute"
                },
                "gateio": {
                    "priority": 4,
                    "specialty": "Comprehensive coverage and new listings",
                    "estimated_pairs": 1200,
                    "features": ["spot", "margin", "new_listings"],
                    "rate_limit": "200 requests/minute"
                },
                "bitget": {
                    "priority": 5,
                    "specialty": "Copy trading and emerging markets",
                    "estimated_pairs": 500,
                    "features": ["spot", "futures", "copy_trading"],
                    "rate_limit": "150 requests/minute"
                }
            },
            "total_estimated_pairs": 3500,
            "capabilities": [
                "Multi-exchange price comparison",
                "Arbitrage opportunity detection", 
                "Intelligent failover routing",
                "Cross-exchange analytics",
                "Real-time health monitoring"
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return coverage_info
    except Exception as e:
        logger.error(f"Exchange coverage error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get exchange coverage: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", 1))
    )

# AI agent endpoints

from fastapi import APIRouter, HTTPException, Request, Body
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Dict, Any, List
from app.core.ai.agent import MarketAgent

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
market_agent = MarketAgent()

@router.post("/ask", tags=["AI Agent"])
@limiter.limit("10/minute")
async def ask_agent(request: Request, query: Dict[str, str] = Body(...)):
    """
    Ask the AI agent a natural language question about cryptocurrency markets.
    
    Examples:
    - "What is the current price of BTC?"
    - "What is the trend for Ethereum?"
    - "Should I buy SOL now?"
    - "How volatile is LINK today?"
    
    Request body should contain a "question" field with the natural language query.
    """
    try:
        if "question" not in query or not query["question"].strip():
            raise HTTPException(status_code=400, detail="Question is required")
            
        # Process the query
        response = await market_agent.process_query(query["question"])
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.get("/exchanges/health", tags=["Multi-Exchange"])
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
        raise HTTPException(status_code=500, detail=f"Failed to get exchange health: {str(e)}")

@router.post("/exchanges/best-prices", tags=["Multi-Exchange"])
@limiter.limit("20/minute")
async def find_best_prices(request: Request, symbols: Dict[str, List[str]] = Body(...)):
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
        if "symbols" not in symbols or not symbols["symbols"]:
            raise HTTPException(status_code=400, detail="Symbols list is required")
            
        if len(symbols["symbols"]) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed per request")
            
        # Find best prices across exchanges
        results = await market_agent.find_best_prices(symbols["symbols"])
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find best prices: {str(e)}")

@router.get("/exchanges/coverage", tags=["Multi-Exchange"])
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
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        return coverage_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get exchange coverage: {str(e)}")

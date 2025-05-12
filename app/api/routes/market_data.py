# Market data endpoints

from fastapi import APIRouter, HTTPException, Request
from typing import Optional, List
from datetime import datetime, timezone
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.services.binance import BinanceClient, SYMBOLS_CACHE

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

ALLOWED_INTERVALS = ["1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
INTERVAL_HOURS = {
    "1h": 1, "2h": 2, "4h": 4, "6h": 6, "8h": 8, "12h": 12, "1d": 24, "3d": 72, "1w": 168, "1M": 720
}

binance = BinanceClient()

@router.get("/symbols", tags=["Market Data"])
@limiter.limit("30/minute")
async def get_active_symbols(
    request: Request, 
    sort_by: Optional[str] = None, 
    descending: bool = True,
    quote_asset: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 500
):
    """
    Get list of available trading symbols.
    
    - **sort_by**: Sort by 'volume' or 'name'
    - **descending**: Sort in descending order if True
    - **quote_asset**: Filter by quote asset (e.g., 'USDT', 'BTC')
    - **search**: Search for specific symbols
    - **limit**: Maximum number of symbols to return
    """
    try:
        cache_key = "symbols"
        if not SYMBOLS_CACHE.get(cache_key):
            SYMBOLS_CACHE[cache_key] = binance.fetch_symbols()
        symbols = SYMBOLS_CACHE[cache_key]
        
        # Filter by quote asset if specified
        if quote_asset:
            quote_upper = quote_asset.upper()
            symbols = [s for s in symbols if s.endswith(quote_upper)]
        
        # Filter by search term if specified
        if search:
            search_upper = search.upper()
            symbols = [s for s in symbols if search_upper in s]

        # Sort by volume if requested
        if sort_by == "volume":
            sort_cache_key = f"symbols_sorted_volume_{quote_asset or ''}_{search or ''}"
            if not SYMBOLS_CACHE.get(sort_cache_key):
                tickers = binance.fetch_tickers()
                ticker_map = {t['symbol']: t for t in tickers}
                sorted_symbols = sorted(
                    symbols,
                    key=lambda s: float(ticker_map.get(s, {}).get('quoteVolume', 0)),
                    reverse=descending
                )
                SYMBOLS_CACHE[sort_cache_key] = sorted_symbols
            symbols = SYMBOLS_CACHE[sort_cache_key]
        # Sort by name if requested
        elif sort_by == "name":
            symbols = sorted(symbols, reverse=descending)
        
        # Apply limit
        symbols = symbols[:limit]
            
        return {
            "symbols": symbols,
            "total_count": len(symbols),
            "filter_applied": bool(quote_asset or search),
            "quote_asset": quote_asset,
            "sorting": f"{sort_by or 'none'}_{'desc' if descending else 'asc'}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@router.get("/intraday/{symbol}", tags=["Market Data"])
@limiter.limit("30/minute")
async def get_intraday_data(request: Request, symbol: str, interval: str = "1h"):
    try:
        symbol = symbol.upper()
        cache_key = "symbols"
        if not SYMBOLS_CACHE.get(cache_key):
            SYMBOLS_CACHE[cache_key] = binance.fetch_symbols()
        valid_symbols = SYMBOLS_CACHE[cache_key]
        
        if symbol not in valid_symbols:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} not found. Please check available symbols with /symbols endpoint."
            )
            
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Allowed values: {', '.join(ALLOWED_INTERVALS)}"
            )
        now = datetime.now(timezone.utc)
        start_of_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        interval_hours = INTERVAL_HOURS[interval]
        intervals_elapsed = int((now - start_of_day).total_seconds() / (interval_hours * 3600)) + 1
        limit = min(intervals_elapsed, 500)
        data = await binance.fetch_ohlcv(symbol, interval, limit=limit)
        if not data:
            raise HTTPException(status_code=404, detail="No intraday data available")
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intraday data retrieval failed: {str(e)}")

@router.get("/historical/{symbol}", tags=["Market Data"])
@limiter.limit("20/minute")
async def get_historical_data(request: Request, symbol: str, interval: str = "1h", limit: int = 100):
    try:
        symbol = symbol.upper()
        cache_key = "symbols"
        if not SYMBOLS_CACHE.get(cache_key):
            SYMBOLS_CACHE[cache_key] = binance.fetch_symbols()
        valid_symbols = SYMBOLS_CACHE[cache_key]
        
        if symbol not in valid_symbols:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} not found. Please check available symbols with /symbols endpoint."
            )
            
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Allowed values: {', '.join(ALLOWED_INTERVALS)}"
            )
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 1000"
            )
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Historical data retrieval failed: {str(e)}")

@router.get("/symbol/{symbol}/info", tags=["Market Data"])
@limiter.limit("30/minute")
async def get_symbol_info(request: Request, symbol: str):
    """
    Get detailed information about a specific trading symbol.
    """
    try:
        symbol = symbol.upper()
        cache_key = "symbols"
        if not SYMBOLS_CACHE.get(cache_key):
            SYMBOLS_CACHE[cache_key] = binance.fetch_symbols()
        valid_symbols = SYMBOLS_CACHE[cache_key]
        
        if symbol not in valid_symbols:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} not found. Please check available symbols with /symbols endpoint."
            )
        
        # Extract base and quote assets
        base_asset = ""
        quote_asset = ""
        
        # Check common quote assets
        common_quotes = ["USDT", "BTC", "ETH", "BNB", "BUSD", "USDC", "EUR", "TRY", "TUSD", "FDUSD"]
        for quote in common_quotes:
            if symbol.endswith(quote):
                quote_asset = quote
                base_asset = symbol[:-len(quote)]
                break
                
        # If still not found, use reasonable defaults
        if not base_asset:
            if len(symbol) >= 6:
                quote_asset = symbol[-4:]
                base_asset = symbol[:-4]
            else:
                quote_asset = "UNKNOWN"
                base_asset = symbol
        
        # Get ticker data
        tickers = binance.fetch_tickers()
        ticker_info = next((t for t in tickers if t['symbol'] == symbol), {})
        
        # Format the response
        return {
            "symbol": symbol,
            "base_asset": base_asset,
            "quote_asset": quote_asset,
            "price_change_24h": ticker_info.get('priceChangePercent', "0"),
            "last_price": ticker_info.get('lastPrice', "0"),
            "high_24h": ticker_info.get('highPrice', "0"),
            "low_24h": ticker_info.get('lowPrice', "0"),
            "volume_24h": ticker_info.get('volume', "0"),
            "quote_volume_24h": ticker_info.get('quoteVolume', "0"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symbol info retrieval failed: {str(e)}")

@router.get("/coins", tags=["Market Data"])
@limiter.limit("10/minute")
async def get_coins_with_trading_pairs(
    request: Request,
    min_quote_assets: int = 1,
    include_volume: bool = False
):
    """
    Get a structured list of all coins and their available trading pairs.
    
    - **min_quote_assets**: Minimum number of quote assets required for a coin to be included
    - **include_volume**: Include 24h volume data for each trading pair
    """
    try:
        # Fetch all symbols
        all_symbols = binance.fetch_symbols()
        
        # Common quote assets to check
        common_quotes = ["USDT", "BTC", "ETH", "BNB", "BUSD", "USDC", "EUR", "TRY", "TUSD", "FDUSD"]
        
        # Dictionary to store coins and their trading pairs
        coins = {}
        
        # Process each symbol
        for symbol in all_symbols:
            base_asset = None
            quote_asset = None
            
            # Try to extract base and quote assets
            for quote in common_quotes:
                if symbol.endswith(quote):
                    quote_asset = quote
                    base_asset = symbol[:-len(quote)]
                    break
            
            # If not found with common quotes, try a generic approach
            if not base_asset:
                # For longer symbols, assume last 3-4 characters are the quote asset
                if len(symbol) >= 6:
                    quote_asset = symbol[-4:]
                    base_asset = symbol[:-4]
                else:
                    # For very short symbols, make a best guess
                    quote_asset = symbol[-3:]
                    base_asset = symbol[:-3]
            
            # Add to the dictionary
            if base_asset not in coins:
                coins[base_asset] = {
                    "name": base_asset,
                    "trading_pairs": {}
                }
            
            coins[base_asset]["trading_pairs"][quote_asset] = {
                "symbol": symbol,
                "full_name": f"{base_asset}/{quote_asset}"
            }
        
        # Get volume data if requested
        if include_volume:
            tickers = binance.fetch_tickers()
            ticker_map = {t['symbol']: t for t in tickers}
            
            # Add volume data to each trading pair
            for coin in coins.values():
                for quote, pair_info in coin["trading_pairs"].items():
                    symbol = pair_info["symbol"]
                    if symbol in ticker_map:
                        pair_info["volume_24h"] = ticker_map[symbol].get('volume', "0")
                        pair_info["quote_volume_24h"] = ticker_map[symbol].get('quoteVolume', "0")
                        pair_info["price_change_24h"] = ticker_map[symbol].get('priceChangePercent', "0")
        
        # Filter coins by minimum number of quote assets
        filtered_coins = {
            name: data for name, data in coins.items() 
            if len(data["trading_pairs"]) >= min_quote_assets
        }
        
        # Convert to list and sort by name
        result = list(filtered_coins.values())
        result.sort(key=lambda x: x["name"])
        
        return {
            "coins": result,
            "total_coins": len(result),
            "total_trading_pairs": sum(len(coin["trading_pairs"]) for coin in result),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@router.get("/volatility/comparison", tags=["Market Data"])
@limiter.limit("10/minute")
async def compare_volatility(
    request: Request,
    symbols: Optional[str] = None,
    top: Optional[int] = 20,
    interval: str = "1h",
    sort: str = "desc"
):
    """
    Compare volatility across multiple cryptocurrencies.
    
    - **symbols**: Comma-separated list of symbols to compare (e.g., "BTCUSDT,ETHUSDT,LINKUSDT")
    - **top**: If symbols not provided, get this many top market cap coins with USDT pairs
    - **interval**: Time interval for data (e.g., "1h", "4h", "1d")
    - **sort**: Sort order for results - "asc" or "desc"
    """
    try:
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Allowed values: {', '.join(ALLOWED_INTERVALS)}"
            )
        
        # Get list of symbols to analyze
        symbols_to_analyze = []
        
        if symbols:
            # User provided symbols list
            symbols_to_analyze = [s.strip().upper() for s in symbols.split(",")]
            
            # Validate symbols
            cache_key = "symbols"
            if not SYMBOLS_CACHE.get(cache_key):
                SYMBOLS_CACHE[cache_key] = binance.fetch_symbols()
            valid_symbols = SYMBOLS_CACHE[cache_key]
            
            invalid_symbols = [s for s in symbols_to_analyze if s not in valid_symbols]
            if invalid_symbols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid symbols: {', '.join(invalid_symbols)}"
                )
        else:
            # Get top market cap USDT pairs
            tickers = binance.fetch_tickers()
            usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
            
            # Sort by volume (approximate for market cap)
            usdt_pairs.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
            
            # Take top N symbols
            symbols_to_analyze = [p['symbol'] for p in usdt_pairs[:top]]
        
        # Import needed classes
        from app.core.indicators.advanced import AverageTrueRange
        
        # Create ATR calculator
        atr_calculator = AverageTrueRange(window=14)
        
        # Collect volatility data for each symbol
        volatility_data = []
        
        for symbol in symbols_to_analyze:
            try:
                # Get OHLCV data
                ohlcv = await binance.fetch_ohlcv(symbol, interval, limit=30)
                
                if not ohlcv or len(ohlcv) < 14:
                    continue
                    
                # Extract price data
                highs = [candle["high"] for candle in ohlcv]
                lows = [candle["low"] for candle in ohlcv]
                closes = [candle["close"] for candle in ohlcv]
                
                # Calculate ATR
                atr_data = atr_calculator.get_signal(highs, lows, closes)
                
                # Calculate 24h price change
                price_change_24h = (closes[-1] - closes[-24]) / closes[-24] if len(closes) >= 24 else None
                
                # Create result object
                symbol_data = {
                    "symbol": symbol,
                    "current_price": closes[-1],
                    "atr_value": atr_data["atr_value"],
                    "atr_percent": atr_data["atr_percent"],
                    "volatility_level": atr_data["volatility"],
                    "price_change_24h": price_change_24h,
                    "market_phase": atr_data.get("market_phase", {}).get("phase", "UNDEFINED"),
                    "historical_percentile": atr_data.get("historical_comparison", {}).get("percentile", 50)
                }
                
                volatility_data.append(symbol_data)
                
            except Exception as e:
                logger.warning(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Sort results
        sort_reverse = sort.lower() != "asc"
        volatility_data.sort(key=lambda x: x["atr_percent"], reverse=sort_reverse)
        
        return {
            "interval": interval,
            "symbols_analyzed": len(volatility_data),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sorted_by": f"volatility_{sort}",
            "volatility_data": volatility_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Volatility comparison failed: {str(e)}")

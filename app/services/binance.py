import requests
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from typing import List, Dict, Optional
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables
load_dotenv()
BINANCE_API = os.getenv("BINANCE_API", "https://api.binance.com/api/v3")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300").split("#")[0].strip())

logger = logging.getLogger("CryptoPredictAPI")

# Caching setup
SYMBOLS_CACHE = TTLCache(maxsize=10, ttl=CACHE_TTL)
OHLCV_CACHE = TTLCache(maxsize=1000, ttl=300)
TICKER_CACHE = TTLCache(maxsize=5, ttl=60)  # More frequent ticker updates
CACHE_LOCK = asyncio.Lock()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

class BinanceClient:
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(requests.HTTPError)
    )
    def fetch_symbols(self) -> List[str]:
        """Synchronously fetch all trading symbols from Binance"""
        try:
            cache_key = "all_symbols"
            if cache_key in SYMBOLS_CACHE:
                return SYMBOLS_CACHE[cache_key]
                
            response = requests.get(f"{BINANCE_API}/exchangeInfo", headers=HEADERS)
            response.raise_for_status()
            symbols = [s["symbol"] for s in response.json()["symbols"] if s["status"] == "TRADING"]
            
            # Store in cache with timestamp
            SYMBOLS_CACHE[cache_key] = symbols
            SYMBOLS_CACHE["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Successfully fetched {len(symbols)} trading symbols")
            return symbols
        except Exception as e:
            logger.error(f"Symbols fetch error: {str(e)}")
            # If we've cached symbols before, return those instead of failing
            if "all_symbols" in SYMBOLS_CACHE:
                logger.warning("Using cached symbols due to API error")
                return SYMBOLS_CACHE["all_symbols"]
            raise

    async def fetch_symbols_async(self) -> List[str]:
        """Asynchronously fetch all trading symbols"""
        return await asyncio.to_thread(self.fetch_symbols)
        
    def get_symbol_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed information about a specific symbol"""
        try:
            # Try to get from cache first
            cache_key = "exchange_info"
            exchange_info = None
            
            if cache_key in SYMBOLS_CACHE:
                exchange_info = SYMBOLS_CACHE[cache_key]
            else:
                response = requests.get(f"{BINANCE_API}/exchangeInfo", headers=HEADERS)
                response.raise_for_status()
                exchange_info = response.json()
                SYMBOLS_CACHE[cache_key] = exchange_info
                
            # Find the symbol in the exchange info
            if exchange_info and "symbols" in exchange_info:
                for s in exchange_info["symbols"]:
                    if s["symbol"].upper() == symbol.upper() and s["status"] == "TRADING":
                        return s
            
            return None
        except Exception as e:
            logger.error(f"Symbol details fetch error for {symbol}: {str(e)}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.HTTPError, requests.Timeout))
    )
    async def fetch_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[Dict]:
        """Fetch OHLCV (candlestick) data for a symbol"""
        async with CACHE_LOCK:
            cache_key = f"{symbol}_{interval}_{limit}"
            if cache_key in OHLCV_CACHE:
                return OHLCV_CACHE[cache_key]
            
            try:
                response = await asyncio.to_thread(
                    requests.get,
                    f"{BINANCE_API}/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}",
                    headers=HEADERS,
                    timeout=5
                )
                response.raise_for_status()
                data = response.json()
                
                ohlcv = [{
                    "timestamp": entry[0],
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]),
                } for entry in data]
                
                OHLCV_CACHE[cache_key] = ohlcv
                return ohlcv
            except Exception as e:
                logger.error(f"OHLCV fetch error for {symbol}: {str(e)}")
                raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(requests.HTTPError)
    )
    def fetch_tickers(self) -> List[dict]:
        """Fetch 24h ticker data for all symbols"""
        try:
            cache_key = 'tickers'
            if cache_key in TICKER_CACHE:
                return TICKER_CACHE[cache_key]
                
            response = requests.get(f"{BINANCE_API}/ticker/24hr", headers=HEADERS)
            response.raise_for_status()
            tickers = response.json()
            TICKER_CACHE[cache_key] = tickers
            TICKER_CACHE['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            return tickers
        except Exception as e:
            logger.error(f"Tickers fetch error: {str(e)}")
            # If we've cached tickers before, return those instead of failing
            if 'tickers' in TICKER_CACHE:
                logger.warning("Using cached tickers due to API error")
                return TICKER_CACHE['tickers']
            raise
            
    async def fetch_tickers_async(self) -> List[dict]:
        """Asynchronously fetch 24h ticker data"""
        return await asyncio.to_thread(self.fetch_tickers)
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker data for a specific symbol"""
        try:
            response = await asyncio.to_thread(
                requests.get,
                f"{BINANCE_API}/ticker/24hr",
                params={"symbol": symbol},
                timeout=5
            )
            response.raise_for_status()
            
            data = response.json()
            return {
                "symbol": symbol,
                "price": float(data["lastPrice"]),
                "bid": float(data["bidPrice"]) if data["bidPrice"] else 0,
                "ask": float(data["askPrice"]) if data["askPrice"] else 0,
                "high24h": float(data["highPrice"]) if data["highPrice"] else 0,
                "low24h": float(data["lowPrice"]) if data["lowPrice"] else 0,
                "volume24h": float(data["volume"]) if data["volume"] else 0,
                "priceChangePercent": float(data["priceChangePercent"]) if data["priceChangePercent"] else 0,
            }
                
        except Exception as e:
            logger.error(f"Binance ticker fetch error for {symbol}: {str(e)}")
            return None
        
    def search_symbols(self, search_term: str, quote_asset: Optional[str] = None) -> List[str]:
        """Search for symbols matching a search term and optional quote asset"""
        try:
            # Get all symbols
            all_symbols = self.fetch_symbols()
            
            # Filter by search term (case insensitive)
            search_upper = search_term.upper()
            results = [s for s in all_symbols if search_upper in s]
            
            # Further filter by quote asset if provided
            if quote_asset:
                quote_upper = quote_asset.upper()
                results = [s for s in results if s.endswith(quote_upper)]
                
            return results
        except Exception as e:
            logger.error(f"Symbol search error: {str(e)}")
            return []
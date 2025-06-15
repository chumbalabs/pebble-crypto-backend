"""
Bybit API Client
Provides standardized interface for Bybit cryptocurrency exchange data
Strong focus on derivatives, futures, and Asian market coverage
"""

import requests
import asyncio
import logging
from typing import List, Dict, Optional
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BYBIT_API = os.getenv("BYBIT_API", "https://api.bybit.com/v5")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300").split("#")[0].strip())

logger = logging.getLogger("CryptoPredictAPI")

# Caching setup
SYMBOLS_CACHE = TTLCache(maxsize=10, ttl=CACHE_TTL)
OHLCV_CACHE = TTLCache(maxsize=1000, ttl=300)
TICKER_CACHE = TTLCache(maxsize=5, ttl=60)
CACHE_LOCK = asyncio.Lock()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

class BybitClient:
    """Bybit API client with focus on derivatives and futures trading"""
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(requests.HTTPError)
    )
    def fetch_symbols(self) -> List[str]:
        """Synchronously fetch all trading symbols from Bybit"""
        try:
            cache_key = "all_symbols"
            if cache_key in SYMBOLS_CACHE:
                return SYMBOLS_CACHE[cache_key]
                
            # Fetch both spot and linear (USDT perpetual) symbols
            symbols = []
            
            # Spot symbols
            response = requests.get(
                f"{BYBIT_API}/market/instruments-info",
                params={"category": "spot"},
                headers=HEADERS
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("retCode") == 0:  # Bybit success code
                spot_symbols = [s["symbol"] for s in data["result"]["list"] if s["status"] == "Trading"]
                symbols.extend(spot_symbols)
            
            # Linear (USDT Perpetual) symbols - Popular for derivatives trading
            response = requests.get(
                f"{BYBIT_API}/market/instruments-info",
                params={"category": "linear"},
                headers=HEADERS
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("retCode") == 0:
                linear_symbols = [s["symbol"] for s in data["result"]["list"] if s["status"] == "Trading"]
                symbols.extend(linear_symbols)
            
            # Store in cache with timestamp
            SYMBOLS_CACHE[cache_key] = symbols
            SYMBOLS_CACHE["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Successfully fetched {len(symbols)} Bybit trading symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Bybit symbols fetch error: {str(e)}")
            # If we've cached symbols before, return those instead of failing
            if "all_symbols" in SYMBOLS_CACHE:
                logger.warning("Using cached Bybit symbols due to API error")
                return SYMBOLS_CACHE["all_symbols"]
            raise

    async def fetch_symbols_async(self) -> List[str]:
        """Asynchronously fetch all trading symbols"""
        return await asyncio.to_thread(self.fetch_symbols)
        
    def get_symbol_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed information about a specific symbol"""
        try:
            # Try spot first, then linear
            for category in ["spot", "linear"]:
                response = requests.get(
                    f"{BYBIT_API}/market/instruments-info",
                    params={"category": category, "symbol": symbol},
                    headers=HEADERS
                )
                response.raise_for_status()
                
                data = response.json()
                if data.get("retCode") == 0 and data["result"]["list"]:
                    symbol_info = data["result"]["list"][0]
                    if symbol_info["status"] == "Trading":
                        return {
                            **symbol_info,
                            "category": category  # Add category info
                        }
            
            return None
        except Exception as e:
            logger.error(f"Bybit symbol details fetch error for {symbol}: {str(e)}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.HTTPError, requests.Timeout))
    )
    async def fetch_ohlcv(self, symbol: str, interval: str = "60", limit: int = 100) -> List[Dict]:
        """Fetch OHLCV (candlestick) data for a symbol"""
        async with CACHE_LOCK:
            cache_key = f"bybit_{symbol}_{interval}_{limit}"
            if cache_key in OHLCV_CACHE:
                return OHLCV_CACHE[cache_key]
            
            try:
                # Convert interval to Bybit format
                bybit_interval = self._convert_interval(interval)
                
                # Determine category (spot vs linear) for the symbol
                category = await self._determine_symbol_category(symbol)
                
                response = await asyncio.to_thread(
                    requests.get,
                    f"{BYBIT_API}/market/kline",
                    params={
                        "category": category,
                        "symbol": symbol,
                        "interval": bybit_interval,
                        "limit": limit
                    },
                    headers=HEADERS,
                    timeout=10
                )
                response.raise_for_status()
                
                data = response.json()
                if data.get("retCode") == 0:
                    raw_data = data["result"]["list"]
                    
                    # Convert Bybit format to standard format
                    ohlcv = []
                    for entry in raw_data:
                        # Bybit format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
                        ohlcv.append({
                            "timestamp": int(entry[0]),
                            "open": float(entry[1]),
                            "high": float(entry[2]),
                            "low": float(entry[3]),
                            "close": float(entry[4]),
                            "volume": float(entry[5]),
                        })
                    
                    # Sort by timestamp (oldest first)
                    ohlcv.sort(key=lambda x: x["timestamp"])
                    
                    OHLCV_CACHE[cache_key] = ohlcv
                    return ohlcv
                else:
                    raise Exception(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Bybit OHLCV fetch error for {symbol}: {str(e)}")
                raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(requests.HTTPError)
    )
    def fetch_tickers(self) -> List[dict]:
        """Fetch 24h ticker data for all symbols"""
        try:
            cache_key = 'bybit_tickers'
            if cache_key in TICKER_CACHE:
                return TICKER_CACHE[cache_key]
                
            tickers = []
            
            # Fetch spot tickers
            for category in ["spot", "linear"]:
                response = requests.get(
                    f"{BYBIT_API}/market/tickers",
                    params={"category": category},
                    headers=HEADERS
                )
                response.raise_for_status()
                
                data = response.json()
                if data.get("retCode") == 0:
                    category_tickers = data["result"]["list"]
                    
                    # Convert to standard format
                    for ticker in category_tickers:
                        tickers.append({
                            "symbol": ticker["symbol"],
                            "price": float(ticker["lastPrice"]) if ticker["lastPrice"] else 0,
                            "priceChange": float(ticker["price24hPcnt"]) * float(ticker["lastPrice"]) / 100 if ticker["price24hPcnt"] and ticker["lastPrice"] else 0,
                            "priceChangePercent": float(ticker["price24hPcnt"]) * 100 if ticker["price24hPcnt"] else 0,
                            "high": float(ticker["highPrice24h"]) if ticker["highPrice24h"] else 0,
                            "low": float(ticker["lowPrice24h"]) if ticker["lowPrice24h"] else 0,
                            "volume": float(ticker["volume24h"]) if ticker["volume24h"] else 0,
                            "quoteVolume": float(ticker["turnover24h"]) if ticker["turnover24h"] else 0,
                            "category": category  # Add category info
                        })
            
            TICKER_CACHE[cache_key] = tickers
            TICKER_CACHE['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            return tickers
                
        except Exception as e:
            logger.error(f"Bybit tickers fetch error: {str(e)}")
            # If we've cached tickers before, return those instead of failing
            if 'bybit_tickers' in TICKER_CACHE:
                logger.warning("Using cached Bybit tickers due to API error")
                return TICKER_CACHE['bybit_tickers']
            raise
            
    async def fetch_tickers_async(self) -> List[dict]:
        """Asynchronously fetch 24h ticker data"""
        return await asyncio.to_thread(self.fetch_tickers)
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker data for a specific symbol"""
        try:
            # Determine category for the symbol
            category = await self._determine_symbol_category(symbol)
            
            response = await asyncio.to_thread(
                requests.get,
                f"{BYBIT_API}/market/tickers",
                params={"category": category, "symbol": symbol},
                headers=HEADERS,
                timeout=5
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("retCode") == 0 and data["result"]["list"]:
                ticker_data = data["result"]["list"][0]
                
                return {
                    "symbol": symbol,
                    "price": float(ticker_data["lastPrice"]) if ticker_data["lastPrice"] else 0,
                    "bid": float(ticker_data["bid1Price"]) if ticker_data["bid1Price"] else 0,
                    "ask": float(ticker_data["ask1Price"]) if ticker_data["ask1Price"] else 0,
                    "high24h": float(ticker_data["highPrice24h"]) if ticker_data["highPrice24h"] else 0,
                    "low24h": float(ticker_data["lowPrice24h"]) if ticker_data["lowPrice24h"] else 0,
                    "volume24h": float(ticker_data["volume24h"]) if ticker_data["volume24h"] else 0,
                    "priceChangePercent": float(ticker_data["price24hPcnt"]) * 100 if ticker_data["price24hPcnt"] else 0,
                    "category": category
                }
            else:
                raise Exception(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Bybit ticker fetch error for {symbol}: {str(e)}")
            return None
    
    async def get_funding_rates(self, symbol: str) -> Optional[Dict]:
        """Get funding rates for perpetual contracts (Bybit specialty)"""
        try:
            response = await asyncio.to_thread(
                requests.get,
                f"{BYBIT_API}/market/funding/history",
                params={"category": "linear", "symbol": symbol, "limit": 1},
                headers=HEADERS,
                timeout=5
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("retCode") == 0 and data["result"]["list"]:
                funding_data = data["result"]["list"][0]
                
                return {
                    "symbol": symbol,
                    "fundingRate": float(funding_data["fundingRate"]),
                    "fundingTime": int(funding_data["fundingRateTimestamp"]),
                    "nextFundingTime": None  # Would need additional API call
                }
            
            return None
                
        except Exception as e:
            logger.error(f"Bybit funding rates fetch error for {symbol}: {str(e)}")
            return None
    
    async def get_derivatives_data(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive derivatives data (futures, perpetuals)"""
        try:
            # Get basic ticker data
            ticker = await self.get_ticker(symbol)
            if not ticker:
                return None
            
            # Get funding rates if it's a linear contract
            funding_rates = None
            if ticker.get("category") == "linear":
                funding_rates = await self.get_funding_rates(symbol)
            
            # Get open interest
            oi_response = await asyncio.to_thread(
                requests.get,
                f"{BYBIT_API}/market/open-interest",
                params={"category": "linear", "symbol": symbol, "intervalTime": "5min", "limit": 1},
                headers=HEADERS,
                timeout=5
            )
            
            open_interest = None
            if oi_response.status_code == 200:
                oi_data = oi_response.json()
                if oi_data.get("retCode") == 0 and oi_data["result"]["list"]:
                    oi_info = oi_data["result"]["list"][0]
                    open_interest = {
                        "openInterest": float(oi_info["openInterest"]),
                        "timestamp": int(oi_info["timestamp"])
                    }
            
            return {
                "ticker": ticker,
                "funding_rates": funding_rates,
                "open_interest": open_interest,
                "exchange": "bybit"
            }
            
        except Exception as e:
            logger.error(f"Bybit derivatives data fetch error for {symbol}: {str(e)}")
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
            logger.error(f"Bybit symbol search error: {str(e)}")
            return []
    
    async def _determine_symbol_category(self, symbol: str) -> str:
        """Determine if a symbol is spot or linear (perpetual)"""
        # Simple heuristic: if symbol ends with USDT and doesn't have specific spot indicators
        # it's likely a linear (perpetual) contract on Bybit
        if symbol.endswith("USDT") and not any(x in symbol for x in ["-", "/"]):
            # Check if it's a known perpetual contract
            try:
                response = await asyncio.to_thread(
                    requests.get,
                    f"{BYBIT_API}/market/instruments-info",
                    params={"category": "linear", "symbol": symbol},
                    headers=HEADERS,
                    timeout=3
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("retCode") == 0 and data["result"]["list"]:
                        return "linear"
            except:
                pass
        
        # Default to spot
        return "spot"
    
    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to Bybit format"""
        interval_map = {
            "1m": "1",
            "3m": "3", 
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "6h": "360",
            "12h": "720",
            "1d": "D",
            "1w": "W",
            "1M": "M"
        }
        return interval_map.get(interval, "60") 
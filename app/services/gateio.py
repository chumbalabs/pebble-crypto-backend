"""
Gate.io API Client
Provides standardized interface for Gate.io cryptocurrency exchange data
Focus on comprehensive coverage and new token listings
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

GATEIO_API = os.getenv("GATEIO_API", "https://api.gateio.ws/api/v4")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300").split("#")[0].strip())

logger = logging.getLogger("CryptoPredictAPI")

# Caching setup
SYMBOLS_CACHE = TTLCache(maxsize=10, ttl=CACHE_TTL)
OHLCV_CACHE = TTLCache(maxsize=1000, ttl=300)
TICKER_CACHE = TTLCache(maxsize=5, ttl=60)
CACHE_LOCK = asyncio.Lock()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json"
}

class GateIOClient:
    """Gate.io API client with comprehensive market coverage"""
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(requests.HTTPError)
    )
    def fetch_symbols(self) -> List[str]:
        """Synchronously fetch all trading symbols from Gate.io"""
        try:
            cache_key = "all_symbols"
            if cache_key in SYMBOLS_CACHE:
                return SYMBOLS_CACHE[cache_key]
                
            response = requests.get(f"{GATEIO_API}/spot/currency_pairs", headers=HEADERS)
            response.raise_for_status()
            
            data = response.json()
            # Gate.io returns array directly
            symbols = [s["id"] for s in data if s["trade_status"] == "tradable"]
            
            # Store in cache with timestamp
            SYMBOLS_CACHE[cache_key] = symbols
            SYMBOLS_CACHE["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Successfully fetched {len(symbols)} Gate.io trading symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Gate.io symbols fetch error: {str(e)}")
            # If we've cached symbols before, return those instead of failing
            if "all_symbols" in SYMBOLS_CACHE:
                logger.warning("Using cached Gate.io symbols due to API error")
                return SYMBOLS_CACHE["all_symbols"]
            raise

    async def fetch_symbols_async(self) -> List[str]:
        """Asynchronously fetch all trading symbols"""
        return await asyncio.to_thread(self.fetch_symbols)
        
    def get_symbol_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed information about a specific symbol"""
        try:
            response = requests.get(
                f"{GATEIO_API}/spot/currency_pairs/{symbol}",
                headers=HEADERS
            )
            response.raise_for_status()
            
            data = response.json()
            if data and data.get("trade_status") == "tradable":
                return data
            
            return None
        except Exception as e:
            logger.error(f"Gate.io symbol details fetch error for {symbol}: {str(e)}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.HTTPError, requests.Timeout))
    )
    async def fetch_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[Dict]:
        """Fetch OHLCV (candlestick) data for a symbol"""
        async with CACHE_LOCK:
            cache_key = f"gateio_{symbol}_{interval}_{limit}"
            if cache_key in OHLCV_CACHE:
                return OHLCV_CACHE[cache_key]
            
            try:
                # Convert interval to Gate.io format
                gateio_interval = self._convert_interval(interval)
                
                # Calculate time range
                end_time = int(datetime.now().timestamp())
                start_time = end_time - (limit * self._interval_to_seconds(interval))
                
                response = await asyncio.to_thread(
                    requests.get,
                    f"{GATEIO_API}/spot/candlesticks",
                    params={
                        "currency_pair": symbol,
                        "interval": gateio_interval,
                        "from": start_time,
                        "to": end_time,
                        "limit": limit
                    },
                    headers=HEADERS,
                    timeout=10
                )
                response.raise_for_status()
                
                data = response.json()
                # Gate.io returns array directly
                
                # Convert Gate.io format to standard format
                ohlcv = []
                for entry in data:
                    # Gate.io format: [timestamp, volume, close, high, low, open]
                    ohlcv.append({
                        "timestamp": int(entry[0]) * 1000,  # Convert to milliseconds
                        "open": float(entry[5]),
                        "high": float(entry[3]),
                        "low": float(entry[4]),
                        "close": float(entry[2]),
                        "volume": float(entry[1]),
                    })
                
                # Sort by timestamp (oldest first)
                ohlcv.sort(key=lambda x: x["timestamp"])
                
                OHLCV_CACHE[cache_key] = ohlcv
                return ohlcv
                    
            except Exception as e:
                logger.error(f"Gate.io OHLCV fetch error for {symbol}: {str(e)}")
                raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(requests.HTTPError)
    )
    def fetch_tickers(self) -> List[dict]:
        """Fetch 24h ticker data for all symbols"""
        try:
            cache_key = 'gateio_tickers'
            if cache_key in TICKER_CACHE:
                return TICKER_CACHE[cache_key]
                
            response = requests.get(f"{GATEIO_API}/spot/tickers", headers=HEADERS)
            response.raise_for_status()
            
            data = response.json()
            # Gate.io returns array directly
            
            # Convert to standard format
            standardized_tickers = []
            for ticker in data:
                try:
                    standardized_tickers.append({
                        "symbol": ticker["currency_pair"],
                        "price": float(ticker["last"]) if ticker["last"] else 0,
                        "priceChange": float(ticker["change_percentage"]) * float(ticker["last"]) / 100 if ticker["change_percentage"] and ticker["last"] else 0,
                        "priceChangePercent": float(ticker["change_percentage"]) if ticker["change_percentage"] else 0,
                        "high": float(ticker["high_24h"]) if ticker["high_24h"] else 0,
                        "low": float(ticker["low_24h"]) if ticker["low_24h"] else 0,
                        "volume": float(ticker["base_volume"]) if ticker["base_volume"] else 0,
                        "quoteVolume": float(ticker["quote_volume"]) if ticker["quote_volume"] else 0,
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing ticker for {ticker.get('currency_pair', 'unknown')}: {str(e)}")
                    continue
            
            TICKER_CACHE[cache_key] = standardized_tickers
            TICKER_CACHE['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            return standardized_tickers
                
        except Exception as e:
            logger.error(f"Gate.io tickers fetch error: {str(e)}")
            # If we've cached tickers before, return those instead of failing
            if 'gateio_tickers' in TICKER_CACHE:
                logger.warning("Using cached Gate.io tickers due to API error")
                return TICKER_CACHE['gateio_tickers']
            raise
            
    async def fetch_tickers_async(self) -> List[dict]:
        """Asynchronously fetch 24h ticker data"""
        return await asyncio.to_thread(self.fetch_tickers)
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker data for a specific symbol"""
        try:
            # Convert BTCUSDT format to BTC_USDT format for Gate.io
            gateio_symbol = self._convert_symbol_format(symbol)
            
            response = await asyncio.to_thread(
                requests.get,
                f"{GATEIO_API}/spot/tickers",
                params={"currency_pair": gateio_symbol},
                headers=HEADERS,
                timeout=5
            )
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                ticker_data = data[0]  # Gate.io returns array even for single symbol
                
                return {
                    "symbol": symbol,
                    "price": float(ticker_data["last"]) if ticker_data["last"] else 0,
                    "bid": float(ticker_data["highest_bid"]) if ticker_data["highest_bid"] else 0,
                    "ask": float(ticker_data["lowest_ask"]) if ticker_data["lowest_ask"] else 0,
                    "high24h": float(ticker_data["high_24h"]) if ticker_data["high_24h"] else 0,
                    "low24h": float(ticker_data["low_24h"]) if ticker_data["low_24h"] else 0,
                    "volume24h": float(ticker_data["base_volume"]) if ticker_data["base_volume"] else 0,
                    "priceChangePercent": float(ticker_data["change_percentage"]) if ticker_data["change_percentage"] else 0,
                }
            
            return None
                
        except Exception as e:
            logger.error(f"Gate.io ticker fetch error for {symbol}: {str(e)}")
            return None
    
    async def get_new_listings(self, days: int = 7) -> List[Dict]:
        """Get recently listed tokens (Gate.io specialty)"""
        try:
            # Get all currency pairs with details
            response = await asyncio.to_thread(
                requests.get,
                f"{GATEIO_API}/spot/currency_pairs",
                headers=HEADERS,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Filter for recently listed tokens (this is a simplified approach)
            # In a real implementation, you'd want to track listing dates
            new_listings = []
            current_time = datetime.now(timezone.utc)
            
            for pair in data:
                if pair["trade_status"] == "tradable":
                    # Get additional details for each pair
                    try:
                        ticker_data = await self.get_ticker(pair["id"])
                        if ticker_data and ticker_data["volume24h"] > 0:  # Has trading activity
                            new_listings.append({
                                "symbol": pair["id"],
                                "base_currency": pair["base"],
                                "quote_currency": pair["quote"],
                                "min_base_amount": pair.get("min_base_amount"),
                                "min_quote_amount": pair.get("min_quote_amount"),
                                "ticker_data": ticker_data,
                                "precision": {
                                    "amount": pair.get("amount_precision"),
                                    "price": pair.get("precision")
                                }
                            })
                    except Exception as e:
                        logger.warning(f"Error getting details for {pair['id']}: {str(e)}")
                        continue
            
            # Sort by volume (proxy for new/active listings)
            new_listings.sort(key=lambda x: x["ticker_data"]["volume24h"], reverse=True)
            
            return new_listings[:50]  # Return top 50
            
        except Exception as e:
            logger.error(f"Gate.io new listings fetch error: {str(e)}")
            return []
    
    async def get_spot_data(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive spot trading data"""
        try:
            # Get ticker data
            ticker = await self.get_ticker(symbol)
            if not ticker:
                return None
            
            # Get order book depth
            depth_response = await asyncio.to_thread(
                requests.get,
                f"{GATEIO_API}/spot/order_book",
                params={"currency_pair": symbol, "limit": 10},
                headers=HEADERS,
                timeout=5
            )
            
            order_book = None
            if depth_response.status_code == 200:
                depth_data = depth_response.json()
                order_book = {
                    "bids": [[float(bid[0]), float(bid[1])] for bid in depth_data.get("bids", [])[:5]],
                    "asks": [[float(ask[0]), float(ask[1])] for ask in depth_data.get("asks", [])[:5]],
                    "timestamp": depth_data.get("id", int(datetime.now().timestamp()))
                }
            
            # Get recent trades
            trades_response = await asyncio.to_thread(
                requests.get,
                f"{GATEIO_API}/spot/trades",
                params={"currency_pair": symbol, "limit": 10},
                headers=HEADERS,
                timeout=5
            )
            
            recent_trades = None
            if trades_response.status_code == 200:
                trades_data = trades_response.json()
                recent_trades = [
                    {
                        "price": float(trade["price"]),
                        "amount": float(trade["amount"]),
                        "side": trade["side"],
                        "timestamp": int(trade["create_time"])
                    }
                    for trade in trades_data[:10]
                ]
            
            return {
                "ticker": ticker,
                "order_book": order_book,
                "recent_trades": recent_trades,
                "exchange": "gateio"
            }
            
        except Exception as e:
            logger.error(f"Gate.io spot data fetch error for {symbol}: {str(e)}")
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
                results = [s for s in results if s.endswith(f"_{quote_upper}")]
                
            return results
        except Exception as e:
            logger.error(f"Gate.io symbol search error: {str(e)}")
            return []
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert BTCUSDT format to BTC_USDT format for Gate.io"""
        # Gate.io uses underscore format: BTC_USDT instead of BTCUSDT
        if "_" not in symbol:
            # Common patterns for conversion
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                return f"{base}_USDT"
            elif symbol.endswith("BTC"):
                base = symbol[:-3]
                return f"{base}_BTC"
            elif symbol.endswith("ETH"):
                base = symbol[:-3]
                return f"{base}_ETH"
            elif symbol.endswith("BNB"):
                base = symbol[:-3]
                return f"{base}_BNB"
        return symbol
    
    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to Gate.io format"""
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "8h": "8h",
            "1d": "1d",
            "7d": "7d",
            "30d": "30d"
        }
        return interval_map.get(interval, "1h")
    
    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds"""
        interval_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "8h": 28800,
            "1d": 86400,
            "7d": 604800,
            "30d": 2592000
        }
        return interval_seconds.get(interval, 3600) 
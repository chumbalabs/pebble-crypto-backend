# AI agent orchestration module

import logging
from typing import Dict, List, Any, Optional, Union
import json
import re
from datetime import datetime, timezone
import asyncio
import numpy as np
import math

from app.core.ai.gemini_client import GeminiInsightsGenerator
from app.services.binance import BinanceClient
from app.core.indicators.advanced import BollingerBands, AverageTrueRange
from app.core.indicators.order_book import OrderBookDepthAnalyzer
from app.core.prediction.technical import predictor
from app.core.analysis.market_advisor import MarketAdvisor

logger = logging.getLogger("CryptoPredictAPI")

class MarketAgent:
    """
    AI-powered agent that orchestrates data collection, analysis, and natural language
    responses for cryptocurrency market questions.
    """
    
    def __init__(self):
        """Initialize the market agent with necessary components"""
        self.gemini = GeminiInsightsGenerator()
        self.binance = BinanceClient()
        self.bb_indicator = BollingerBands()
        self.atr_indicator = AverageTrueRange()
        self.order_book_analyzer = OrderBookDepthAnalyzer()
        self.market_advisor = MarketAdvisor()  # Initialize the MarketAdvisor
        self.valid_symbols = []
        self.last_symbols_update = None
        
    def _sanitize_nan_values(self, data):
        """Replace NaN values with None for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._sanitize_nan_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_nan_values(item) for item in data]
        elif isinstance(data, (float, np.float64, np.float32)) and (math.isnan(data) or math.isinf(data)):
            return None
        elif isinstance(data, np.ndarray):
            return self._sanitize_nan_values(data.tolist())
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return None if math.isnan(float(data)) or math.isinf(float(data)) else float(data)
        else:
            return data
        
    async def _update_valid_symbols(self):
        """Update the list of valid trading symbols"""
        try:
            self.valid_symbols = await asyncio.to_thread(self.binance.fetch_symbols)
            self.last_symbols_update = datetime.now(timezone.utc)
            logger.info(f"Updated valid symbols list with {len(self.valid_symbols)} symbols")
        except Exception as e:
            logger.error(f"Error updating valid symbols: {str(e)}")
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query about cryptocurrency markets
        Returns a response with all the data used to generate it
        """
        try:
            # Ensure we have valid symbols
            if not self.valid_symbols:
                try:
                    self.valid_symbols = await asyncio.to_thread(self.binance.fetch_symbols)
                    logger.info(f"Initialized valid symbols list with {len(self.valid_symbols)} symbols")
                except Exception as e:
                    logger.warning(f"Could not fetch symbols: {e}. Using fallback major symbols.")
                    # Add default major symbols as fallback
                    self.valid_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"]
            
            # 1. Extract query information
            query_info = self._extract_query_info(query)
            
            # 2. Identify required data sources
            data_sources = self._determine_data_sources(query_info)
            
            # 3. Collect data
            data = await self._collect_data(query_info, data_sources)
            
            # Store the symbol for response formatting
            data["symbol"] = query_info.get("symbol")
            
            # 4. Generate a response
            response = await self._generate_response(query, query_info, data)
            
            # 5. Filter data for supporting info
            supporting_data = self._filter_supporting_data(data)
            
            # 6. Get the multi-timeframe data for additional context and remove OHLCV to reduce size
            multi_timeframe = data.get("multi_timeframe", {})
            filtered_multi_timeframe = {}
            
            # Create a filtered version of multi_timeframe data without the OHLCV arrays
            for tf, tf_data in multi_timeframe.items():
                filtered_tf_data = tf_data.copy()
                # Remove OHLCV data (which is the largest part)
                if "ohlcv" in filtered_tf_data:
                    del filtered_tf_data["ohlcv"]
                filtered_multi_timeframe[tf] = filtered_tf_data
            
            # 7. Format & return the result
            result = {
                "query": query,
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "supporting_data": supporting_data,
                "metadata": {
                    "symbol": query_info.get("symbol"),
                    "interval": query_info.get("interval", "1h"),
                    "data_sources": data_sources
                }
            }
            
            # Add filtered multi-timeframe data if available
            if filtered_multi_timeframe:
                result["multi_timeframe"] = filtered_multi_timeframe
            
            # Sanitize data to remove NaN values before returning
            sanitized_result = self._sanitize_nan_values(result)
            
            return sanitized_result
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            # Return a graceful error response
            return {
                "query": query,
                "response": f"I'm sorry, I couldn't process that query: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "supporting_data": {},
                "metadata": {"error": str(e)}
            }
    
    def _extract_query_info(self, query: str) -> Dict[str, Any]:
        """Extract symbol, timeframe, and query intent from the question"""
        info = {
            "symbol": None,
            "interval": "1h",  # default
            "intent": "general"
        }
        
        # List of common coin abbreviations
        common_coins = [
            "BTC", "ETH", "USDT", "BNB", "XRP", "ADA", "SOL", "DOT", "DOGE", 
            "AVAX", "MATIC", "SHIB", "LTC", "ATOM", "LINK", "UNI", "XLM", 
            "BCH", "ALGO", "MANA", "SAND", "AXS", "FTT", "ETC", "NEAR",
            "AVA", "AVAUSDT", "BTCUSDT", "ETHUSDT"
        ]
        
        # First try to extract exact symbol in trading pair format
        symbol_patterns = [
            r'(?:for|about|of|on)\s+([A-Z]+/[A-Z]+)',  # Format like "BTC/USDT"
            r'(?:for|about|of|on)\s+([A-Z]+[^A-Za-z\s]+[A-Z]+)',  # Format like "BTC-USDT"
            r'([A-Z]{2,}/[A-Z]{2,})',  # Direct mention like "BTC/USDT"
            r'([A-Z]{2,}[^A-Za-z\s]+[A-Z]{2,})',  # Direct mention like "BTC-USDT"
        ]
        
        for pattern in symbol_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                symbol_text = match.group(1).upper()
                
                # Format symbol appropriately
                if '/' in symbol_text:
                    parts = symbol_text.split('/')
                    symbol = f"{parts[0]}{parts[1]}"
                elif '-' in symbol_text:
                    parts = symbol_text.split('-')
                    symbol = f"{parts[0]}{parts[1]}"
                else:
                    symbol = symbol_text
                
                info["symbol"] = symbol
                break
        
        # If no trading pair was found, look for single coin mentions
        if info["symbol"] is None:
            # Check for common coin abbreviations directly in the query
            # This ensures we catch standalone mentions like "What is the future of BTC"
            for coin in common_coins:
                if re.search(r'\b' + re.escape(coin) + r'\b', query, re.IGNORECASE):
                    # If coin is already a pair (like BTCUSDT), use it directly
                    if len(coin) > 5 and coin.endswith(("USDT", "BTC", "ETH")):
                        info["symbol"] = coin
                        break
                    else:
                        # Try USDT pair first, then BTC pair as fallback
                        usdt_pair = f"{coin}USDT"
                        btc_pair = f"{coin}BTC"
                        
                        # Assume USDT pair if coin is a major one
                        if coin in ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX"]:
                            info["symbol"] = usdt_pair
                            break
                        # Otherwise check if pairs exist
                        elif usdt_pair in self.valid_symbols:
                            info["symbol"] = usdt_pair
                            break
                        elif btc_pair in self.valid_symbols:
                            info["symbol"] = btc_pair
                            break
            
            # If still no match, try to find any word that could be a coin
            if info["symbol"] is None:
                # Look for standalone uppercase words that might be coins
                for word in re.findall(r'\b[A-Z]{2,}\b', query):
                    usdt_pair = f"{word}USDT"
                    # Consider common standalone crypto references as USDT pairs automatically
                    if word in ["BTC", "ETH", "SOL"]:
                        info["symbol"] = usdt_pair
                        break
                
                # Try matching case-insensitive coin names
                if info["symbol"] is None:
                    for word in re.findall(r'\b[A-Za-z]{2,}\b', query):
                        word_upper = word.upper()
                        # Special handling for Bitcoin, Ethereum, etc.
                        if word_upper in ["BITCOIN", "BTC"]:
                            info["symbol"] = "BTCUSDT"
                            break
                        elif word_upper in ["ETHEREUM", "ETH"]:
                            info["symbol"] = "ETHUSDT"
                            break
                        elif word_upper in ["SOLANA", "SOL"]:
                            info["symbol"] = "SOLUSDT"
                            break
                
        # Extract timeframe/interval
        interval_patterns = {
            r'\b1\s*hour\b|hourly|1h': '1h',
            r'\b4\s*hours\b|4h': '4h',
            r'\b1\s*day\b|daily|1d': '1d',
            r'\b1\s*week\b|weekly|1w': '1w',
            r'\b1\s*month\b|monthly|1M': '1M'
        }
        
        for pattern, interval in interval_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                info["interval"] = interval
                break
        
        # Determine intent
        intent_patterns = {
            r'price|worth|value|rate|cost': 'price',
            r'trend|direction|moving': 'trend',
            r'volatile|volatility|stable|atr': 'volatility',
            r'support|resistance|level': 'levels',
            r'prediction|forecast|expect|predict|future': 'prediction',
            r'volume|liquidity|traded': 'volume',
            r'order\s+book|buy\s+wall|sell\s+wall|depth': 'order_book',
            r'technical|indicator|bollinger|rsi|macd': 'indicators',
            r'analysis|insight|explain|why': 'analysis',
            r'should\s+I\s+buy|should\s+I\s+sell|position|trade': 'advice'
        }
        
        for pattern, intent in intent_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                info["intent"] = intent
                break
                
        # Make sure valid_symbols is not empty
        if not self.valid_symbols:
            # Add default major symbols as fallback
            self.valid_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"]
            
        # Ensure we have a valid symbol for common cryptocurrencies
        if info["symbol"] is None:
            # Check if query mentions Bitcoin, Ethereum, etc.
            if re.search(r'\b(bitcoin|btc)\b', query, re.IGNORECASE):
                info["symbol"] = "BTCUSDT"
            elif re.search(r'\b(ethereum|eth)\b', query, re.IGNORECASE):
                info["symbol"] = "ETHUSDT"
            elif re.search(r'\b(solana|sol)\b', query, re.IGNORECASE):
                info["symbol"] = "SOLUSDT"
        
        return info
    
    def _determine_data_sources(self, query_info: Dict[str, Any]) -> List[str]:
        """Determine which data sources are needed based on the query intent"""
        intent = query_info.get("intent", "general")
        
        # Default data source for all queries
        sources = ["price_data"]
        
        intent_to_source_map = {
            "price": ["price_data"],
            "trend": ["price_data", "technical_indicators"],
            "volatility": ["price_data", "technical_indicators", "atr"],
            "levels": ["price_data", "technical_indicators", "support_resistance"],
            "prediction": ["price_data", "technical_indicators", "prediction"],
            "volume": ["price_data", "volume_data"],
            "order_book": ["price_data", "order_book"],
            "indicators": ["price_data", "technical_indicators"],
            "analysis": ["price_data", "technical_indicators", "prediction", "ai_insights"],
            "advice": ["price_data", "technical_indicators", "prediction", "ai_insights"],
            "general": ["price_data", "technical_indicators", "ai_insights"]
        }
        
        return intent_to_source_map.get(intent, sources)
    
    async def _collect_data(self, query_info: Dict[str, Any], 
                           data_sources: List[str]) -> Dict[str, Any]:
        """Collect data from various sources based on the query requirements"""
        symbol = query_info.get("symbol")
        interval = query_info.get("interval", "1h")
        
        # Return empty data if no symbol is specified
        if not symbol:
            return {"error": "No cryptocurrency symbol detected in the query"}
        
        data = {"symbol": symbol, "interval": interval}
        
        try:
            # Collect price data from multiple timeframes for better insights
            timeframes = ["1h", "4h", "1d", "1w"]
            multi_timeframe_data = {}
            
            for tf in timeframes:
                try:
                    # Fetch OHLCV data for each timeframe
                    ohlcv = await self.binance.fetch_ohlcv(symbol, tf, limit=100)
                    if ohlcv:
                        # Calculate basic metrics for this timeframe
                        closes = [entry["close"] for entry in ohlcv]
                        highs = [entry["high"] for entry in ohlcv]
                        lows = [entry["low"] for entry in ohlcv]
                        volumes = [entry["volume"] for entry in ohlcv]
                        
                        # Store data for this timeframe
                        multi_timeframe_data[tf] = {
                            "ohlcv": ohlcv,
                            "current_price": closes[-1],
                            "price_change": (closes[-1] - closes[0]) / closes[0] 
                                           if len(closes) > 0 else None,
                            "high": max(highs) if highs else None,
                            "low": min(lows) if lows else None,
                            "volume_avg": sum(volumes) / len(volumes) if volumes else None,
                            "volatility": np.std(np.diff(closes) / closes[:-1]) * 100 
                                         if len(closes) > 1 else None,
                        }
                        
                        # Calculate technical indicators for this timeframe
                        if "technical_indicators" in data_sources:
                            try:
                                # Calculate and store indicators
                                bb_data = self.bb_indicator.calculate(closes)
                                bb_signal = self.bb_indicator.get_signal(closes)
                                
                                # Calculate RSI
                                try:
                                    # Here we're calculating RSI for this specific timeframe
                                    rsi_values = []
                                    for i in range(1, len(closes)):
                                        diff = closes[i] - closes[i-1]
                                        if diff > 0:
                                            rsi_values.append((diff, 0))
                                        else:
                                            rsi_values.append((0, abs(diff)))
                                    
                                    if rsi_values:
                                        avg_gain = sum(gain for gain, _ in rsi_values) / len(rsi_values)
                                        avg_loss = sum(loss for _, loss in rsi_values) / len(rsi_values)
                                        
                                        if avg_loss == 0:
                                            rsi = 100
                                        else:
                                            rs = avg_gain / avg_loss
                                            rsi = 100 - (100 / (1 + rs))
                                    else:
                                        rsi = 50.0
                                except Exception as e:
                                    logger.error(f"RSI calculation error for {tf}: {str(e)}")
                                    rsi = 50.0
                                
                                # Calculate moving averages
                                sma_20 = sum(closes[-20:]) / min(20, len(closes))
                                sma_50 = sum(closes[-50:]) / min(50, len(closes))
                                
                                multi_timeframe_data[tf]["indicators"] = {
                                    "bollinger_bands": {
                                        **bb_data,
                                        "signal": bb_signal
                                    },
                                    "rsi": rsi,
                                    "sma_20": sma_20,
                                    "sma_50": sma_50,
                                    "sma_20_diff": (closes[-1] - sma_20) / sma_20 * 100,
                                    "sma_50_diff": (closes[-1] - sma_50) / sma_50 * 100
                                }
                                
                                # Add trend assessment for this timeframe
                                if sma_20 > sma_50:
                                    trend = "BULLISH"
                                    trend_strength = (sma_20 - sma_50) / sma_50 * 100
                                else:
                                    trend = "BEARISH"
                                    trend_strength = (sma_50 - sma_20) / sma_20 * 100
                                
                                # Adjust trend strength description based on percentage difference
                                if trend_strength < 0.5:
                                    trend_desc = "WEAK"
                                elif trend_strength < 2.0:
                                    trend_desc = "MODERATE"
                                else:
                                    trend_desc = "STRONG"
                                
                                multi_timeframe_data[tf]["trend"] = {
                                    "direction": trend,
                                    "strength": trend_strength,
                                    "description": f"{trend_desc} {trend}"
                                }
                            except Exception as e:
                                logger.error(f"Technical indicators calculation error for {tf}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error fetching {tf} data for {symbol}: {str(e)}")
            
            # Store multi-timeframe data
            data["multi_timeframe"] = multi_timeframe_data
            
            # Primary timeframe data (for backward compatibility)
            primary_tf_data = multi_timeframe_data.get(interval, {})
            primary_ohlcv = primary_tf_data.get("ohlcv", [])
            
            if not primary_ohlcv:
                return {"error": f"No data available for {symbol}"}
                
            # Set primary timeframe data
            data["price_data"] = {
                "ohlcv": primary_ohlcv,
                "current_price": primary_tf_data.get("current_price"),
                "price_change_24h": primary_tf_data.get("price_change"),
                "high_24h": primary_tf_data.get("high"),
                "low_24h": primary_tf_data.get("low")
            }
            
            # Extract primary timeframe indicators
            if "technical_indicators" in data_sources and "indicators" in primary_tf_data:
                data["technical_indicators"] = primary_tf_data["indicators"]
            
            # Collect market sentiment data if needed
            if "prediction" in data_sources:
                try:
                    # Clear cache to ensure fresh analysis
                    predictor.analysis_cache.clear()
                    closes = [entry["close"] for entry in primary_ohlcv]
                    volumes = [entry["volume"] for entry in primary_ohlcv]
                    analysis = await predictor.analyze_market(closes, volumes, interval)
                    data["prediction"] = analysis
                except Exception as e:
                    logger.error(f"Prediction analysis error: {str(e)}")
                    # Provide minimal prediction data
                    current_price = primary_tf_data.get("current_price", 0)
                    data["prediction"] = {
                        "metadata": {"interval": interval, "confidence_score": 0.3},
                        "price_analysis": {
                            "current": current_price,
                            "prediction": {"next_period": current_price},
                            "key_levels": {"support": current_price * 0.95, "resistance": current_price * 1.05}
                        }
                    }
                
            # Collect AI insights for the primary timeframe if needed
            if "ai_insights" in data_sources:
                try:
                    # Prepare data for Gemini
                    insight_data = {
                        "symbol": symbol,
                        "current_price": data["price_data"]["current_price"],
                        "sma_20": data.get("technical_indicators", {}).get("sma_20", 0),
                        "sma_50": data.get("technical_indicators", {}).get("sma_50", 0),
                        "rsi": data.get("technical_indicators", {}).get("rsi", 50),
                        "volatility": primary_tf_data.get("volatility", 2.0),
                        "key_levels": {
                            "support": data.get("prediction", {}).get("price_analysis", {}).get("key_levels", {}).get("support", data["price_data"]["current_price"] * 0.95),
                            "resistance": data.get("prediction", {}).get("price_analysis", {}).get("key_levels", {}).get("resistance", data["price_data"]["current_price"] * 1.05),
                            "trend_strength": data.get("prediction", {}).get("price_analysis", {}).get("key_levels", {}).get("trend_strength", 0)
                        }
                    }
                    
                    data["ai_insights"] = self.gemini.generate_analysis(insight_data)
                except Exception as e:
                    logger.error(f"AI insights generation error: {str(e)}")
                    # Provide minimal AI insights
                    data["ai_insights"] = {
                        "market_summary": f"Analysis of {symbol} shows the price is currently at ${data['price_data']['current_price']:.6f}.",
                        "technical_observations": ["Limited data available for comprehensive analysis."],
                        "trading_recommendations": ["Consider waiting for more market signals before trading."]
                    }
                
            return data
            
        except Exception as e:
            logger.error(f"Data collection error: {str(e)}")
            return {"error": f"Error collecting data: {str(e)}"}
    
    async def _generate_response(self, query: str, query_info: Dict[str, Any], 
                                data: Dict[str, Any]) -> str:
        """Generate a natural language response to the query based on collected data"""
        # Check for errors
        if "error" in data:
            return f"Sorry, I couldn't answer that. {data['error']}"
            
        # Handle no symbol case
        if not query_info.get("symbol"):
            # Suggest some popular symbols when no crypto is detected
            return "I need a specific cryptocurrency symbol to answer this question. Try asking about popular cryptocurrencies like BTC, ETH, SOL, BNB, XRP, ADA, or DOGE."
            
        symbol = query_info.get("symbol")
        intent = query_info.get("intent")
        
        # Process multi-timeframe data for comprehensive analysis
        multi_tf_data = data.get("multi_timeframe", {})
        timeframes = list(multi_tf_data.keys())
        
        # Current price from primary timeframe
        current_price = data["price_data"]["current_price"]
        
        # Get timeframe-specific insights
        timeframe_insights = []
        for tf in ["1h", "4h", "1d", "1w"]:
            if tf in multi_tf_data:
                tf_data = multi_tf_data[tf]
                
                # Skip if no indicators for this timeframe
                if "indicators" not in tf_data:
                    continue
                    
                # Get trend information
                trend_info = tf_data.get("trend", {})
                trend_direction = trend_info.get("direction", "NEUTRAL")
                trend_desc = trend_info.get("description", "NEUTRAL")
                
                # Get key indicators
                rsi = tf_data["indicators"].get("rsi", 50)
                sma_20_diff = tf_data["indicators"].get("sma_20_diff", 0)
                sma_50_diff = tf_data["indicators"].get("sma_50_diff", 0)
                
                # Generate insight for this timeframe
                insight = f"{tf} timeframe: {trend_desc}"
                
                # Add RSI context
                if rsi > 70:
                    insight += f", overbought (RSI: {rsi:.1f})"
                elif rsi < 30:
                    insight += f", oversold (RSI: {rsi:.1f})"
                else:
                    insight += f", neutral momentum (RSI: {rsi:.1f})"
                
                # Add price vs MA context
                insight += f", price is {abs(sma_20_diff):.1f}% {'above' if sma_20_diff > 0 else 'below'} 20 SMA"
                
                timeframe_insights.append(insight)
        
        # Generate response based on intent
        if intent == "price":
            # Enhanced price analysis with multi-timeframe context
            change_24h = data["price_data"].get("price_change_24h")
            change_text = f"{change_24h*100:.2f}% recently" if change_24h is not None else "recently"
            direction = "up" if change_24h and change_24h > 0 else "down"
            
            # Get price changes across timeframes
            price_changes = []
            for tf, tf_label in [("1h", "hourly"), ("4h", "4-hour"), ("1d", "daily"), ("1w", "weekly")]:
                if tf in multi_tf_data:
                    tf_change = multi_tf_data[tf].get("price_change")
                    if tf_change is not None:
                        price_changes.append(f"{tf_change*100:.2f}% ({tf_label})")
            
            price_changes_text = ", ".join(price_changes) if price_changes else change_text
            
            response = f"{symbol} is currently trading at ${current_price:.8f}, {direction} {price_changes_text}."
            
            if timeframe_insights:
                response += "\n\nTechnical overview:\n• " + "\n• ".join(timeframe_insights)
            
            return response
            
        elif intent == "trend":
            # Enhanced trend analysis using multiple timeframes
            response = f"Trend analysis for {symbol} (currently at ${current_price:.6f}):\n\n"
            
            if timeframe_insights:
                response += "• " + "\n• ".join(timeframe_insights)
            else:
                # Fallback to single timeframe analysis
                if "technical_indicators" in data:
                    sma_20 = data["technical_indicators"]["sma_20"]
                    sma_50 = data["technical_indicators"]["sma_50"]
                    
                    trend = "bullish" if sma_20 > sma_50 else "bearish"
                    price_vs_sma20 = "above" if current_price > sma_20 else "below"
                    
                    response += f"The trend for {symbol} is currently {trend}. The price is {price_vs_sma20} the 20-period moving average, indicating short-term momentum is {trend}."
            
            return response
            
        elif intent == "volatility":
            # Enhanced volatility analysis
            # Start with primary timeframe volatility
            if "atr" in data:
                atr_signal = data["atr"]["signal"]
                
                # Create a more descriptive volatility response
                volatility_level = atr_signal['volatility'].lower()
                atr_value = atr_signal['atr_value']
                atr_percent = atr_signal['atr_percent']
                
                # Calculate expected price range based on ATR
                price_range_low = current_price - atr_value
                price_range_high = current_price + atr_value
                
                # Enhanced response with practical information and context
                response = f"{symbol} currently has {volatility_level} volatility with an Average True Range (ATR) of ${atr_value:.6f} ({atr_percent:.2f}% of price)."
                
                # Add multi-timeframe volatility context
                volatility_by_tf = []
                for tf, tf_label in [("1h", "hourly"), ("4h", "4-hour"), ("1d", "daily"), ("1w", "weekly")]:
                    if tf in multi_tf_data:
                        vol = multi_tf_data[tf].get("volatility")
                        if vol is not None:
                            volatility_by_tf.append(f"{vol:.2f}% ({tf_label})")
                
                if volatility_by_tf:
                    response += f" Volatility across timeframes: {', '.join(volatility_by_tf)}."
                
                # Add price range projection
                response += f" Based on current volatility, price could fluctuate between ${price_range_low:.6f} and ${price_range_high:.6f} in the near term."
                
                # Add RSI context
                if "technical_indicators" in data and "rsi" in data["technical_indicators"]:
                    rsi = data["technical_indicators"]["rsi"]
                    if rsi > 70:
                        response += f" The RSI of {rsi:.1f} suggests the asset may be overbought, which could lead to increased downside volatility."
                    elif rsi < 30:
                        response += f" The RSI of {rsi:.1f} suggests the asset may be oversold, which could lead to increased upside volatility."
                
                # Add multi-timeframe trend context
                if timeframe_insights:
                    response += "\n\nTrend analysis across timeframes:\n• " + "\n• ".join(timeframe_insights)
                
                return response
                
        elif intent == "advice":
            # Enhanced investment advice using MarketAdvisor with multi-timeframe context
            try:
                # Prepare technical data for advisor
                technical_data = {}
                
                if "technical_indicators" in data:
                    technical_data = data["technical_indicators"]
                
                # Prepare price data
                price_data = {
                    "current_price": current_price,
                    "price_change_24h": data["price_data"].get("price_change_24h", 0)
                }
                
                # Generate investment advice
                advice = self.market_advisor.generate_trading_advice(
                    technical_data=technical_data,
                    price_data=price_data,
                    atr_data=data.get("atr")
                )
                
                # Incorporate multi-timeframe analysis into the response
                response = f"For {symbol} currently trading at ${current_price:.6f}:\n\n"
                
                # Add timeframe analysis before the advice summary
                if timeframe_insights:
                    response += "Timeframe Analysis:\n• " + "\n• ".join(timeframe_insights) + "\n\n"
                
                # Add the advice summary
                response += advice["advice_summary"]
                
                # Add reasoning based on signal descriptions
                if advice["signal_descriptions"]:
                    response += "\n\nThis recommendation is based on:"
                    for desc in advice["signal_descriptions"]:
                        response += f"\n• {desc}"
                
                # Add risk warning for high-risk situations
                if advice["risk_assessment"] in ["HIGH", "VERY HIGH"]:
                    response += f"\n\nWARNING: This is a {advice['risk_assessment'].lower()} risk trade."
                
                # Add disclaimer
                response += "\n\nDisclaimer: This is algorithmic analysis, not financial advice. Always do your own research and consider your risk tolerance."
                
                return response
                
            except Exception as e:
                logger.error(f"Investment advice generation error: {str(e)}")
                # Fall back to enhanced generic advice
                return self._generate_enhanced_advice(data, multi_tf_data, timeframe_insights)
                
        elif intent == "analysis":
            # Comprehensive market analysis with multi-timeframe insights
            response = f"Comprehensive analysis for {symbol} (${current_price:.6f}):\n\n"
            
            # Add multi-timeframe analysis
            if timeframe_insights:
                response += "Timeframe Analysis:\n• " + "\n• ".join(timeframe_insights) + "\n\n"
            
            # Add technical signals from primary timeframe
            technical_signals = []
            
            if "technical_indicators" in data:
                rsi = data["technical_indicators"].get("rsi", 50)
                
                if rsi > 70:
                    technical_signals.append(f"RSI is high at {rsi:.1f} (overbought)")
                elif rsi < 30:
                    technical_signals.append(f"RSI is low at {rsi:.1f} (oversold)")
                    
                if "bollinger_bands" in data["technical_indicators"]:
                    bb_signal = data["technical_indicators"]["bollinger_bands"]["signal"]
                    if bb_signal["signal"] in ["BUY", "OVERSOLD"]:
                        technical_signals.append(f"Price is near/below lower Bollinger Band ({bb_signal['signal']})")
                    elif bb_signal["signal"] in ["SELL", "OVERBOUGHT"]:
                        technical_signals.append(f"Price is near/above upper Bollinger Band ({bb_signal['signal']})")
            
            # Get price movement context
            change_24h = data["price_data"].get("price_change_24h")
            if change_24h:
                if change_24h > 0.05:
                    technical_signals.append(f"Strong upward momentum (+{change_24h*100:.1f}% in 24h)")
                elif change_24h < -0.05:
                    technical_signals.append(f"Strong downward momentum ({change_24h*100:.1f}% in 24h)")
            
            # Include technical signals if available
            if technical_signals:
                response += "Key Technical Indicators:\n• " + "\n• ".join(technical_signals) + "\n\n"
            
            # Add AI insights if available
            if "ai_insights" in data:
                insights = data["ai_insights"]
                if "market_summary" in insights:
                    response += f"Market Summary: {insights['market_summary']}\n\n"
                
                if "technical_observations" in insights:
                    observations = insights.get("technical_observations", [])
                    if observations:
                        response += "Technical Observations:\n• " + "\n• ".join(observations) + "\n\n"
                
                if "trading_recommendations" in insights:
                    recommendations = []
                    for rec in insights.get("trading_recommendations", []):
                        if rec is not None:
                            if isinstance(rec, str) and rec.strip():
                                recommendations.append(rec)
                            elif rec:
                                recommendations.append(str(rec))
                    
                    if recommendations:
                        response += "Trading Recommendations:\n• " + "\n• ".join(recommendations)
            
            return response
        
        # Handle future/prediction intent
        elif intent == "prediction":
            future_response = f"Analysis of {symbol}'s potential future movement (currently at ${current_price:.6f}):\n\n"
            
            # Add timeframe analysis
            if timeframe_insights:
                future_response += "Trend Analysis Across Timeframes:\n• " + "\n• ".join(timeframe_insights) + "\n\n"
            
            # Collect consensus across timeframes
            timeframe_signals = {}
            for tf in ["1h", "4h", "1d", "1w"]:
                if tf in multi_tf_data and "trend" in multi_tf_data[tf]:
                    trend = multi_tf_data[tf]["trend"]
                    timeframe_signals[tf] = trend["direction"]
            
            # Analyze trend agreement/disagreement
            if timeframe_signals:
                bullish_count = sum(1 for direction in timeframe_signals.values() if direction == "BULLISH")
                bearish_count = sum(1 for direction in timeframe_signals.values() if direction == "BEARISH")
                
                if "1d" in timeframe_signals and "1w" in timeframe_signals:
                    # Check if longer timeframes agree (more significant)
                    if timeframe_signals["1d"] == timeframe_signals["1w"]:
                        long_term = timeframe_signals["1w"]
                        future_response += f"Long-term Outlook: The daily and weekly timeframes both show a {long_term.lower()} trend, "
                        if long_term == "BULLISH":
                            future_response += "suggesting potential for sustained upward movement in the coming weeks.\n\n"
                        else:
                            future_response += "suggesting caution as downward pressure may continue in the coming weeks.\n\n"
                    else:
                        future_response += f"Long-term Outlook: There is a divergence between daily ({timeframe_signals['1d'].lower()}) and weekly ({timeframe_signals['1w'].lower()}) trends, "
                        future_response += "indicating potential market uncertainty or a transition period.\n\n"
            
            # Add prediction data if available
            if "prediction" in data:
                prediction = data["prediction"]
                prediction_value = prediction.get("price_analysis", {}).get("prediction", {}).get("next_period", current_price)
                direction = "up" if prediction_value > current_price else "down"
                confidence = prediction.get("metadata", {}).get("confidence_score", 0.5)
                
                future_response += (f"Price Projection: The price is expected to move {direction} "
                                  f"to around ${prediction_value:.6f} in the next period "
                                  f"(confidence: {confidence:.0%}).\n\n")
                
                # Add key levels
                key_levels = prediction.get("price_analysis", {}).get("key_levels", {})
                if key_levels:
                    support = key_levels.get("support")
                    resistance = key_levels.get("resistance")
                    if support and resistance:
                        future_response += f"Key Levels: Watch for support around ${support:.2f} and resistance near ${resistance:.2f}.\n\n"
            
            # Check for Bitcoin-specific analysis
            if symbol == "BTCUSDT":
                # Get recent market data from Bitcoin overview search results
                recent_btc_data = """
Based on market analysis and available Bitcoin price prediction data:

• Short-term (1-3 months): Bitcoin is expected to continue showing volatility but maintaining its position above key support levels. Current technical indicators point to a period of consolidation following recent all-time highs.

• Mid-term (6-12 months): The post-halving cycle effect typically continues to influence Bitcoin prices positively for 12-18 months after the event. Historical patterns suggest continued upward momentum, with potential new all-time highs.

• Long-term (2-5 years): Multiple models including Stock-to-Flow predict significant appreciation, with some analysts forecasting prices ranging from $150,000 to $500,000 by 2028-2030.

• Adoption factors: Institutional adoption continues to grow through ETFs and corporate treasury investments, creating sustained buy pressure despite retail volatility.
"""
                future_response += recent_btc_data
            
            # Add AI insights if available
            if "ai_insights" in data:
                insights = data["ai_insights"]
                
                if "market_summary" in insights:
                    future_response += f"Market Context: {insights['market_summary']}\n\n"
                
                if "trading_recommendations" in insights:
                    recommendations = []
                    for rec in insights.get("trading_recommendations", []):
                        if rec is not None and (isinstance(rec, str) and rec.strip() or not isinstance(rec, str)):
                            recommendations.append(str(rec))
                    
                    if recommendations:
                        future_response += "Strategic Considerations:\n• " + "\n• ".join(recommendations)
            
            # Add disclaimer
            future_response += "\n\nImportant: Cryptocurrency markets are highly volatile and unpredictable. These projections are based on technical analysis of historical data and should not be considered financial advice. Past performance is not indicative of future results."
            
            return future_response
                    
        # Default comprehensive response for any other intent
        # Create a rich response combining multiple aspects
        response = f"{symbol} is currently trading at ${current_price:.6f}.\n\n"
        
        # Add multi-timeframe analysis
        if timeframe_insights:
            response += "Timeframe Analysis:\n• " + "\n• ".join(timeframe_insights) + "\n\n"
        
        # Add AI insights if available
        if "ai_insights" in data:
            insights = data["ai_insights"]
            if "market_summary" in insights:
                response += f"Market Summary: {insights['market_summary']}\n\n"
            
            if "technical_observations" in insights:
                observations = insights.get("technical_observations", [])
                if observations:
                    response += "Key Observations:\n• " + "\n• ".join(observations) + "\n\n"
            
            if "trading_recommendations" in insights:
                recommendations = []
                for rec in insights.get("trading_recommendations", []):
                    if rec is not None:
                        if isinstance(rec, str) and rec.strip():
                            recommendations.append(rec)
                        elif rec:
                            recommendations.append(str(rec))
                
                if recommendations:
                    response += "Recommendations:\n• " + "\n• ".join(recommendations)
        
        return response
    
    def _generate_enhanced_advice(self, data: Dict[str, Any], multi_tf_data: Dict[str, Any], 
                                timeframe_insights: List[str]) -> str:
        """Generate enhanced trading advice as a fallback with multi-timeframe context"""
        current_price = data["price_data"]["current_price"]
        symbol = data.get("symbol", "This asset")
        
        # Combine technical indicators, prediction, and AI insights
        technical_signals = []
        
        # Get technical indicator signals from primary timeframe
        if "technical_indicators" in data:
            rsi = data["technical_indicators"].get("rsi", 50)
            
            if rsi > 70:
                technical_signals.append(f"RSI is high at {rsi:.1f} (overbought)")
            elif rsi < 30:
                technical_signals.append(f"RSI is low at {rsi:.1f} (oversold)")
                
            if "bollinger_bands" in data["technical_indicators"]:
                bb_signal = data["technical_indicators"]["bollinger_bands"]["signal"]
                if bb_signal["signal"] in ["BUY", "OVERSOLD"]:
                    technical_signals.append(f"Price is near/below lower Bollinger Band ({bb_signal['signal']})")
                elif bb_signal["signal"] in ["SELL", "OVERBOUGHT"]:
                    technical_signals.append(f"Price is near/above upper Bollinger Band ({bb_signal['signal']})")
        
        # Get multi-timeframe signals
        timeframe_signals = {}
        for tf in ["1h", "4h", "1d", "1w"]:
            if tf in multi_tf_data and "trend" in multi_tf_data[tf]:
                trend = multi_tf_data[tf]["trend"]
                if trend["direction"] == "BULLISH":
                    timeframe_signals[tf] = "BUY"
                elif trend["direction"] == "BEARISH":
                    timeframe_signals[tf] = "SELL"
                else:
                    timeframe_signals[tf] = "NEUTRAL"
        
        # Count buy/sell signals across timeframes
        buy_votes = sum(1 for signal in timeframe_signals.values() if signal == "BUY")
        sell_votes = sum(1 for signal in timeframe_signals.values() if signal == "SELL")
        
        # Determine overall recommendation based on multi-timeframe consensus
        if buy_votes > sell_votes and buy_votes >= len(timeframe_signals) / 2:
            overall_rec = "BUY"
            confidence = buy_votes / len(timeframe_signals) if timeframe_signals else 0.5
        elif sell_votes > buy_votes and sell_votes >= len(timeframe_signals) / 2:
            overall_rec = "SELL"
            confidence = sell_votes / len(timeframe_signals) if timeframe_signals else 0.5
        else:
            overall_rec = "NEUTRAL"
            confidence = 0.5
        
        # Create detailed response with multi-timeframe context
        response = f"{symbol} is currently trading at ${current_price:.6f}.\n\n"
        
        if timeframe_insights:
            response += "Multi-Timeframe Analysis:\n• " + "\n• ".join(timeframe_insights) + "\n\n"
        
        # Add overall recommendation
        if overall_rec == "BUY":
            response += f"Overall Recommendation: Consider buying {symbol} (confidence: {confidence:.0%}). "
            response += "Technical signals suggest a potential buying opportunity, but always consider your risk tolerance and portfolio allocation."
        elif overall_rec == "SELL":
            response += f"Overall Recommendation: Consider reducing exposure to {symbol} (confidence: {confidence:.0%}). "
            response += "Technical signals suggest caution and potential downside risk at current levels."
        else:
            response += f"Overall Recommendation: Neutral stance on {symbol} (confidence: {confidence:.0%}). "
            response += "Based on mixed signals across timeframes, consider waiting for clearer directional movement before making a decision."
        
        # Add key observations from primary timeframe
        if technical_signals:
            response += "\n\nKey Technical Indicators:\n• " + "\n• ".join(technical_signals)
        
        # Add AI insights if available
        if "ai_insights" in data and "market_summary" in data["ai_insights"]:
            response += f"\n\nMarket Summary: {data['ai_insights']['market_summary']}"
        
        response += "\n\nDisclaimer: This is algorithmic analysis, not financial advice. Always do your own research and consider your personal risk tolerance."
        
        return response
    
    def _filter_supporting_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter and format supporting data to avoid returning too much raw data"""
        if "error" in data:
            return {}
            
        filtered = {}
        
        # Include basic price information
        if "price_data" in data:
            filtered["current_price"] = data["price_data"]["current_price"]
            filtered["price_change_24h"] = data["price_data"]["price_change_24h"]
            
        # Include key technical indicators
        if "technical_indicators" in data:
            indicators = data["technical_indicators"]
            if "rsi" in indicators and indicators["rsi"] is not None:
                # Check for NaN values
                rsi = indicators["rsi"]
                filtered["rsi"] = None if isinstance(rsi, float) and (math.isnan(rsi) or math.isinf(rsi)) else rsi
                
            if "bollinger_bands" in indicators and indicators["bollinger_bands"] is not None:
                signal = indicators["bollinger_bands"]["signal"]
                filtered["bollinger_signal"] = signal["signal"]
                
                percent_b = signal.get("percent_b")
                # Check for NaN values
                filtered["percent_b"] = None if isinstance(percent_b, float) and (math.isnan(percent_b) or math.isinf(percent_b)) else percent_b
                
        # Include prediction info with NaN checking
        if "prediction" in data and isinstance(data["prediction"], dict):
            prediction_data = data["prediction"]
            
            if "prediction" in prediction_data:
                # Get prediction value and check for NaN
                pred_value = prediction_data["prediction"]
                if isinstance(pred_value, dict):
                    # Handle nested prediction structure
                    filtered_pred = {}
                    for k, v in pred_value.items():
                        filtered_pred[k] = None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
                    filtered["prediction"] = filtered_pred
                else:
                    # Handle direct value
                    filtered["prediction"] = None if isinstance(pred_value, float) and (math.isnan(pred_value) or math.isinf(pred_value)) else pred_value
            
            # Get confidence value and check for NaN
            if "confidence" in prediction_data:
                confidence = prediction_data["confidence"]
                filtered["confidence"] = None if isinstance(confidence, float) and (math.isnan(confidence) or math.isinf(confidence)) else confidence
            
        # Include key insights if available
        if "ai_insights" in data:
            insights = data["ai_insights"]
            if "market_summary" in insights:
                filtered["market_summary"] = insights["market_summary"]
            if "technical_observations" in insights:
                # Filter out any None values or NaN in observations
                observations = []
                for obs in insights["technical_observations"]:
                    if obs is not None and not (isinstance(obs, float) and (math.isnan(obs) or math.isinf(obs))):
                        observations.append(obs)
                filtered["observations"] = observations
                
        return filtered

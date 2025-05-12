import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger("CryptoPredictAPI")

class MarketAdvisor:
    """Advanced market analysis system that generates detailed buy/sell advice"""
    
    def __init__(self):
        """Initialize the market advisor"""
        self.confidence_thresholds = {
            "very_low": 0.2,
            "low": 0.35,
            "medium": 0.5,
            "high": 0.65,
            "very_high": 0.8
        }
        
    def generate_trading_advice(self, 
                               technical_data: Dict, 
                               price_data: Dict, 
                               atr_data: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive trading advice based on multiple indicators
        
        Args:
            technical_data: Dictionary containing technical indicators
            price_data: Dictionary containing price data
            atr_data: Optional dictionary containing ATR data
            
        Returns:
            Dictionary with trading advice and supporting data
        """
        # Extract key data points
        current_price = price_data.get("current_price", 0)
        price_change_24h = price_data.get("price_change_24h", 0)
        
        # Default values for missing data
        rsi = technical_data.get("rsi", 50)
        sma_20 = technical_data.get("sma_20", current_price)
        sma_50 = technical_data.get("sma_50", current_price)
        
        # Bollinger bands data if available
        bb_data = technical_data.get("bollinger_bands", {})
        bb_signal = bb_data.get("signal", {})
        percent_b = bb_signal.get("percent_b", 0.5)
        
        # ATR data if available
        volatility = "MODERATE"
        atr_percent = 2.0
        market_phase = {"phase": "UNDEFINED", "confidence": 0.0}
        
        if atr_data and "signal" in atr_data:
            atr_signal = atr_data.get("signal", {})
            volatility = atr_signal.get("volatility", "MODERATE")
            atr_percent = atr_signal.get("atr_percent", 2.0)
            market_phase = atr_signal.get("market_phase", market_phase)
        
        # Generate separate signals from each indicator
        signals = self._generate_indicator_signals(
            current_price=current_price,
            rsi=rsi,
            sma_20=sma_20,
            sma_50=sma_50,
            percent_b=percent_b,
            price_change_24h=price_change_24h,
            volatility=volatility,
            atr_percent=atr_percent,
            market_phase=market_phase
        )
        
        # Calculate overall signal strength and direction
        signal_strength, signal_direction = self._calculate_overall_signal(signals)
        
        # Generate entry/exit price targets
        entry_targets, exit_targets, stop_loss = self._calculate_price_targets(
            current_price=current_price,
            signal_direction=signal_direction,
            atr_percent=atr_percent,
            volatility=volatility,
            price_change_24h=price_change_24h
        )
        
        # Generate confidence level and descriptions
        confidence_level = self._determine_confidence_level(signal_strength)
        signal_descriptions = self._generate_signal_descriptions(signals, signal_direction)
        
        # Generate time horizon recommendation
        time_horizon = self._recommend_time_horizon(
            signal_strength=signal_strength, 
            volatility=volatility, 
            market_phase=market_phase.get("phase", "UNDEFINED")
        )
        
        # Generate risk assessment
        risk_assessment = self._assess_risk(
            volatility=volatility,
            atr_percent=atr_percent,
            signal_strength=signal_strength,
            rsi=rsi,
            market_phase=market_phase.get("phase", "UNDEFINED")
        )
        
        # Format the advice for display
        advice = self._format_advice(
            signal_direction=signal_direction,
            confidence_level=confidence_level,
            entry_targets=entry_targets,
            exit_targets=exit_targets,
            stop_loss=stop_loss,
            time_horizon=time_horizon,
            risk_assessment=risk_assessment
        )
        
        return {
            "advice_summary": advice,
            "signal_direction": signal_direction,
            "signal_strength": signal_strength,
            "confidence_level": confidence_level,
            "entry_targets": entry_targets,
            "exit_targets": exit_targets,
            "stop_loss": stop_loss,
            "time_horizon": time_horizon,
            "risk_assessment": risk_assessment,
            "signal_descriptions": signal_descriptions,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_indicator_signals(self, 
                                   current_price: float,
                                   rsi: float,
                                   sma_20: float,
                                   sma_50: float,
                                   percent_b: float,
                                   price_change_24h: float,
                                   volatility: str,
                                   atr_percent: float,
                                   market_phase: Dict) -> Dict[str, Tuple[float, str]]:
        """
        Generate signals from each individual indicator
        
        Returns:
            Dictionary mapping indicator name to (strength, direction) tuples
        """
        signals = {}
        
        # RSI signal
        if rsi > 70:
            signals["rsi"] = (min(1.0, (rsi - 70) / 30 + 0.5), "SELL")
        elif rsi < 30:
            signals["rsi"] = (min(1.0, (30 - rsi) / 30 + 0.5), "BUY")
        else:
            # Neutral zone with weaker signals
            if rsi > 60:
                signals["rsi"] = ((rsi - 60) / 10 * 0.3, "SELL")  # Weaker sell signal
            elif rsi < 40:
                signals["rsi"] = ((40 - rsi) / 10 * 0.3, "BUY")  # Weaker buy signal
            else:
                signals["rsi"] = (0.1, "NEUTRAL")
        
        # Moving averages signal
        if sma_20 > sma_50:
            # Bullish MA alignment
            ma_strength = min(1.0, (sma_20 - sma_50) / sma_50 * 20)  # Normalize strength
            signals["moving_averages"] = (ma_strength, "BUY")
        else:
            # Bearish MA alignment
            ma_strength = min(1.0, (sma_50 - sma_20) / sma_20 * 20)  # Normalize strength
            signals["moving_averages"] = (ma_strength, "SELL")
        
        # Price vs SMA signal
        if current_price > sma_20:
            price_sma_strength = min(1.0, (current_price - sma_20) / sma_20 * 10)
            signals["price_vs_sma"] = (price_sma_strength, "BUY")
        else:
            price_sma_strength = min(1.0, (sma_20 - current_price) / current_price * 10)
            signals["price_vs_sma"] = (price_sma_strength, "SELL")
        
        # Bollinger Bands signal
        if percent_b > 1.0:
            # Price above upper band - strong sell
            bb_strength = min(1.0, (percent_b - 1.0) * 2 + 0.5)
            signals["bollinger_bands"] = (bb_strength, "SELL")
        elif percent_b < 0.0:
            # Price below lower band - strong buy
            bb_strength = min(1.0, (0.0 - percent_b) * 2 + 0.5)
            signals["bollinger_bands"] = (bb_strength, "BUY")
        elif percent_b > 0.8:
            # Near upper band - weaker sell
            bb_strength = (percent_b - 0.8) * 5  # Scale to 0-1 range
            signals["bollinger_bands"] = (bb_strength, "SELL")
        elif percent_b < 0.2:
            # Near lower band - weaker buy
            bb_strength = (0.2 - percent_b) * 5  # Scale to 0-1 range
            signals["bollinger_bands"] = (bb_strength, "BUY")
        else:
            # Middle range - neutral
            signals["bollinger_bands"] = (0.1, "NEUTRAL")
        
        # Price momentum signal
        if price_change_24h > 0.05:
            # Strong positive momentum
            momentum_strength = min(1.0, price_change_24h * 5)
            signals["momentum"] = (momentum_strength, "BUY")
        elif price_change_24h < -0.05:
            # Strong negative momentum
            momentum_strength = min(1.0, abs(price_change_24h) * 5)
            signals["momentum"] = (momentum_strength, "SELL")
        else:
            # Weak or neutral momentum
            signals["momentum"] = (0.1, "NEUTRAL")
        
        # Market phase signal
        phase = market_phase.get("phase", "UNDEFINED")
        phase_confidence = market_phase.get("confidence", 0.0)
        
        if phase == "TRENDING_UP" and phase_confidence > 0.4:
            signals["market_phase"] = (phase_confidence, "BUY")
        elif phase == "TRENDING_DOWN" and phase_confidence > 0.4:
            signals["market_phase"] = (phase_confidence, "SELL")
        elif phase == "COILING" and phase_confidence > 0.6:
            # Coiling patterns suggest waiting for breakout
            signals["market_phase"] = (phase_confidence * 0.5, "NEUTRAL")
        elif phase == "CONSOLIDATION" and phase_confidence > 0.5:
            # Consolidations in bullish market tend to break up, bearish down
            if sma_20 > sma_50:
                signals["market_phase"] = (phase_confidence * 0.3, "BUY")
            else:
                signals["market_phase"] = (phase_confidence * 0.3, "SELL")
        else:
            signals["market_phase"] = (0.1, "NEUTRAL")
        
        # Volatility signal
        # High volatility increases uncertainty, but can also indicate potential entry/exit points
        volatility_signal = "NEUTRAL"
        volatility_strength = 0.1
        
        if volatility == "HIGH" or volatility == "VERY HIGH":
            # High volatility with strong trend suggests riding the trend
            if phase == "TRENDING_UP":
                volatility_signal = "BUY"
                volatility_strength = min(1.0, atr_percent / 10)
            elif phase == "TRENDING_DOWN":
                volatility_signal = "SELL"
                volatility_strength = min(1.0, atr_percent / 10)
        
        signals["volatility"] = (volatility_strength, volatility_signal)
        
        return signals
    
    def _calculate_overall_signal(self, signals: Dict[str, Tuple[float, str]]) -> Tuple[float, str]:
        """
        Calculate overall signal strength and direction
        
        Args:
            signals: Dictionary mapping indicator name to (strength, direction) tuples
            
        Returns:
            Tuple of (overall_strength, overall_direction)
        """
        if not signals:
            return 0.0, "NEUTRAL"
            
        # Indicator weights for weighted average
        weights = {
            "rsi": 0.15,
            "moving_averages": 0.2,
            "price_vs_sma": 0.1,
            "bollinger_bands": 0.15,
            "momentum": 0.15,
            "market_phase": 0.15,
            "volatility": 0.1
        }
        
        # Calculate weighted buy and sell signals
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        
        for indicator, (strength, direction) in signals.items():
            weight = weights.get(indicator, 0.1)
            total_weight += weight
            
            if direction == "BUY":
                buy_score += strength * weight
            elif direction == "SELL":
                sell_score += strength * weight
        
        # Normalize by total weight
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine overall direction and strength
        if buy_score > sell_score:
            overall_direction = "BUY"
            overall_strength = buy_score - (sell_score / 2)  # Subtracting half the sell score to reflect counter signals
        elif sell_score > buy_score:
            overall_direction = "SELL"
            overall_strength = sell_score - (buy_score / 2)  # Subtracting half the buy score to reflect counter signals
        else:
            overall_direction = "NEUTRAL"
            overall_strength = 0.1
        
        # Ensure strength is in 0-1 range
        overall_strength = max(0.1, min(1.0, overall_strength))
        
        return overall_strength, overall_direction
    
    def _calculate_price_targets(self, 
                               current_price: float,
                               signal_direction: str,
                               atr_percent: float,
                               volatility: str,
                               price_change_24h: float) -> Tuple[List[float], List[float], float]:
        """
        Calculate entry, exit and stop loss price targets
        
        Returns:
            Tuple of (entry_targets, exit_targets, stop_loss)
        """
        # Calculate ATR in price terms
        atr_value = current_price * (atr_percent / 100)
        
        # Set multipliers based on volatility
        if volatility == "VERY HIGH":
            entry_multipliers = [0.5, 1.0, 2.0]
            exit_multipliers = [1.0, 2.0, 3.0]
            stop_multiplier = 2.0
        elif volatility == "HIGH":
            entry_multipliers = [0.5, 1.0, 1.5]
            exit_multipliers = [1.0, 1.5, 2.5]
            stop_multiplier = 1.5
        elif volatility == "MODERATE":
            entry_multipliers = [0.3, 0.7, 1.2]
            exit_multipliers = [0.7, 1.2, 2.0]
            stop_multiplier = 1.2
        else:  # LOW or VERY LOW
            entry_multipliers = [0.2, 0.5, 1.0]
            exit_multipliers = [0.5, 1.0, 1.5]
            stop_multiplier = 1.0
        
        # Adjust for recent price momentum
        if abs(price_change_24h) > 0.1:  # Strong momentum
            momentum_factor = 1.2
        elif abs(price_change_24h) > 0.05:  # Moderate momentum
            momentum_factor = 1.1
        else:
            momentum_factor = 1.0
            
        # Apply momentum adjustment
        exit_multipliers = [m * momentum_factor for m in exit_multipliers]
        
        # Calculate price targets based on signal direction
        if signal_direction == "BUY":
            # For buy signals, entry targets are below current price, exit targets above
            entry_targets = [round(current_price * (1 - m * atr_percent / 100), 8) for m in entry_multipliers]
            exit_targets = [round(current_price * (1 + m * atr_percent / 100), 8) for m in exit_multipliers]
            stop_loss = round(current_price * (1 - stop_multiplier * atr_percent / 100), 8)
        elif signal_direction == "SELL":
            # For sell signals, entry targets are above current price, exit targets below
            entry_targets = [round(current_price * (1 + m * atr_percent / 100), 8) for m in entry_multipliers]
            exit_targets = [round(current_price * (1 - m * atr_percent / 100), 8) for m in exit_multipliers]
            stop_loss = round(current_price * (1 + stop_multiplier * atr_percent / 100), 8)
        else:
            # For neutral signals, provide modest price targets in both directions
            entry_targets = [
                round(current_price * (1 - 0.5 * atr_percent / 100), 8),
                current_price,
                round(current_price * (1 + 0.5 * atr_percent / 100), 8)
            ]
            exit_targets = [
                round(current_price * (1 - 1.0 * atr_percent / 100), 8),
                current_price,
                round(current_price * (1 + 1.0 * atr_percent / 100), 8)
            ]
            stop_loss = round(current_price * (1 - 1.5 * atr_percent / 100), 8)
        
        return entry_targets, exit_targets, stop_loss
    
    def _determine_confidence_level(self, signal_strength: float) -> str:
        """Convert signal strength to a descriptive confidence level"""
        if signal_strength >= self.confidence_thresholds["very_high"]:
            return "VERY HIGH"
        elif signal_strength >= self.confidence_thresholds["high"]:
            return "HIGH"
        elif signal_strength >= self.confidence_thresholds["medium"]:
            return "MEDIUM"
        elif signal_strength >= self.confidence_thresholds["low"]:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _generate_signal_descriptions(self, signals: Dict[str, Tuple[float, str]], 
                                    overall_direction: str) -> List[str]:
        """
        Generate human-readable descriptions of the signals
        
        Returns:
            List of signal descriptions
        """
        descriptions = []
        
        # Process signals in order of strength
        sorted_signals = sorted(
            [(indicator, strength, direction) for indicator, (strength, direction) in signals.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top supporting signals
        supporting_signals = [s for s in sorted_signals if s[2] == overall_direction]
        opposing_signals = [s for s in sorted_signals if s[2] != overall_direction and s[2] != "NEUTRAL"]
        
        # Add descriptions for top 3 supporting signals
        for indicator, strength, direction in supporting_signals[:3]:
            if indicator == "rsi" and direction == "BUY":
                descriptions.append(f"RSI indicates oversold conditions ({signals['rsi'][0]:.0%} strength)")
            elif indicator == "rsi" and direction == "SELL":
                descriptions.append(f"RSI indicates overbought conditions ({signals['rsi'][0]:.0%} strength)")
            elif indicator == "moving_averages" and direction == "BUY":
                descriptions.append(f"Moving averages show bullish alignment ({signals['moving_averages'][0]:.0%} strength)")
            elif indicator == "moving_averages" and direction == "SELL":
                descriptions.append(f"Moving averages show bearish alignment ({signals['moving_averages'][0]:.0%} strength)")
            elif indicator == "bollinger_bands" and direction == "BUY":
                descriptions.append(f"Price is near/below lower Bollinger Band ({signals['bollinger_bands'][0]:.0%} strength)")
            elif indicator == "bollinger_bands" and direction == "SELL":
                descriptions.append(f"Price is near/above upper Bollinger Band ({signals['bollinger_bands'][0]:.0%} strength)")
            elif indicator == "momentum" and direction == "BUY":
                descriptions.append(f"Strong positive price momentum ({signals['momentum'][0]:.0%} strength)")
            elif indicator == "momentum" and direction == "SELL":
                descriptions.append(f"Strong negative price momentum ({signals['momentum'][0]:.0%} strength)")
            elif indicator == "market_phase" and direction == "BUY":
                descriptions.append(f"Market is trending upward ({signals['market_phase'][0]:.0%} strength)")
            elif indicator == "market_phase" and direction == "SELL":
                descriptions.append(f"Market is trending downward ({signals['market_phase'][0]:.0%} strength)")
        
        # Add top opposing signal with caveat
        if opposing_signals:
            indicator, strength, direction = opposing_signals[0]
            if strength > 0.4:  # Only mention strong opposing signals
                if indicator == "rsi":
                    descriptions.append(f"Warning: RSI contradicts overall signal ({strength:.0%} strength)")
                elif indicator == "moving_averages":
                    descriptions.append(f"Warning: Moving averages contradict overall signal ({strength:.0%} strength)")
                elif indicator == "bollinger_bands":
                    descriptions.append(f"Warning: Bollinger Bands contradict overall signal ({strength:.0%} strength)")
                elif indicator == "momentum":
                    descriptions.append(f"Warning: Price momentum contradicts overall signal ({strength:.0%} strength)")
        
        return descriptions
    
    def _recommend_time_horizon(self, signal_strength: float, volatility: str, market_phase: str) -> str:
        """Recommend trading time horizon based on signal characteristics"""
        if signal_strength < self.confidence_thresholds["low"]:
            return "NOT RECOMMENDED"
            
        if volatility in ["VERY HIGH", "HIGH"]:
            if market_phase == "TRENDING_UP" or market_phase == "TRENDING_DOWN":
                return "SHORT TERM (0-3 days)"
            else:
                return "DAY TRADE"
        elif signal_strength > self.confidence_thresholds["high"]:
            if market_phase == "TRENDING_UP" or market_phase == "TRENDING_DOWN":
                return "MEDIUM TERM (1-2 weeks)"
            else:
                return "SHORT TERM (0-3 days)"
        elif signal_strength > self.confidence_thresholds["medium"]:
            return "SHORT TERM (0-3 days)"
        else:
            return "DAY TRADE"
    
    def _assess_risk(self, volatility: str, atr_percent: float, 
                    signal_strength: float, rsi: float, market_phase: str) -> str:
        """Assess trading risk level"""
        risk_factors = []
        
        # Volatility factor
        if volatility == "VERY HIGH":
            risk_factors.append("VERY HIGH VOLATILITY")
        elif volatility == "HIGH":
            risk_factors.append("HIGH VOLATILITY")
            
        # Extreme RSI factor
        if rsi > 80 or rsi < 20:
            risk_factors.append("EXTREME RSI")
            
        # Market phase factor
        if market_phase == "CHOPPY":
            risk_factors.append("CHOPPY MARKET")
        elif market_phase == "COILING":
            risk_factors.append("INCREASED VOLATILITY EXPECTED")
            
        # Low confidence factor
        if signal_strength < self.confidence_thresholds["medium"]:
            risk_factors.append("LOW SIGNAL CONFIDENCE")
            
        # Determine risk level
        if len(risk_factors) >= 3 or "VERY HIGH VOLATILITY" in risk_factors:
            return "VERY HIGH"
        elif len(risk_factors) >= 2 or "HIGH VOLATILITY" in risk_factors:
            return "HIGH"
        elif len(risk_factors) >= 1:
            return "MEDIUM"
        else:
            return "MODERATE"
    
    def _format_advice(self, signal_direction: str, confidence_level: str,
                     entry_targets: List[float], exit_targets: List[float],
                     stop_loss: float, time_horizon: str, risk_assessment: str) -> str:
        """Format the trading advice for display"""
        if signal_direction == "NEUTRAL" or confidence_level == "VERY LOW":
            return "HOLD/WAIT - Insufficient signals for clear trading direction."
            
        action = signal_direction
        entry_desc = ""
        exit_desc = ""
        
        # Format entry targets
        if len(entry_targets) >= 3:
            if signal_direction == "BUY":
                entry_desc = f"Entry targets: ${entry_targets[0]} (aggressive), ${entry_targets[1]} (moderate), ${entry_targets[2]} (conservative)"
            else:
                entry_desc = f"Entry targets: ${entry_targets[0]} (aggressive), ${entry_targets[1]} (moderate), ${entry_targets[2]} (conservative)"
                
        # Format exit targets
        if len(exit_targets) >= 3:
            if signal_direction == "BUY":
                exit_desc = f"Take profit targets: ${exit_targets[0]} (conservative), ${exit_targets[1]} (moderate), ${exit_targets[2]} (aggressive)"
            else:
                exit_desc = f"Take profit targets: ${exit_targets[0]} (conservative), ${exit_targets[1]} (moderate), ${exit_targets[2]} (aggressive)"
        
        # Format advice based on confidence
        if confidence_level in ["VERY HIGH", "HIGH"]:
            advice = f"{action} - {confidence_level} CONFIDENCE ({time_horizon})"
        else:
            advice = f"{action} WITH CAUTION - {confidence_level} CONFIDENCE ({time_horizon})"
            
        # Add risk warning for high risk
        if risk_assessment in ["VERY HIGH", "HIGH"]:
            advice += f" | {risk_assessment} RISK"
            
        return f"{advice}\n{entry_desc}\n{exit_desc}\nStop loss: ${stop_loss}"
        
class MarketComparisonAnalyzer:
    """Compares and benchmarks cryptocurrency performance against peers and market indices"""
    
    def __init__(self, binance_client=None):
        """Initialize the market comparison analyzer"""
        self.binance_client = binance_client
        
    async def compare_assets(self, primary_symbol: str, comparison_assets: List[str],
                     time_period: str = "7d") -> Dict:
        """
        Compare a cryptocurrency against other assets
        
        Args:
            primary_symbol: Main cryptocurrency symbol to analyze
            comparison_assets: List of other assets to compare against
            time_period: Time period for comparison (1d, 7d, 30d, etc)
            
        Returns:
            Dictionary with comparison results
        """
        if not self.binance_client:
            raise ValueError("Binance client is required for asset comparison")
            
        # Convert time period to number of candles
        period_to_candles = {
            "1d": 24,    # 1 day with 1h candles
            "3d": 72,    # 3 days with 1h candles
            "7d": 168,   # 7 days with 1h candles
            "14d": 336,  # 14 days with 1h candles
            "30d": 720   # 30 days with 1h candles
        }
        
        # Default to 7 days if period not recognized
        candles = period_to_candles.get(time_period, 168)
        
        # Use 1h interval for reasonable data granularity
        interval = "1h"
        
        # Add primary symbol to comparison list if not already included
        all_assets = [primary_symbol] + [asset for asset in comparison_assets if asset != primary_symbol]
        
        # Fetch data for all assets
        asset_data = {}
        
        for symbol in all_assets:
            try:
                # Fetch OHLCV data
                ohlcv = await self.binance_client.fetch_ohlcv(symbol, interval, limit=candles)
                
                if not ohlcv or len(ohlcv) < 5:  # Need at least a few candles
                    continue
                    
                # Extract price data
                closes = [entry["close"] for entry in ohlcv]
                volumes = [entry["volume"] for entry in ohlcv]
                
                # Calculate key metrics
                start_price = closes[0]
                end_price = closes[-1]
                percent_change = ((end_price - start_price) / start_price) * 100
                
                # Calculate volatility
                returns = np.diff(closes) / closes[:-1]
                volatility = float(np.std(returns) * 100)  # Convert to percentage
                
                # Calculate daily returns
                period_hours = len(closes)
                if period_hours >= 24:
                    daily_closes = [closes[i] for i in range(0, period_hours, 24)]
                    daily_returns = np.diff(daily_closes) / daily_closes[:-1]
                    avg_daily_return = float(np.mean(daily_returns) * 100) if len(daily_returns) > 0 else 0
                else:
                    avg_daily_return = percent_change / max(1, period_hours / 24)
                
                # Calculate strength metrics
                strength_index = percent_change / (volatility + 0.001)  # Avoid division by zero
                
                # Store asset data
                asset_data[symbol] = {
                    "start_price": start_price,
                    "end_price": end_price,
                    "percent_change": percent_change,
                    "volatility": volatility,
                    "avg_daily_return": avg_daily_return,
                    "strength_index": strength_index,
                    "data_points": len(closes)
                }
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                # Skip this asset
                continue
                
        # If no data was collected, return error
        if not asset_data:
            return {"error": "Could not collect data for any of the requested assets"}
            
        # Calculate rankings
        rankings = self._calculate_rankings(asset_data)
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(
            primary_symbol=primary_symbol,
            asset_data=asset_data,
            rankings=rankings,
            time_period=time_period
        )
        
        # Prepare the response
        result = {
            "primary_symbol": primary_symbol,
            "time_period": time_period,
            "comparison_assets": list(asset_data.keys()),
            "metrics": asset_data,
            "rankings": rankings,
            "comparison_summary": comparison_summary
        }
        
        return result
        
    def _calculate_rankings(self, asset_data: Dict) -> Dict:
        """Calculate rankings for different metrics"""
        if not asset_data:
            return {}
            
        rankings = {
            "percent_change": [],
            "volatility": [],
            "strength_index": []
        }
        
        # Sort assets by different metrics
        symbols = list(asset_data.keys())
        
        # Percent change ranking (higher is better)
        percent_change_sorted = sorted(symbols, key=lambda s: asset_data[s]["percent_change"], reverse=True)
        rankings["percent_change"] = percent_change_sorted
        
        # Volatility ranking (lower is better)
        volatility_sorted = sorted(symbols, key=lambda s: asset_data[s]["volatility"])
        rankings["volatility"] = volatility_sorted
        
        # Strength index ranking (higher is better)
        strength_sorted = sorted(symbols, key=lambda s: asset_data[s]["strength_index"], reverse=True)
        rankings["strength_index"] = strength_sorted
        
        return rankings
        
    def _generate_comparison_summary(self, primary_symbol: str, asset_data: Dict,
                                   rankings: Dict, time_period: str) -> str:
        """Generate a comparison summary focusing on the primary asset"""
        if primary_symbol not in asset_data:
            return f"No data available for {primary_symbol}"
            
        # Extract primary asset data
        primary_data = asset_data[primary_symbol]
        
        # Get rankings for primary asset
        return_rank = rankings["percent_change"].index(primary_symbol) + 1
        volatility_rank = rankings["volatility"].index(primary_symbol) + 1
        strength_rank = rankings["strength_index"].index(primary_symbol) + 1
        
        total_assets = len(asset_data)
        
        # Format the summary
        summary = f"{primary_symbol} {time_period} performance: {primary_data['percent_change']:.2f}% change "
        
        # Add ranking context
        if total_assets > 1:
            summary += f"(ranked {return_rank}/{total_assets} in return, "
            summary += f"{volatility_rank}/{total_assets} in volatility, "
            summary += f"{strength_rank}/{total_assets} in risk-adjusted return)"
        
        # Add relative performance
        if return_rank == 1:
            summary += f"\nBest performer in the group over {time_period}."
        elif return_rank <= total_assets // 3:
            summary += f"\nOutperforming most assets in the comparison group."
        elif return_rank > 2 * (total_assets // 3):
            summary += f"\nUnderperforming compared to most assets in the group."
        
        # Add volatility context
        if volatility_rank == 1:
            summary += f"\nLowest volatility among compared assets ({primary_data['volatility']:.2f}%)."
        elif volatility_rank <= total_assets // 3:
            summary += f"\nLower volatility than most compared assets ({primary_data['volatility']:.2f}%)."
        elif volatility_rank > 2 * (total_assets // 3):
            summary += f"\nHigher volatility than most compared assets ({primary_data['volatility']:.2f}%)."
        
        # Add strength index context
        if strength_rank == 1:
            summary += f"\nBest risk-adjusted return in the comparison group."
        elif strength_rank <= total_assets // 3:
            summary += f"\nBetter risk-adjusted return than most compared assets."
        
        return summary 
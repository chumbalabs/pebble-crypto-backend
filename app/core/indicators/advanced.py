import numpy as np
from typing import List, Dict, Tuple

class BollingerBands:
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std
        
    def calculate(self, prices: List[float]) -> Dict:
        """
        Calculate Bollinger Bands for a given price series
        
        Args:
            prices: List of closing prices
            
        Returns:
            Dict containing upper band, middle band, and lower band values
        """
        prices_array = np.array(prices)
        if len(prices_array) < self.window:
            # Handle case with insufficient data
            middle_band = np.mean(prices_array)
            std = np.std(prices_array)
            upper_band = middle_band + self.num_std * std
            lower_band = middle_band - self.num_std * std
        else:
            # Calculate using the specified window
            middle_band = np.convolve(prices_array, 
                                     np.ones(self.window)/self.window, 
                                     mode='valid')
            # Calculate rolling standard deviation
            r_std = []
            for i in range(len(prices_array) - self.window + 1):
                std = np.std(prices_array[i:(i + self.window)])
                r_std.append(std)
            
            r_std = np.array(r_std)
            upper_band = middle_band + self.num_std * r_std
            lower_band = middle_band - self.num_std * r_std
            
        return {
            "upper_band": upper_band.tolist(),
            "middle_band": middle_band.tolist(),
            "lower_band": lower_band.tolist()
        }
    
    def get_signal(self, prices: List[float]) -> Dict:
        """
        Get trading signal based on Bollinger Bands
        
        Args:
            prices: List of closing prices
            
        Returns:
            Dict containing signal and additional metrics
        """
        bands = self.calculate(prices)
        current_price = prices[-1]
        
        # Get the latest band values
        upper = bands["upper_band"][-1]
        middle = bands["middle_band"][-1]
        lower = bands["lower_band"][-1]
        
        # Calculate %B (percent bandwidth)
        percent_b = (current_price - lower) / (upper - lower) if upper != lower else 0.5
        
        # Calculate bandwidth
        bandwidth = (upper - lower) / middle if middle != 0 else 0
        
        # Determine signal
        if current_price > upper:
            signal = "SELL"
            desc = "Price is above the upper band, indicating a potential sell signal"
        elif current_price < lower:
            signal = "BUY"
            desc = "Price is below the lower band, indicating a potential buy signal"
        else:
            # Price is within bands
            if percent_b > 0.8:
                signal = "OVERBOUGHT"
                desc = "Price is near the upper band, potentially overbought"
            elif percent_b < 0.2:
                signal = "OVERSOLD"
                desc = "Price is near the lower band, potentially oversold"
            else:
                signal = "NEUTRAL"
                desc = "Price is within normal range of the bands"
                
        return {
            "signal": signal,
            "percent_b": percent_b,
            "bandwidth": bandwidth,
            "description": desc,
            "upper": upper,
            "middle": middle,
            "lower": lower
        }

class AverageTrueRange:
    def __init__(self, window: int = 14):
        self.window = window
        
    def calculate(self, highs: List[float], lows: List[float], closes: List[float]) -> Dict:
        """
        Calculate Average True Range (ATR)
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            
        Returns:
            Dict containing ATR values and true ranges
        """
        if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
            return {"atr": [0], "tr": [0]}
            
        # Calculate True Range series
        tr_values = []
        
        # First TR value is simply High - Low
        tr_values.append(highs[0] - lows[0])
        
        # Calculate subsequent TR values
        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            close_prev = closes[i-1]
            
            tr = max(high - low,                     # Current high - low
                     abs(high - close_prev),         # Current high - previous close
                     abs(low - close_prev))          # Current low - previous close
            tr_values.append(tr)
        
        # Calculate ATR using the specified window
        atr_values = []
        if len(tr_values) < self.window:
            # Simple average for insufficient data
            atr_values.append(np.mean(tr_values))
        else:
            # Initial ATR is simple average of first 'window' TR values
            first_atr = np.mean(tr_values[:self.window])
            atr_values.append(first_atr)
            
            # Subsequent ATR values use the smoothing formula
            for i in range(self.window, len(tr_values)):
                atr = (atr_values[-1] * (self.window - 1) + tr_values[i]) / self.window
                atr_values.append(atr)
                
        return {
            "atr": atr_values,
            "tr": tr_values
        }
    
    def get_signal(self, highs: List[float], lows: List[float], closes: List[float]) -> Dict:
        """
        Get trading signal based on ATR
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            
        Returns:
            Dict containing ATR signal and derived metrics
        """
        result = self.calculate(highs, lows, closes)
        atr_values = result["atr"]
        current_atr = atr_values[-1]
        current_price = closes[-1]
        
        # Calculate ATR as percentage of price
        atr_percent = (current_atr / current_price) * 100
        
        # Calculate basic stop levels based on ATR multipliers
        stop_buy = current_price - current_atr * 1.5  # Long position stop loss
        stop_sell = current_price + current_atr * 1.5  # Short position stop loss
        
        # Determine volatility level based on ATR percentage
        if atr_percent < 1.0:
            volatility = "VERY LOW"
        elif atr_percent < 2.0:
            volatility = "LOW"
        elif atr_percent < 4.0:
            volatility = "MODERATE"
        elif atr_percent < 7.0:
            volatility = "HIGH"
        else:
            volatility = "VERY HIGH"
            
        # Calculate market phase information
        market_phase = self._determine_market_phase(highs, lows, closes, atr_values)
        
        # Calculate historical volatility comparison
        volatility_comparison = self._compare_historical_volatility(atr_values, closes)
            
        return {
            "atr_value": current_atr,
            "atr_percent": atr_percent,
            "volatility": volatility,
            "stop_buy": stop_buy,
            "stop_sell": stop_sell,
            "market_phase": market_phase,
            "historical_comparison": volatility_comparison
        }
    
    def _determine_market_phase(self, highs: List[float], lows: List[float], 
                              closes: List[float], atr_values: List[float]) -> Dict:
        """Determine the current market phase based on price action and ATR"""
        if len(closes) < 10 or len(atr_values) < 10:
            return {"phase": "UNKNOWN", "confidence": 0.0}
            
        # Calculate short-term price change (last 5 periods)
        short_term_change = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
        
        # Calculate ATR trend (expanding or contracting)
        atr_change = (atr_values[-1] - atr_values[-6]) / atr_values[-6] if len(atr_values) >= 6 else 0
        
        # Calculate price swing (high to low range)
        recent_swing = (max(highs[-10:]) - min(lows[-10:])) / min(lows[-10:])
        
        # Determine market phase
        phase = ""
        confidence = 0.0
        
        if short_term_change > 0.03 and atr_change > 0:
            # Rising prices with expanding volatility
            phase = "TRENDING_UP"
            confidence = min(0.9, 0.5 + abs(short_term_change) + abs(atr_change))
        elif short_term_change < -0.03 and atr_change > 0:
            # Falling prices with expanding volatility
            phase = "TRENDING_DOWN"
            confidence = min(0.9, 0.5 + abs(short_term_change) + abs(atr_change))
        elif abs(short_term_change) < 0.02 and atr_change < 0:
            # Sideways prices with contracting volatility
            phase = "CONSOLIDATION"
            confidence = min(0.9, 0.5 + (1.0 - abs(short_term_change) * 10) + abs(atr_change))
        elif abs(short_term_change) < 0.01 and atr_change > 0.05:
            # Sideways prices with expanding volatility
            phase = "COILING"
            confidence = min(0.9, 0.4 + abs(atr_change) * 2)
        elif recent_swing > 0.08 and abs(short_term_change) < 0.02:
            # Large swings with minimal directional change
            phase = "CHOPPY"
            confidence = min(0.8, 0.3 + recent_swing)
        else:
            phase = "UNDEFINED"
            confidence = 0.3
            
        return {
            "phase": phase,
            "confidence": confidence,
            "price_change": short_term_change,
            "volatility_change": atr_change
        }
    
    def _compare_historical_volatility(self, atr_values: List[float], closes: List[float]) -> Dict:
        """Compare current volatility to historical context"""
        if len(atr_values) < 30:
            return {"percentile": 50, "relative_level": "NORMAL"}
            
        current_atr = atr_values[-1]
        current_price = closes[-1]
        current_atr_pct = current_atr / current_price
        
        # Calculate historical ATR percentages
        historical_atr_pct = []
        for i in range(min(90, len(atr_values) - 1)):
            idx = -(i + 2)  # Start from second-to-last and go backward
            if idx >= -len(atr_values) and idx < 0 and idx >= -len(closes):
                historical_atr_pct.append(atr_values[idx] / closes[idx])
        
        # Calculate percentile
        if historical_atr_pct:
            count_below = sum(1 for x in historical_atr_pct if x < current_atr_pct)
            percentile = (count_below / len(historical_atr_pct)) * 100
        else:
            percentile = 50
        
        # Determine relative level
        if percentile < 10:
            relative_level = "EXTREMELY LOW"
        elif percentile < 25:
            relative_level = "LOW"
        elif percentile < 40:
            relative_level = "BELOW AVERAGE"
        elif percentile < 60:
            relative_level = "AVERAGE"
        elif percentile < 75:
            relative_level = "ABOVE AVERAGE"
        elif percentile < 90:
            relative_level = "HIGH"
        else:
            relative_level = "EXTREMELY HIGH"
            
        return {
            "percentile": percentile,
            "relative_level": relative_level,
            "atr_percent": current_atr_pct * 100
        } 
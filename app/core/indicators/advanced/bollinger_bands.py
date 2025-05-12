import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger("CryptoPredictAPI")

class BollingerBands:
    """
    Bollinger Bands technical indicator implementation.
    Calculates standard deviation-based bands around a moving average.
    """
    
    def __init__(self, window: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands calculator
        
        Args:
            window: The lookback window for SMA calculation (default: 20)
            std_dev: The number of standard deviations for bands (default: 2.0)
        """
        self.window = window
        self.std_dev = std_dev
        
    def calculate(self, prices: List[float]) -> Dict[str, List[float]]:
        """
        Calculate Bollinger Bands for a price series
        
        Args:
            prices: List of closing prices
            
        Returns:
            Dict containing 'upper', 'middle' (SMA), and 'lower' bands
        """
        try:
            prices_array = np.array(prices)
            
            # Validate we have enough data
            if len(prices_array) < self.window:
                logger.warning(f"Insufficient data for Bollinger Bands calculation: {len(prices_array)} < {self.window}")
                middle_band = self._safe_sma(prices_array)
                
                # Create placeholder bands with available data
                std = np.std(prices_array) if len(prices_array) > 1 else prices_array[0] * 0.02
                upper_band = middle_band + self.std_dev * std
                lower_band = middle_band - self.std_dev * std
                
                # Convert to list
                return {
                    'upper': upper_band.tolist() if isinstance(upper_band, np.ndarray) else [upper_band],
                    'middle': middle_band.tolist() if isinstance(middle_band, np.ndarray) else [middle_band],
                    'lower': lower_band.tolist() if isinstance(lower_band, np.ndarray) else [lower_band]
                }
            
            # Calculate SMA
            middle_band = self._calculate_sma(prices_array)
            
            # Calculate standard deviation
            rolling_std = self._calculate_rolling_std(prices_array)
            
            # Calculate upper and lower bands
            upper_band = middle_band + self.std_dev * rolling_std
            lower_band = middle_band - self.std_dev * rolling_std
            
            return {
                'upper': upper_band.tolist(),
                'middle': middle_band.tolist(),
                'lower': lower_band.tolist()
            }
            
        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {str(e)}")
            # Return safe fallback
            avg_price = np.mean(prices)
            std = np.std(prices) if len(prices) > 1 else avg_price * 0.02
            return {
                'upper': [avg_price + self.std_dev * std],
                'middle': [avg_price],
                'lower': [avg_price - self.std_dev * std]
            }
    
    def _calculate_sma(self, prices: np.ndarray) -> np.ndarray:
        """Calculate Simple Moving Average with rolling window"""
        sma = np.convolve(prices, np.ones(self.window)/self.window, mode='valid')
        # Pad with NaN or first available values to match input length
        padding = np.array([np.nan] * (len(prices) - len(sma)))
        return np.concatenate((padding, sma))
    
    def _calculate_rolling_std(self, prices: np.ndarray) -> np.ndarray:
        """Calculate rolling standard deviation"""
        # Create output array
        rolling_std = np.full(len(prices), np.nan)
        
        # Calculate standard deviation for each window
        for i in range(self.window - 1, len(prices)):
            rolling_std[i] = np.std(prices[i - self.window + 1:i + 1])
            
        return rolling_std
    
    def _safe_sma(self, prices: np.ndarray) -> np.ndarray:
        """Safe SMA calculation for insufficient data"""
        if len(prices) <= 1:
            return prices
            
        # Use maximum available window
        window = min(self.window, len(prices))
        return self._calculate_sma(prices) if window > 1 else prices
    
    def get_signal(self, prices: List[float]) -> Dict[str, str]:
        """
        Get trading signal based on Bollinger Bands
        
        Args:
            prices: List of closing prices
            
        Returns:
            Dict with signal information
        """
        bands = self.calculate(prices)
        
        # Check if the most recent price breaks through bands
        current_price = prices[-1]
        upper_band = bands['upper'][-1]
        lower_band = bands['lower'][-1]
        middle_band = bands['middle'][-1]
        
        # Calculate bandwidth percentage
        bandwidth = (upper_band - lower_band) / middle_band * 100
        
        # Calculate %B indicator (position within bands)
        percent_b = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        
        # Determine signal
        if current_price > upper_band:
            signal = "SELL"
            strength = min(1.0, (current_price - upper_band) / (upper_band * 0.01))
        elif current_price < lower_band:
            signal = "BUY"
            strength = min(1.0, (lower_band - current_price) / (lower_band * 0.01))
        else:
            # Inside bands, check proximity
            if percent_b > 0.8:
                signal = "WEAK_SELL"
                strength = (percent_b - 0.8) * 5
            elif percent_b < 0.2:
                signal = "WEAK_BUY"
                strength = (0.2 - percent_b) * 5
            else:
                signal = "NEUTRAL"
                strength = 0.0
                
        return {
            'signal': signal,
            'strength': round(strength, 2),
            'bandwidth': round(bandwidth, 2),
            'percent_b': round(percent_b, 2)
        } 
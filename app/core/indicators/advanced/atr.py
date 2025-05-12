import numpy as np
from typing import Dict, List, Union, Optional
import logging

logger = logging.getLogger("CryptoPredictAPI")

class AverageTrueRange:
    """
    Average True Range (ATR) technical indicator implementation.
    Measures market volatility by calculating the moving average of the true range.
    """
    
    def __init__(self, window: int = 14, smoothing: str = 'rma'):
        """
        Initialize ATR calculator
        
        Args:
            window: The lookback window for calculation (default: 14)
            smoothing: Smoothing method ('simple', 'exponential', or 'rma')
        """
        self.window = window
        self.smoothing = smoothing
        
    def calculate(self, 
                 high: List[float], 
                 low: List[float], 
                 close: List[float], 
                 previous_close: Optional[List[float]] = None) -> Dict[str, List[float]]:
        """
        Calculate ATR for price data
        
        Args:
            high: List of high prices
            low: List of low prices
            close: List of closing prices
            previous_close: List of previous closing prices (optional)
            
        Returns:
            Dict containing 'atr' values and 'true_range' values
        """
        try:
            # Convert lists to numpy arrays
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)
            
            # Calculate previous close with offset, or use the provided previous_close
            if previous_close is None:
                prev_close = np.roll(close_array, 1)
                # Handle first element which has no previous close
                if len(close_array) > 0:
                    prev_close[0] = close_array[0]
            else:
                prev_close = np.array(previous_close)
                
            # Calculate true range
            tr1 = high_array - low_array  # Current high - current low
            tr2 = np.abs(high_array - prev_close)  # Current high - previous close
            tr3 = np.abs(low_array - prev_close)  # Current low - previous close
            
            # Combine to get true range
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Calculate ATR based on smoothing method
            atr = self._apply_smoothing(true_range)
            
            return {
                'atr': atr.tolist(),
                'true_range': true_range.tolist()
            }
            
        except Exception as e:
            logger.error(f"ATR calculation error: {str(e)}")
            # Return safe fallback
            if len(high) > 0:
                avg_price = np.mean(high)
                volatility = np.std(close) if len(close) > 1 else avg_price * 0.01
                return {
                    'atr': [volatility],
                    'true_range': [volatility]
                }
            return {
                'atr': [0.01],
                'true_range': [0.01]
            }
    
    def _apply_smoothing(self, true_range: np.ndarray) -> np.ndarray:
        """Apply selected smoothing method to true range values"""
        n = len(true_range)
        
        # Handle insufficient data case
        if n < self.window:
            logger.warning(f"Insufficient data for ATR calculation: {n} < {self.window}")
            return true_range
            
        # Output array
        atr = np.zeros(n)
        
        if self.smoothing == 'simple':
            # Simple moving average
            for i in range(n):
                if i < self.window - 1:
                    atr[i] = np.mean(true_range[:i+1])
                else:
                    atr[i] = np.mean(true_range[i-self.window+1:i+1])
        
        elif self.smoothing == 'exponential':
            # Exponential moving average
            alpha = 2 / (self.window + 1)
            
            # Initialize first value
            atr[0] = true_range[0]
            
            # Calculate EMA
            for i in range(1, n):
                atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i-1]
                
        else:  # Default to Wilder's smoothing (RMA)
            # Initialize first value
            atr[0] = true_range[0]
            
            # Smoothing factor
            alpha = 1 / self.window
            
            # Calculate RMA (Wilder's smoothing)
            for i in range(1, n):
                atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i-1]
                
        return atr
    
    def get_signal(self, 
                  high: List[float], 
                  low: List[float], 
                  close: List[float], 
                  multiplier: float = 2.0) -> Dict[str, Union[str, float]]:
        """
        Get volatility signal based on ATR
        
        Args:
            high: List of high prices
            low: List of low prices
            close: List of closing prices
            multiplier: Multiplier for ATR to determine stop levels
            
        Returns:
            Dict with signal information including volatility assessment 
            and potential stop-loss levels
        """
        result = self.calculate(high, low, close)
        atr = result['atr']
        
        # Get current values
        current_atr = atr[-1]
        current_close = close[-1]
        
        # Calculate percentage ATR
        atr_percent = current_atr / current_close * 100
        
        # Determine volatility level
        if atr_percent < 1.0:
            volatility = "LOW"
        elif atr_percent < 3.0:
            volatility = "MEDIUM"
        else:
            volatility = "HIGH"
            
        # Calculate stop levels
        stop_buy = current_close - current_atr * multiplier
        stop_sell = current_close + current_atr * multiplier
        
        return {
            'volatility': volatility,
            'atr_value': round(current_atr, 4),
            'atr_percent': round(atr_percent, 2),
            'stop_buy': round(stop_buy, 4),
            'stop_sell': round(stop_sell, 4)
        } 
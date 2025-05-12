import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from collections import defaultdict

logger = logging.getLogger("CryptoPredictAPI")

class OrderBookDepthAnalyzer:
    """
    Order book depth analyzer that identifies key price levels,
    imbalances, and potential support/resistance zones.
    """
    
    def __init__(self, price_grouping: float = 0.001, depth_percentage: float = 0.05):
        """
        Initialize Order Book Depth Analyzer
        
        Args:
            price_grouping: Percentage to group prices by (default: 0.1%)
            depth_percentage: Percentage of order book depth to analyze (default: 5%)
        """
        self.price_grouping = price_grouping
        self.depth_percentage = depth_percentage
        
    def analyze(self, bids: List[List[float]], asks: List[List[float]]) -> Dict[str, Any]:
        """
        Analyze order book data
        
        Args:
            bids: List of [price, quantity] lists for buy orders
            asks: List of [price, quantity] lists for sell orders
            
        Returns:
            Dict containing analysis results
        """
        try:
            if not bids or not asks:
                logger.warning("Empty order book data provided")
                return self._empty_result()
                
            # Extract data
            bid_prices = np.array([bid[0] for bid in bids])
            bid_quantities = np.array([bid[1] for bid in bids])
            ask_prices = np.array([ask[0] for ask in asks])
            ask_quantities = np.array([ask[1] for ask in asks])
            
            # Current mid price
            mid_price = (bid_prices[0] + ask_prices[0]) / 2
            spread = ask_prices[0] - bid_prices[0]
            spread_percentage = (spread / mid_price) * 100
            
            # Calculate depth percentage range
            max_depth = mid_price * self.depth_percentage
            
            # Filter depth
            bids_in_range = bid_prices >= (mid_price - max_depth)
            asks_in_range = ask_prices <= (mid_price + max_depth)
            
            filtered_bid_prices = bid_prices[bids_in_range]
            filtered_bid_quantities = bid_quantities[bids_in_range]
            filtered_ask_prices = ask_prices[asks_in_range]
            filtered_ask_quantities = ask_quantities[asks_in_range]
            
            # Group by price level
            bid_grouped = self._group_by_price(filtered_bid_prices, filtered_bid_quantities, mid_price)
            ask_grouped = self._group_by_price(filtered_ask_prices, filtered_ask_quantities, mid_price)
            
            # Calculate imbalance
            bid_volume = np.sum(filtered_bid_quantities)
            ask_volume = np.sum(filtered_ask_quantities)
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                bid_ratio = bid_volume / total_volume
                ask_ratio = ask_volume / total_volume
                imbalance = (bid_volume - ask_volume) / total_volume
            else:
                bid_ratio = 0.5
                ask_ratio = 0.5
                imbalance = 0
                
            # Find walls (large orders)
            bid_walls = self._find_walls(bid_grouped, "bid")
            ask_walls = self._find_walls(ask_grouped, "ask")
            
            # Identify support/resistance zones
            support_levels = self._identify_support(bid_grouped, mid_price)
            resistance_levels = self._identify_resistance(ask_grouped, mid_price)
            
            return {
                "mid_price": float(mid_price),
                "spread": float(spread),
                "spread_percentage": float(spread_percentage),
                "bid_volume": float(bid_volume),
                "ask_volume": float(ask_volume),
                "total_volume": float(total_volume),
                "bid_ratio": float(bid_ratio),
                "ask_ratio": float(ask_ratio),
                "imbalance": float(imbalance),
                "imbalance_signal": self._get_imbalance_signal(imbalance),
                "bid_walls": bid_walls,
                "ask_walls": ask_walls,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Order book analysis error: {str(e)}")
            return self._empty_result()
    
    def _group_by_price(self, prices: np.ndarray, quantities: np.ndarray, mid_price: float) -> Dict[float, float]:
        """Group quantities by price levels based on price_grouping"""
        result = defaultdict(float)
        
        for price, quantity in zip(prices, quantities):
            # Group price by percentage from mid price
            group_size = mid_price * self.price_grouping
            grouped_price = round(price / group_size) * group_size
            result[grouped_price] += quantity
            
        return dict(result)
    
    def _find_walls(self, grouped_data: Dict[float, float], side: str) -> List[Dict[str, float]]:
        """Find price walls (large orders) in the order book"""
        if not grouped_data:
            return []
            
        # Calculate average and std dev of quantities
        quantities = np.array(list(grouped_data.values()))
        avg_quantity = np.mean(quantities)
        std_quantity = np.std(quantities)
        
        # Wall threshold (>2 standard deviations above mean)
        threshold = avg_quantity + 2 * std_quantity if len(quantities) > 1 else avg_quantity * 2
        
        walls = []
        for price, quantity in grouped_data.items():
            if quantity > threshold:
                walls.append({
                    "price": float(price),
                    "quantity": float(quantity),
                    "type": side,
                    "strength": float(min(1.0, (quantity - threshold) / (threshold * 0.5)))
                })
                
        # Sort by quantity descending
        return sorted(walls, key=lambda x: x["quantity"], reverse=True)[:5]
    
    def _identify_support(self, bid_data: Dict[float, float], mid_price: float) -> List[Dict[str, float]]:
        """Identify support levels from bid data"""
        if not bid_data:
            return []
            
        # Convert to numpy arrays
        prices = np.array(list(bid_data.keys()))
        quantities = np.array(list(bid_data.values()))
        
        # Sort by price descending
        sorted_indices = np.argsort(prices)[::-1]
        sorted_prices = prices[sorted_indices]
        sorted_quantities = quantities[sorted_indices]
        
        # Calculate cumulative volume
        cumulative_volume = np.cumsum(sorted_quantities)
        total_volume = cumulative_volume[-1] if len(cumulative_volume) > 0 else 0
        
        support_levels = []
        
        # Look for significant volume at price levels
        if total_volume > 0:
            for i, (price, quantity) in enumerate(zip(sorted_prices, sorted_quantities)):
                vol_ratio = quantity / total_volume
                cum_ratio = cumulative_volume[i] / total_volume
                
                # Check if this level has significant volume
                if vol_ratio > 0.05 or (i > 0 and cum_ratio > 0.2 and cum_ratio < 0.8):
                    strength = min(1.0, vol_ratio * 10)  # Scale from 0 to 1
                    support_levels.append({
                        "price": float(price),
                        "volume_ratio": float(vol_ratio),
                        "cumulative_ratio": float(cum_ratio),
                        "strength": float(strength)
                    })
        
        # Sort by strength and limit to top 3
        return sorted(support_levels, key=lambda x: x["strength"], reverse=True)[:3]
    
    def _identify_resistance(self, ask_data: Dict[float, float], mid_price: float) -> List[Dict[str, float]]:
        """Identify resistance levels from ask data"""
        if not ask_data:
            return []
            
        # Convert to numpy arrays
        prices = np.array(list(ask_data.keys()))
        quantities = np.array(list(ask_data.values()))
        
        # Sort by price ascending
        sorted_indices = np.argsort(prices)
        sorted_prices = prices[sorted_indices]
        sorted_quantities = quantities[sorted_indices]
        
        # Calculate cumulative volume
        cumulative_volume = np.cumsum(sorted_quantities)
        total_volume = cumulative_volume[-1] if len(cumulative_volume) > 0 else 0
        
        resistance_levels = []
        
        # Look for significant volume at price levels
        if total_volume > 0:
            for i, (price, quantity) in enumerate(zip(sorted_prices, sorted_quantities)):
                vol_ratio = quantity / total_volume
                cum_ratio = cumulative_volume[i] / total_volume
                
                # Check if this level has significant volume
                if vol_ratio > 0.05 or (i > 0 and cum_ratio > 0.2 and cum_ratio < 0.8):
                    strength = min(1.0, vol_ratio * 10)  # Scale from 0 to 1
                    resistance_levels.append({
                        "price": float(price),
                        "volume_ratio": float(vol_ratio),
                        "cumulative_ratio": float(cum_ratio),
                        "strength": float(strength)
                    })
        
        # Sort by strength and limit to top 3
        return sorted(resistance_levels, key=lambda x: x["strength"], reverse=True)[:3]
    
    def _get_imbalance_signal(self, imbalance: float) -> str:
        """Get market signal based on order book imbalance"""
        if imbalance > 0.2:
            return "STRONG_BUY"
        elif imbalance > 0.1:
            return "BUY"
        elif imbalance < -0.2:
            return "STRONG_SELL"
        elif imbalance < -0.1:
            return "SELL"
        else:
            return "NEUTRAL"
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "mid_price": 0.0,
            "spread": 0.0,
            "spread_percentage": 0.0,
            "bid_volume": 0.0,
            "ask_volume": 0.0,
            "total_volume": 0.0,
            "bid_ratio": 0.5,
            "ask_ratio": 0.5,
            "imbalance": 0.0,
            "imbalance_signal": "NEUTRAL",
            "bid_walls": [],
            "ask_walls": [],
            "support_levels": [],
            "resistance_levels": []
        } 
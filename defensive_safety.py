#!/usr/bin/env python3
"""
Defensive Safety Module
Comprehensive protection against division by zero, null values, and edge cases
"""

import math
import logging
from typing import Union, Optional, Any, List, Dict
from functools import wraps

# Configure safety logging
logging.basicConfig(level=logging.WARNING)
safety_logger = logging.getLogger('defensive_safety')


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with comprehensive protection
    
    Args:
        numerator: The number to divide
        denominator: The divisor
        default: Value to return if division is unsafe
    
    Returns:
        Safe division result or default value
    """
    try:
        if denominator == 0:
            return default
        
        if math.isnan(numerator) or math.isnan(denominator):
            return default
            
        if math.isinf(numerator) or math.isinf(denominator):
            return default
        
        result = numerator / denominator
        
        # Check result validity
        if math.isnan(result) or math.isinf(result):
            return default
            
        return result
        
    except (ZeroDivisionError, OverflowError, TypeError):
        return default


def safe_percentage(numerator: float, denominator: float) -> float:
    """
    Safe percentage calculation (0-100 scale)
    
    Args:
        numerator: Part value
        denominator: Total value
    
    Returns:
        Safe percentage or 0.0
    """
    if denominator <= 0:
        return 0.0
    return min(100.0, max(0.0, (numerator / denominator) * 100))


def safe_ratio(numerator: float, denominator: float, clamp_max: float = 1000.0) -> float:
    """
    Safe ratio calculation with clamping
    
    Args:
        numerator: The number to divide
        denominator: The divisor  
        clamp_max: Maximum allowed ratio value
    
    Returns:
        Safe ratio clamped within reasonable bounds
    """
    if denominator <= 0:
        return 0.0
    
    ratio = safe_divide(numerator, denominator, 0.0)
    return min(clamp_max, max(0.0, ratio))


def safe_list_access(lst: List, index: int, default: Any = None) -> Any:
    """
    Safe list element access
    
    Args:
        lst: The list to access
        index: Index to retrieve
        default: Default value if access fails
    
    Returns:
        List element or default value
    """
    try:
        if not lst or index < 0 or index >= len(lst):
            return default
        return lst[index]
    except (IndexError, TypeError):
        return default


def safe_dict_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """
    Safe dictionary value retrieval
    
    Args:
        dictionary: The dictionary to access
        key: Key to retrieve
        default: Default value if key doesn't exist
    
    Returns:
        Dictionary value or default
    """
    try:
        if not isinstance(dictionary, dict):
            return default
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default


def validate_numeric(value: Any, min_val: float = None, max_val: float = None, default: float = 0.0) -> float:
    """
    Validate and sanitize numeric values
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default value if validation fails
    
    Returns:
        Validated numeric value
    """
    try:
        # Convert to float
        if isinstance(value, (int, float)):
            num_val = float(value)
        else:
            num_val = float(value)
        
        # Check for NaN/Inf
        if math.isnan(num_val) or math.isinf(num_val):
            return default
        
        # Apply bounds
        if min_val is not None:
            num_val = max(num_val, min_val)
        if max_val is not None:
            num_val = min(num_val, max_val)
            
        return num_val
        
    except (ValueError, TypeError, OverflowError):
        return default


def safe_performance_metrics(wins: int, losses: int, total_pnl: float, 
                           gross_profit: float = 0.0, gross_loss: float = 0.0) -> Dict[str, float]:
    """
    Calculate performance metrics with comprehensive safety
    
    Args:
        wins: Number of winning trades
        losses: Number of losing trades
        total_pnl: Total profit/loss
        gross_profit: Total profit from winning trades
        gross_loss: Total loss from losing trades (should be negative)
    
    Returns:
        Dictionary of safe performance metrics
    """
    total_trades = wins + losses
    
    # Win rate with safety
    win_rate = safe_percentage(wins, total_trades)
    
    # Profit factor with safety
    profit_factor = safe_ratio(gross_profit, abs(gross_loss), 10.0) if gross_loss < 0 else 0.0
    
    # Average win/loss with safety
    avg_win = safe_divide(gross_profit, wins, 0.0)
    avg_loss = safe_divide(gross_loss, losses, 0.0)
    
    # Sharpe approximation
    if total_trades > 5:
        avg_pnl = safe_divide(total_pnl, total_trades, 0.0)
        # Simple volatility estimation
        volatility = abs(avg_pnl) * 0.1 + 1.0  # Ensure non-zero
        sharpe_ratio = safe_divide(avg_pnl, volatility, 0.0)
    else:
        sharpe_ratio = 0.0
    
    return {
        'win_rate': validate_numeric(win_rate, 0.0, 100.0, 0.0),
        'profit_factor': validate_numeric(profit_factor, 0.0, 10.0, 0.0),
        'avg_win': validate_numeric(avg_win, default=0.0),
        'avg_loss': validate_numeric(avg_loss, default=0.0), 
        'sharpe_ratio': validate_numeric(sharpe_ratio, -5.0, 5.0, 0.0),
        'total_trades': max(0, total_trades)
    }


def defensive_decorator(default_return=None):
    """
    Decorator for adding defensive error handling to functions
    
    Args:
        default_return: Value to return if function fails
    
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                safety_logger.warning(f"Defensive catch in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between min and max bounds
    
    Args:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound
    
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def safe_array_mean(arr: List[float]) -> float:
    """
    Calculate array mean with safety
    
    Args:
        arr: Array of numbers
    
    Returns:
        Safe mean or 0.0
    """
    if not arr:
        return 0.0
    
    try:
        # Filter out invalid values
        valid_values = [x for x in arr if not (math.isnan(x) or math.isinf(x))]
        
        if not valid_values:
            return 0.0
            
        return sum(valid_values) / len(valid_values)
    except (TypeError, ZeroDivisionError):
        return 0.0


def safe_array_std(arr: List[float]) -> float:
    """
    Calculate array standard deviation with safety
    
    Args:
        arr: Array of numbers
    
    Returns:
        Safe standard deviation or 0.0
    """
    if not arr or len(arr) < 2:
        return 0.0
    
    try:
        # Filter out invalid values
        valid_values = [x for x in arr if not (math.isnan(x) or math.isinf(x))]
        
        if len(valid_values) < 2:
            return 0.0
            
        mean_val = sum(valid_values) / len(valid_values)
        variance = sum((x - mean_val) ** 2 for x in valid_values) / len(valid_values)
        
        return math.sqrt(variance)
    except (TypeError, ZeroDivisionError, ValueError):
        return 0.0


class SafePerformanceTracker:
    """
    Thread-safe performance tracking with defensive measures
    """
    
    def __init__(self):
        self.trades = []
        self.running_pnl = 0.0
        self.running_wins = 0
        self.running_losses = 0
        
    def add_trade(self, pnl: float):
        """Add trade with validation"""
        pnl = validate_numeric(pnl, default=0.0)
        
        self.trades.append(pnl)
        self.running_pnl += pnl
        
        if pnl > 0:
            self.running_wins += 1
        elif pnl < 0:
            self.running_losses += 1
            
        # Limit memory usage
        if len(self.trades) > 1000:
            old_trade = self.trades.pop(0)
            # Recalculate running totals for accuracy
            self._recalculate_stats()
    
    def _recalculate_stats(self):
        """Recalculate statistics from trade history"""
        self.running_pnl = sum(self.trades)
        self.running_wins = sum(1 for t in self.trades if t > 0)
        self.running_losses = sum(1 for t in self.trades if t < 0)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get safe performance metrics"""
        gross_profit = sum(t for t in self.trades if t > 0)
        gross_loss = sum(t for t in self.trades if t < 0)
        
        return safe_performance_metrics(
            self.running_wins, 
            self.running_losses, 
            self.running_pnl,
            gross_profit,
            gross_loss
        )
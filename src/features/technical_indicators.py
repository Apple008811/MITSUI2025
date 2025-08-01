"""
Technical indicators for financial time series analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Class for calculating various technical indicators.
    """
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        pass
    
    def simple_moving_average(self, prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            prices: Price series
            window: Window size for calculation
            
        Returns:
            Series containing SMA values
        """
        return prices.rolling(window=window).mean()
    
    def exponential_moving_average(self, prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            prices: Price series
            window: Window size for calculation
            
        Returns:
            Series containing EMA values
        """
        return prices.ewm(span=window).mean()
    
    def relative_strength_index(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            window: Window size for calculation (default: 14)
            
        Returns:
            Series containing RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = self.exponential_moving_average(prices, fast)
        ema_slow = self.exponential_moving_average(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.exponential_moving_average(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            window: Window size for calculation (default: 20)
            num_std: Number of standard deviations (default: 2)
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = self.simple_moving_average(prices, window)
        std = prices.rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        return upper_band, middle_band, lower_band
    
    def stochastic_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_window: %K window size (default: 14)
            d_window: %D window size (default: 3)
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Window size for calculation (default: 14)
            
        Returns:
            Series containing Williams %R values
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def commodity_channel_index(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Window size for calculation (default: 20)
            
        Returns:
            Series containing CCI values
        """
        typical_price = (high + low + close) / 3
        sma_tp = self.simple_moving_average(typical_price, window)
        mean_deviation = (typical_price - sma_tp).abs().rolling(window=window).mean()
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def average_directional_index(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Window size for calculation (default: 14)
            
        Returns:
            Series containing ADX values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth the values
        atr = true_range.rolling(window=window).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    def calculate_all_indicators(self, df: pd.DataFrame, price_col: str = 'close',
                               high_col: str = 'high', low_col: str = 'low') -> pd.DataFrame:
        """
        Calculate all technical indicators for a given DataFrame.
        
        Args:
            df: DataFrame containing price data
            price_col: Name of the close price column
            high_col: Name of the high price column
            low_col: Name of the low price column
            
        Returns:
            DataFrame with all technical indicators added
        """
        result_df = df.copy()
        
        # Price-based indicators
        result_df['sma_20'] = self.simple_moving_average(df[price_col], 20)
        result_df['sma_50'] = self.simple_moving_average(df[price_col], 50)
        result_df['ema_12'] = self.exponential_moving_average(df[price_col], 12)
        result_df['ema_26'] = self.exponential_moving_average(df[price_col], 26)
        result_df['rsi'] = self.relative_strength_index(df[price_col])
        
        # MACD
        macd_line, signal_line, histogram = self.macd(df[price_col])
        result_df['macd'] = macd_line
        result_df['macd_signal'] = signal_line
        result_df['macd_histogram'] = histogram
        
        # Bollinger Bands
        upper, middle, lower = self.bollinger_bands(df[price_col])
        result_df['bb_upper'] = upper
        result_df['bb_middle'] = middle
        result_df['bb_lower'] = lower
        result_df['bb_width'] = (upper - lower) / middle
        
        # Stochastic Oscillator
        k_percent, d_percent = self.stochastic_oscillator(df[high_col], df[low_col], df[price_col])
        result_df['stoch_k'] = k_percent
        result_df['stoch_d'] = d_percent
        
        # Williams %R
        result_df['williams_r'] = self.williams_r(df[high_col], df[low_col], df[price_col])
        
        # CCI
        result_df['cci'] = self.commodity_channel_index(df[high_col], df[low_col], df[price_col])
        
        # ADX
        result_df['adx'] = self.average_directional_index(df[high_col], df[low_col], df[price_col])
        
        logger.info(f"Calculated {len(result_df.columns) - len(df.columns)} technical indicators")
        return result_df 
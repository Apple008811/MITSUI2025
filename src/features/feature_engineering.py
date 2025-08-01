"""
Feature engineering pipeline for commodity prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for creating comprehensive features from financial data.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.technical_indicators = TechnicalIndicators()
        self.feature_columns = []
        
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], 
                          lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lag features for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        result_df = df.copy()
        
        for col in columns:
            for lag in lags:
                feature_name = f"{col}_lag_{lag}"
                result_df[feature_name] = df[col].shift(lag)
                self.feature_columns.append(feature_name)
        
        logger.info(f"Created {len(columns) * len(lags)} lag features")
        return result_df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str],
                              windows: List[int], functions: List[str]) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            functions: List of functions to apply ('mean', 'std', 'min', 'max', 'median')
            
        Returns:
            DataFrame with rolling features added
        """
        result_df = df.copy()
        
        for col in columns:
            for window in windows:
                for func in functions:
                    feature_name = f"{col}_rolling_{func}_{window}"
                    
                    if func == 'mean':
                        result_df[feature_name] = df[col].rolling(window=window).mean()
                    elif func == 'std':
                        result_df[feature_name] = df[col].rolling(window=window).std()
                    elif func == 'min':
                        result_df[feature_name] = df[col].rolling(window=window).min()
                    elif func == 'max':
                        result_df[feature_name] = df[col].rolling(window=window).max()
                    elif func == 'median':
                        result_df[feature_name] = df[col].rolling(window=window).median()
                    elif func == 'skew':
                        result_df[feature_name] = df[col].rolling(window=window).skew()
                    elif func == 'kurt':
                        result_df[feature_name] = df[col].rolling(window=window).kurt()
                    
                    self.feature_columns.append(feature_name)
        
        logger.info(f"Created {len(columns) * len(windows) * len(functions)} rolling features")
        return result_df
    
    def create_price_difference_features(self, df: pd.DataFrame, price_columns: List[str],
                                       windows: List[int]) -> pd.DataFrame:
        """
        Create price difference features.
        
        Args:
            df: Input DataFrame
            price_columns: Price columns to create differences for
            windows: List of difference windows
            
        Returns:
            DataFrame with price difference features added
        """
        result_df = df.copy()
        
        for col in price_columns:
            for window in windows:
                feature_name = f"{col}_diff_{window}"
                result_df[feature_name] = df[col].diff(window)
                self.feature_columns.append(feature_name)
        
        logger.info(f"Created {len(price_columns) * len(windows)} price difference features")
        return result_df
    
    def create_ratio_features(self, df: pd.DataFrame, numerator_cols: List[str],
                            denominator_cols: List[str]) -> pd.DataFrame:
        """
        Create ratio features between columns.
        
        Args:
            df: Input DataFrame
            numerator_cols: Numerator columns
            denominator_cols: Denominator columns
            
        Returns:
            DataFrame with ratio features added
        """
        result_df = df.copy()
        
        for num_col in numerator_cols:
            for den_col in denominator_cols:
                if num_col != den_col:
                    feature_name = f"{num_col}_div_{den_col}"
                    result_df[feature_name] = df[num_col] / df[den_col]
                    self.feature_columns.append(feature_name)
        
        logger.info(f"Created {len(numerator_cols) * len(denominator_cols)} ratio features")
        return result_df
    
    def create_interaction_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Create interaction features between columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create interactions for
            
        Returns:
            DataFrame with interaction features added
        """
        result_df = df.copy()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                feature_name = f"{col1}_times_{col2}"
                result_df[feature_name] = df[col1] * df[col2]
                self.feature_columns.append(feature_name)
        
        logger.info(f"Created {len(columns) * (len(columns) - 1) // 2} interaction features")
        return result_df
    
    def create_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create time-based features from date column.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            
        Returns:
            DataFrame with time features added
        """
        result_df = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            result_df[date_column] = pd.to_datetime(df[date_column])
        
        # Extract time components
        result_df['year'] = result_df[date_column].dt.year
        result_df['month'] = result_df[date_column].dt.month
        result_df['day'] = result_df[date_column].dt.day
        result_df['dayofweek'] = result_df[date_column].dt.dayofweek
        result_df['quarter'] = result_df[date_column].dt.quarter
        result_df['dayofyear'] = result_df[date_column].dt.dayofyear
        
        # Cyclical encoding for periodic features
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        result_df['dayofweek_sin'] = np.sin(2 * np.pi * result_df['dayofweek'] / 7)
        result_df['dayofweek_cos'] = np.cos(2 * np.pi * result_df['dayofweek'] / 7)
        
        time_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 'dayofyear',
                        'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos']
        self.feature_columns.extend(time_features)
        
        logger.info(f"Created {len(time_features)} time features")
        return result_df
    
    def create_volatility_features(self, df: pd.DataFrame, price_columns: List[str],
                                 windows: List[int]) -> pd.DataFrame:
        """
        Create volatility features.
        
        Args:
            df: Input DataFrame
            price_columns: Price columns to calculate volatility for
            windows: List of window sizes
            
        Returns:
            DataFrame with volatility features added
        """
        result_df = df.copy()
        
        for col in price_columns:
            # Calculate returns
            returns = df[col].pct_change()
            
            for window in windows:
                # Rolling volatility
                vol_feature = f"{col}_volatility_{window}"
                result_df[vol_feature] = returns.rolling(window=window).std()
                self.feature_columns.append(vol_feature)
                
                # Realized volatility
                realized_vol_feature = f"{col}_realized_vol_{window}"
                result_df[realized_vol_feature] = np.sqrt((returns**2).rolling(window=window).sum())
                self.feature_columns.append(realized_vol_feature)
        
        logger.info(f"Created {len(price_columns) * len(windows) * 2} volatility features")
        return result_df
    
    def create_momentum_features(self, df: pd.DataFrame, price_columns: List[str],
                               windows: List[int]) -> pd.DataFrame:
        """
        Create momentum features.
        
        Args:
            df: Input DataFrame
            price_columns: Price columns to calculate momentum for
            windows: List of window sizes
            
        Returns:
            DataFrame with momentum features added
        """
        result_df = df.copy()
        
        for col in price_columns:
            for window in windows:
                # Price momentum
                momentum_feature = f"{col}_momentum_{window}"
                result_df[momentum_feature] = df[col] / df[col].shift(window) - 1
                self.feature_columns.append(momentum_feature)
                
                # Rate of change
                roc_feature = f"{col}_roc_{window}"
                result_df[roc_feature] = (df[col] - df[col].shift(window)) / df[col].shift(window)
                self.feature_columns.append(roc_feature)
        
        logger.info(f"Created {len(price_columns) * len(windows) * 2} momentum features")
        return result_df
    
    def engineer_all_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Create all features based on configuration.
        
        Args:
            df: Input DataFrame
            config: Configuration dictionary containing feature parameters
            
        Returns:
            DataFrame with all features added
        """
        result_df = df.copy()
        
        # Get feature parameters from config
        feature_params = config.get('features', {})
        
        # Identify numeric columns for feature creation
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        price_columns = [col for col in numeric_columns if any(price_term in col.lower() 
                                                             for price_term in ['price', 'close', 'open', 'high', 'low'])]
        
        # Create lag features
        if 'lag_features' in feature_params:
            lags = feature_params['lag_features']
            result_df = self.create_lag_features(result_df, numeric_columns, lags)
        
        # Create rolling features
        if 'rolling_windows' in feature_params:
            windows = feature_params['rolling_windows']
            functions = ['mean', 'std', 'min', 'max', 'median']
            result_df = self.create_rolling_features(result_df, numeric_columns, windows, functions)
        
        # Create price difference features
        if 'price_diff_windows' in feature_params:
            windows = feature_params['price_diff_windows']
            result_df = self.create_price_difference_features(result_df, price_columns, windows)
        
        # Create volatility features
        if 'rolling_windows' in feature_params:
            windows = feature_params['rolling_windows']
            result_df = self.create_volatility_features(result_df, price_columns, windows)
        
        # Create momentum features
        if 'rolling_windows' in feature_params:
            windows = feature_params['rolling_windows']
            result_df = self.create_momentum_features(result_df, price_columns, windows)
        
        # Create time features if date column exists
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            result_df = self.create_time_features(result_df, date_columns[0])
        
        # Add technical indicators if price columns exist
        if price_columns:
            high_col = next((col for col in price_columns if 'high' in col.lower()), price_columns[0])
            low_col = next((col for col in price_columns if 'low' in col.lower()), price_columns[0])
            close_col = next((col for col in price_columns if 'close' in col.lower()), price_columns[0])
            
            result_df = self.technical_indicators.calculate_all_indicators(
                result_df, close_col, high_col, low_col
            )
            
            # Add technical indicator columns to feature list
            tech_indicators = [col for col in result_df.columns if col not in df.columns]
            self.feature_columns.extend(tech_indicators)
        
        logger.info(f"Total features created: {len(self.feature_columns)}")
        return result_df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of created feature columns.
        
        Returns:
            List of feature column names
        """
        return self.feature_columns.copy()
    
    def remove_high_correlation_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features to reduce multicollinearity.
        
        Args:
            df: Input DataFrame
            threshold: Correlation threshold for removal
            
        Returns:
            DataFrame with highly correlated features removed
        """
        # Calculate correlation matrix
        corr_matrix = df[self.feature_columns].corr().abs()
        
        # Find upper triangle of correlation matrix
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation above threshold
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Remove highly correlated features
        result_df = df.drop(columns=to_drop)
        
        # Update feature columns list
        self.feature_columns = [col for col in self.feature_columns if col not in to_drop]
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        return result_df 
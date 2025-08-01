"""
Competition evaluation metrics for the MITSUI&CO. Commodity Prediction Challenge.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Union, Tuple


def sharpe_ratio_variant(y_true: Union[np.ndarray, pd.Series], 
                        y_pred: Union[np.ndarray, pd.Series],
                        risk_free_rate: float = 0.0) -> float:
    """
    Calculate the competition metric: variant of Sharpe ratio.
    
    The metric is computed as:
    mean(Spearman rank correlation between predictions and targets) / 
    standard deviation(Spearman rank correlation between predictions and targets)
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        risk_free_rate: Risk-free rate (default: 0.0)
        
    Returns:
        Sharpe ratio variant score
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Calculate Spearman rank correlation
    correlation, _ = spearmanr(y_true, y_pred)
    
    # If correlation is NaN, return 0
    if np.isnan(correlation):
        return 0.0
    
    # Calculate excess return (correlation - risk_free_rate)
    excess_return = correlation - risk_free_rate
    
    # For single correlation, standard deviation is 0, so we return the correlation
    # In practice, this would be calculated over multiple time periods
    return excess_return


def calculate_rolling_sharpe_ratio(y_true: Union[np.ndarray, pd.Series],
                                 y_pred: Union[np.ndarray, pd.Series],
                                 window: int = 252,
                                 risk_free_rate: float = 0.0) -> float:
    """
    Calculate rolling Sharpe ratio variant over multiple time windows.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        window: Rolling window size (default: 252 for daily data)
        risk_free_rate: Risk-free rate (default: 0.0)
        
    Returns:
        Rolling Sharpe ratio variant score
    """
    if len(y_true) < window:
        return sharpe_ratio_variant(y_true, y_pred, risk_free_rate)
    
    correlations = []
    
    for i in range(window, len(y_true)):
        y_true_window = y_true[i-window:i]
        y_pred_window = y_pred[i-window:i]
        
        correlation, _ = spearmanr(y_true_window, y_pred_window)
        if not np.isnan(correlation):
            correlations.append(correlation)
    
    if not correlations:
        return 0.0
    
    correlations = np.array(correlations)
    mean_correlation = np.mean(correlations)
    std_correlation = np.std(correlations)
    
    if std_correlation == 0:
        return mean_correlation - risk_free_rate
    
    return (mean_correlation - risk_free_rate) / std_correlation


def evaluate_predictions(y_true: Union[np.ndarray, pd.Series],
                        y_pred: Union[np.ndarray, pd.Series],
                        verbose: bool = True) -> dict:
    """
    Comprehensive evaluation of predictions using multiple metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        verbose: Whether to print results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Competition metric
    sharpe_score = sharpe_ratio_variant(y_true, y_pred)
    
    # Additional metrics for analysis
    correlation, p_value = spearmanr(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    results = {
        'sharpe_ratio_variant': sharpe_score,
        'spearman_correlation': correlation,
        'spearman_p_value': p_value,
        'mse': mse,
        'mae': mae
    }
    
    if verbose:
        print("Evaluation Results:")
        print(f"Sharpe Ratio Variant: {sharpe_score:.4f}")
        print(f"Spearman Correlation: {correlation:.4f}")
        print(f"Spearman P-value: {p_value:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
    
    return results 
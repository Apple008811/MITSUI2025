"""
MITSUI&CO. Commodity Prediction Challenge - Data Exploration

This script explores the data sources and provides initial insights for the commodity prediction challenge.

Overview:
- Objective: Predict future commodity returns using historical data
- Data Sources: LME, JPX, US Stock, and Forex markets
- Evaluation Metric: Sharpe ratio variant (Spearman correlation / std)
- Target: Price-difference series between two assets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import project modules
import sys
sys.path.append('..')

from src.utils.config import load_config
from src.data.loader import DataLoader
from src.utils.metrics import sharpe_ratio_variant
from scipy import stats
from statsmodels.tsa.stattools import adfuller


def create_sample_data():
    """Create sample data for demonstration purposes."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create date range
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create sample price data with realistic patterns
    base_price = 100
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create sample data
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
        'volume': np.random.randint(1000, 10000, n_samples),
        'lme_copper': np.random.normal(8000, 500, n_samples),
        'jpx_nikkei': np.random.normal(28000, 1000, n_samples),
        'us_sp500': np.random.normal(4000, 100, n_samples),
        'forex_usdjpy': np.random.normal(110, 2, n_samples)
    })
    
    # Create target variable (price difference series)
    data['target'] = data['close'].diff(1)
    
    return data


def analyze_target_variable(data):
    """Analyze the target variable."""
    print("Target Variable Analysis:")
    print("=" * 50)
    
    # Remove NaN values for analysis
    target_clean = data['target'].dropna()
    
    print(f"Target statistics:")
    print(f"Mean: {target_clean.mean():.6f}")
    print(f"Std: {target_clean.std():.6f}")
    print(f"Min: {target_clean.min():.6f}")
    print(f"Max: {target_clean.max():.6f}")
    print(f"Skewness: {target_clean.skew():.6f}")
    print(f"Kurtosis: {target_clean.kurtosis():.6f}")
    
    return target_clean


def plot_target_analysis(data, target_clean):
    """Plot target variable analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    axes[0, 0].plot(data['date'], data['target'])
    axes[0, 0].set_title('Target Variable Time Series')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Target Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Histogram
    axes[0, 1].hist(target_clean, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Target Variable Distribution')
    axes[0, 1].set_xlabel('Target Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Box plot
    axes[1, 0].boxplot(target_clean)
    axes[1, 0].set_title('Target Variable Box Plot')
    axes[1, 0].set_ylabel('Target Value')
    
    # Q-Q plot
    stats.probplot(target_clean, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
    
    plt.tight_layout()
    plt.show()


def analyze_features(data):
    """Analyze individual features."""
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns.remove('target')  # Remove target from features
    
    print(f"Numeric features: {len(numeric_columns)}")
    print("Features:", numeric_columns)
    
    return numeric_columns


def plot_feature_time_series(data):
    """Plot feature time series."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    features_to_plot = ['close', 'lme_copper', 'jpx_nikkei', 'us_sp500']
    
    for i, feature in enumerate(features_to_plot):
        row = i // 2
        col = i % 2
        
        axes[row, col].plot(data['date'], data[feature])
        axes[row, col].set_title(f'{feature.upper()} Time Series')
        axes[row, col].set_xlabel('Date')
        axes[row, col].set_ylabel('Value')
        axes[row, col].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def correlation_analysis(data, numeric_columns):
    """Perform correlation analysis."""
    correlation_matrix = data[numeric_columns + ['target']].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Target correlation with features
    target_correlations = correlation_matrix['target'].sort_values(ascending=False)
    print("Target Variable Correlations:")
    print("=" * 50)
    for feature, corr in target_correlations.items():
        if feature != 'target':
            print(f"{feature}: {corr:.4f}")
    
    return target_correlations


def detect_outliers(series):
    """Detect outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers


def outlier_analysis(data, numeric_columns):
    """Analyze outliers in the data."""
    print("Outlier Analysis:")
    print("=" * 50)
    
    for col in numeric_columns:
        outliers = detect_outliers(data[col])
        outlier_percent = (len(outliers) / len(data[col])) * 100
        print(f"{col}: {len(outliers)} outliers ({outlier_percent:.2f}%)")


def stationarity_test(data):
    """Check for stationarity using Augmented Dickey-Fuller test."""
    print("Stationarity Test (Augmented Dickey-Fuller):")
    print("=" * 50)
    
    for col in ['close', 'target']:
        series = data[col].dropna()
        result = adfuller(series)
        
        print(f"\n{col.upper()}:")
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6f}")
        print(f"Critical values:")
        for key, value in result[4].items():
            print(f"\t{key}: {value:.3f}")
        
        if result[1] <= 0.05:
            print(f"\t→ {col} is stationary (reject null hypothesis)")
        else:
            print(f"\t→ {col} is non-stationary (fail to reject null hypothesis)")


def print_insights(data, target_clean, target_correlations):
    """Print key insights from the analysis."""
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns.remove('target')
    
    print("Key Insights from Data Exploration:")
    print("=" * 50)
    print("1. Data Structure:")
    print(f"   - {len(data)} observations with {len(numeric_columns)} numeric features")
    print(f"   - Date range: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
    
    print("\n2. Target Variable:")
    print(f"   - Mean: {target_clean.mean():.6f}")
    print(f"   - Standard deviation: {target_clean.std():.6f}")
    print(f"   - Skewness: {target_clean.skew():.6f}")
    
    print("\n3. Feature Correlations:")
    top_correlations = target_correlations[target_correlations.index != 'target'].head(3)
    for feature, corr in top_correlations.items():
        print(f"   - {feature}: {corr:.4f}")
    
    print("\n4. Next Steps:")
    print("   - Feature engineering (technical indicators, lag features)")
    print("   - Model selection and training")
    print("   - Cross-validation and hyperparameter tuning")
    print("   - Ensemble methods for improved performance")


def main():
    """Main data exploration function."""
    print("MITSUI&CO. Commodity Prediction Challenge - Data Exploration")
    print("=" * 70)
    
    # Load configuration
    try:
        config = load_config()
        print("Configuration loaded successfully")
    except:
        print("Using default configuration")
        config = {}
    
    # Create sample data
    print("\nCreating sample data for demonstration...")
    data = create_sample_data()
    print(f"Sample data created with shape: {data.shape}")
    
    # Basic information
    print(f"\nDataset Info:")
    print(f"Shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Number of trading days: {len(data)}")
    
    # Check for missing values
    missing_data = data.isnull().sum()
    missing_percent = (missing_data / len(data)) * 100
    print(f"\nMissing values in target: {missing_data['target']} ({missing_percent['target']:.2f}%)")
    
    # Analyze target variable
    target_clean = analyze_target_variable(data)
    
    # Plot target analysis
    plot_target_analysis(data, target_clean)
    
    # Analyze features
    numeric_columns = analyze_features(data)
    
    # Plot feature time series
    plot_feature_time_series(data)
    
    # Correlation analysis
    target_correlations = correlation_analysis(data, numeric_columns)
    
    # Outlier analysis
    outlier_analysis(data, numeric_columns)
    
    # Stationarity test
    stationarity_test(data)
    
    # Print insights
    print_insights(data, target_clean, target_correlations)
    
    print("\nData exploration completed!")


if __name__ == "__main__":
    main() 
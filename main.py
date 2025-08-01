"""
Main training script for the MITSUI&CO. Commodity Prediction Challenge.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.utils.config import load_config, get_data_paths, get_model_params, get_training_params
from src.data.loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
from src.utils.metrics import evaluate_predictions, sharpe_ratio_variant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and preprocess all data sources.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Loading and preprocessing data...")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Get data paths
    data_paths = get_data_paths(config)
    
    # Load all data sources (this would be updated with actual file paths)
    # For now, we'll create sample data for demonstration
    sample_data = create_sample_data()
    
    # Save sample data
    os.makedirs(data_paths.get('raw_data_path', 'data/raw/'), exist_ok=True)
    sample_data.to_csv(data_paths.get('raw_data_path', 'data/raw/') + 'sample_data.csv', index=False)
    
    logger.info(f"Data loaded with shape: {sample_data.shape}")
    return sample_data


def create_sample_data() -> pd.DataFrame:
    """
    Create sample data for demonstration purposes.
    
    Returns:
        Sample DataFrame with financial data
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Create date range
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create sample price data
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


def engineer_features(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, list]:
    """
    Engineer features from the input data.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (DataFrame with features, list of feature column names)
    """
    logger.info("Engineering features...")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create all features
    df_with_features = feature_engineer.engineer_all_features(df, config)
    
    # Get feature columns
    feature_columns = feature_engineer.get_feature_columns()
    
    # Remove highly correlated features
    df_with_features = feature_engineer.remove_high_correlation_features(df_with_features, threshold=0.95)
    
    logger.info(f"Feature engineering completed. Total features: {len(feature_columns)}")
    return df_with_features, feature_columns


def prepare_data(df: pd.DataFrame, feature_columns: list, target_column: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for training by handling missing values and selecting features.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names
        target_column: Name of the target column
        
    Returns:
        Tuple of (feature DataFrame, target Series)
    """
    logger.info("Preparing data for training...")
    
    # Select features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Handle missing values
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    y = y.fillna(method='ffill').fillna(0)
    
    # Remove rows where target is still NaN
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    logger.info(f"Data prepared. X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any], 
                feature_columns: list) -> XGBoostModel:
    """
    Train the prediction model.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        config: Configuration dictionary
        feature_columns: List of feature column names
        
    Returns:
        Trained model
    """
    logger.info("Training model...")
    
    # Get model parameters
    model_params = get_model_params(config, 'xgboost')
    
    # Initialize model
    model = XGBoostModel(model_params)
    model.set_feature_columns(feature_columns)
    
    # Get training parameters
    training_params = get_training_params(config)
    
    # Split data for validation
    test_size = training_params.get('test_size', 0.2)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train model
    model.fit(X_train, y_train, X_val, y_val)
    
    logger.info("Model training completed")
    return model


def evaluate_model(model: XGBoostModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Make predictions
    predictions = model.predict(X)
    
    # Evaluate using competition metric
    results = evaluate_predictions(y.values, predictions)
    
    logger.info("Model evaluation completed")
    return results


def save_results(model: XGBoostModel, results: Dict[str, float], config: Dict[str, Any]) -> None:
    """
    Save model and results.
    
    Args:
        model: Trained model
        results: Evaluation results
        config: Configuration dictionary
    """
    logger.info("Saving results...")
    
    # Get data paths
    data_paths = get_data_paths(config)
    
    # Create directories if they don't exist
    models_path = data_paths.get('models_path', 'models/')
    os.makedirs(models_path, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_path, 'xgboost_model.pkl')
    model.save_model(model_path)
    
    # Save results
    results_path = os.path.join(models_path, 'training_results.txt')
    with open(results_path, 'w') as f:
        f.write("Training Results:\n")
        f.write("=" * 50 + "\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    logger.info(f"Results saved to {models_path}")


def main():
    """Main training pipeline."""
    try:
        # Load configuration
        logger.info("Starting commodity prediction training pipeline...")
        config = load_config()
        
        # Load and preprocess data
        df = load_and_preprocess_data(config)
        
        # Engineer features
        df_with_features, feature_columns = engineer_features(df, config)
        
        # Prepare data for training
        X, y = prepare_data(df_with_features, feature_columns)
        
        # Train model
        model = train_model(X, y, config, feature_columns)
        
        # Evaluate model
        results = evaluate_model(model, X, y)
        
        # Save results
        save_results(model, results, config)
        
        # Print final results
        logger.info("Training pipeline completed successfully!")
        logger.info("Final Results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Print feature importance
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            logger.info("Top 10 Most Important Features:")
            for feature, importance in feature_importance.head(10).items():
                logger.info(f"{feature}: {importance:.4f}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    main() 
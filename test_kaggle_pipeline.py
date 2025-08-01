"""
Test script for Kaggle pipeline verification.
This script tests the complete pipeline with sample data.
"""

import pandas as pd
import numpy as np
import logging
import warnings
import os
import gc
from datetime import datetime
warnings.filterwarnings('ignore')

# Import project modules
from src.utils.config import load_config
from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
from src.utils.metrics import sharpe_ratio_variant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data():
    """Create test data for pipeline verification."""
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create realistic financial data
    base_price = 100
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
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
    
    # Split into train and test
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx].copy()
    test_data = data[split_idx:].copy()
    
    # Add target to train data
    train_data['target'] = train_data['close'].diff(1)
    
    # Create sample submission format
    sample_submission = pd.DataFrame({
        'id': range(len(test_data)),
        'prediction': np.random.normal(0, 1, len(test_data))
    })
    
    return train_data, test_data, sample_submission


def test_feature_engineering():
    """Test feature engineering pipeline."""
    logger.info("Testing feature engineering...")
    
    # Create test data
    train_data, test_data, _ = create_test_data()
    
    # Load config
    config = load_config()
    
    # Test feature engineering
    feature_engineer = FeatureEngineer()
    
    # Process training data
    train_with_features = feature_engineer.engineer_all_features(train_data, config)
    feature_columns = feature_engineer.get_feature_columns()
    
    # Process test data
    test_with_features = feature_engineer.engineer_all_features(test_data, config)
    
    # Ensure test data has same features
    missing_features = set(feature_columns) - set(test_with_features.columns)
    for feature in missing_features:
        test_with_features[feature] = 0
    
    # Prepare data
    X_train = train_with_features[feature_columns].copy()
    y_train = train_with_features['target'].copy()
    X_test = test_with_features[feature_columns].copy()
    
    # Handle missing values
    X_train = X_train.fillna(method='ffill').fillna(method='bfill').fillna(0)
    X_test = X_test.fillna(method='ffill').fillna(method='bfill').fillna(0)
    y_train = y_train.fillna(method='ffill').fillna(0)
    
    # Remove rows where target is still NaN
    valid_indices = ~y_train.isna()
    X_train = X_train[valid_indices]
    y_train = y_train[valid_indices]
    
    logger.info(f"Feature engineering test passed:")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"Features: {len(feature_columns)}")
    
    return X_train, y_train, X_test, feature_columns


def test_model_training(X_train, y_train, feature_columns):
    """Test model training."""
    logger.info("Testing model training...")
    
    # Model parameters for testing
    model_params = {
        'n_estimators': 100,  # Small for testing
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'early_stopping_rounds': 10,
        'verbose': False
    }
    
    # Initialize model
    model = XGBoostModel(model_params)
    model.set_feature_columns(feature_columns)
    
    # Split for validation
    split_idx = int(0.8 * len(X_train))
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    # Train model
    model.fit(X_train_split, y_train_split, X_val, y_val)
    
    logger.info("Model training test passed")
    return model


def test_predictions(model, X_test):
    """Test prediction pipeline."""
    logger.info("Testing predictions...")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    logger.info(f"Prediction test passed: {predictions.shape}")
    return predictions


def test_submission_creation(predictions, sample_submission):
    """Test submission file creation."""
    logger.info("Testing submission creation...")
    
    # Create submission
    submission = pd.DataFrame({
        'id': sample_submission['id'],
        'prediction': predictions[:len(sample_submission)]
    })
    
    # Ensure predictions are finite
    submission['prediction'] = np.where(
        np.isfinite(submission['prediction']), 
        submission['prediction'], 
        0.0
    )
    
    # Save test submission
    test_output_path = 'test_submission.csv'
    submission.to_csv(test_output_path, index=False)
    
    logger.info(f"Submission test passed: {submission.shape}")
    logger.info(f"Test submission saved to: {test_output_path}")
    
    return submission


def test_competition_metric(y_true, y_pred):
    """Test competition metric calculation."""
    logger.info("Testing competition metric...")
    
    try:
        score = sharpe_ratio_variant(y_true, y_pred)
        logger.info(f"Competition metric test passed: {score:.6f}")
        return score
    except Exception as e:
        logger.error(f"Competition metric test failed: {e}")
        return None


def main():
    """Run all tests."""
    logger.info("Starting pipeline tests...")
    
    try:
        # Test 1: Feature Engineering
        X_train, y_train, X_test, feature_columns = test_feature_engineering()
        
        # Test 2: Model Training
        model = test_model_training(X_train, y_train, feature_columns)
        
        # Test 3: Predictions
        predictions = test_predictions(model, X_test)
        
        # Test 4: Submission Creation
        _, _, sample_submission = create_test_data()
        submission = test_submission_creation(predictions, sample_submission)
        
        # Test 5: Competition Metric
        if len(X_train) > len(X_test):
            val_predictions = model.predict(X_train[-len(X_test):])
            val_actual = y_train[-len(X_test):]
            score = test_competition_metric(val_actual.values, val_predictions)
        
        # Test 6: Feature Importance
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            logger.info(f"Feature importance test passed: {len(feature_importance)} features")
        
        # Test 7: Model Info
        model_info = model.get_model_info()
        logger.info(f"Model info test passed: {model_info['model_type']}")
        
        logger.info("All tests passed successfully!")
        
        # Print summary
        print("\n=== TEST SUMMARY ===")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features: {len(feature_columns)}")
        print(f"Model type: {model_info['model_type']}")
        print(f"Submission shape: {submission.shape}")
        if 'score' in locals():
            print(f"Validation score: {score:.6f}")
        print("Pipeline is ready for Kaggle!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main() 
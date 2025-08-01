# MITSUI&CO. Commodity Prediction Challenge - Kaggle Notebook
# 
# IMPORTANT: This notebook is designed to run without internet access
# All required libraries should be pre-installed in Kaggle environment

# ============================================================================
# CELL 1: Import Libraries (No internet access required)
# ============================================================================

import pandas as pd
import numpy as np
import logging
import warnings
import os
import gc
from datetime import datetime
warnings.filterwarnings('ignore')

# Verify XGBoost is available (should be pre-installed)
try:
    import xgboost as xgb
    print("✅ XGBoost is available")
except ImportError:
    print("❌ XGBoost not available. Please check Kaggle environment.")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 50)
print("MITSUI&CO. COMMODITY PREDICTION CHALLENGE")
print("KAGGLE NOTEBOOK - NO INTERNET ACCESS")
print("=" * 50)

# ============================================================================
# CELL 2: Load Data
# ============================================================================

print("📊 Loading competition data...")

try:
    # Load data from Kaggle competition
    train_data = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
    test_data = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/test.csv')
    
    print(f"✅ Data loaded successfully!")
    print(f"   Train data: {train_data.shape}")
    print(f"   Test data: {test_data.shape}")
    
    # Check data columns
    print(f"\n Train data columns: {len(train_data.columns)}")
    print(f" Test data columns: {len(test_data.columns)}")
    
    # Check for target variable in train_labels.csv
    try:
        train_labels = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')
        print(f"✅ Train labels loaded: {train_labels.shape}")
        print(f"   Labels columns: {train_labels.columns.tolist()}")
        print(f"   Labels head:")
        print(train_labels.head())
    except:
        print("⚠️  Train labels not found, will create dummy target")
    
    # Create sample_submission
    sample_submission = pd.DataFrame({
        'id': [f"{i}_{j}" for i in range(len(test_data)) for j in range(424)],
        'prediction': 0.0
    })
    
    print(f"📋 Sample submission created: {sample_submission.shape}")
    
    print("🔍 Checking data content...")
    
    # Check training data
    print("Train data info:")
    print(train_data.info())
    print("\nTrain data head:")
    print(train_data.head())
    
    print("\n" + "="*50 + "\n")
    
    # Check test data
    print("Test data info:")
    print(test_data.info())
    print("\nTest data head:")
    print(test_data.head())
    
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}")
    print("Please ensure the competition data is properly attached to the notebook.")
    raise

# ============================================================================
# CELL 3: Feature Engineering
# ============================================================================

print(" Preparing features...")

# Use train_labels loaded in Cell 2
print("✅ Using train labels loaded in Cell 2")
print(f"   Labels shape: {train_labels.shape}")

# Merge train data with labels
train_data = train_data.merge(train_labels, on='date_id', how='left')
print(f"   After merge: {train_data.shape}")

# Get target columns (target_0 to target_423)
target_cols = [col for col in train_data.columns if col.startswith('target_')]
print(f"   Number of targets: {len(target_cols)}")

# Remove date_id, is_scored, and target columns from features
feature_cols = [col for col in train_data.columns if col not in ['date_id', 'is_scored'] + target_cols]
test_feature_cols = [col for col in test_data.columns if col not in ['date_id', 'is_scored']]

print(f"📊 Feature preparation:")
print(f"   Training features: {len(feature_cols)}")
print(f"   Test features: {len(test_feature_cols)}")

# Prepare training data
X_train = train_data[feature_cols].fillna(0)
y_train = train_data[target_cols].fillna(0)

# Prepare test data
X_test = test_data[test_feature_cols].fillna(0)

print(f"✅ Data prepared successfully!")
print(f"   X_train: {X_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   Target columns: {len(target_cols)}")

# ============================================================================
# CELL 4: Train Model
# ============================================================================

print("🤖 Training XGBoost model for multi-target prediction...")

from sklearn.multioutput import MultiOutputRegressor

# Model parameters optimized for multi-target prediction
base_params = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

print(f"   Model parameters: {base_params}")

# Create multi-output regressor
base_model = xgb.XGBRegressor(**base_params)
model = MultiOutputRegressor(base_model, n_jobs=-1)

# Train the model
model.fit(X_train, y_train)

print("✅ Model training completed!")

# ============================================================================
# CELL 5: Make Predictions
# ============================================================================

print(" Making predictions...")

# Make predictions for all targets
predictions = model.predict(X_test)
predictions = np.nan_to_num(predictions, nan=0.0)

print(f"✅ Predictions completed! Shape: {predictions.shape}")
print(f"   Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
print(f"   Prediction mean: {predictions.mean():.4f}")

# Create submission dataframe
submission_data = []
for i in range(len(test_data)):
    for j in range(predictions.shape[1]):
        submission_data.append({
            'id': f"{i}_{j}",
            'prediction': predictions[i, j]
        })

submission = pd.DataFrame(submission_data)

# Save as parquet
submission.to_parquet('/kaggle/working/submission.parquet', index=False)

print("✅ Submission file created!")
print(f"   File: /kaggle/working/submission.parquet")
print(f"   Shape: {submission.shape}")
print(f"   Sample predictions:")
print(submission.head(10))
print("🎉 Ready for submission!")

# Verify submission file
print("\n🔍 Verifying submission file...")
file_path = '/kaggle/working/submission.parquet'
if os.path.exists(file_path):
    file_size = os.path.getsize(file_path)
    print(f"✅ Submission file exists: {file_size} bytes")
    
    # Load and check content
    submission_check = pd.read_parquet(file_path)
    print(f"📊 Submission shape: {submission_check.shape}")
    print(f"📋 Columns: {submission_check.columns.tolist()}")
    print(f"📈 First 5 rows:")
    print(submission_check.head())
else:
    print("❌ Submission file not found")

# Final completion message
print("\n" + "=" * 50)
print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
print("=" * 50)
print("🚀 You can now submit this file to the competition!")
print("=" * 50)

print("\n📋 IMPORTANT REMINDERS:")
print("1. This notebook runs WITHOUT internet access")
print("2. All libraries are pre-installed in Kaggle environment")
print("3. Output file is in .parquet format as required")
print("4. Ready for submission!")
 
# MITSUI&CO. Commodity Prediction Challenge - Kaggle Guide

A comprehensive guide for setting up and running the commodity prediction model on Kaggle's GPU environment.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [File Structure](#file-structure)
4. [Competition Requirements](#competition-requirements)
5. [Model Features](#model-features)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Submission Process](#submission-process)

## 🎯 Project Overview

This project implements a robust machine learning pipeline for predicting commodity returns using historical data from multiple financial markets:

- **Objective**: Predict future commodity returns
- **Evaluation Metric**: Sharpe ratio variant (Spearman correlation / std)
- **Data Sources**: LME, JPX, US Stock, Forex markets
- **Target**: Price-difference series between two assets
- **Model**: XGBoost with comprehensive feature engineering

## 🚀 Quick Start

### Step 1: Create Kaggle Notebook

1. Visit [MITSUI&CO. Commodity Prediction Challenge](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge)
2. Click "Create" → "Notebook"
3. Select "GPU" as accelerator
4. Set language to "Python"

### Step 2: Disable Internet Access

**CRITICAL STEP:**
1. In your Kaggle notebook, go to **Settings** (gear icon)
2. Turn **OFF** "Internet" option
3. Save the notebook

### Step 3: Use the Submission Code

**Recommended Approach:**
1. Copy the code from `kaggle_submission_final.py`
2. Paste into separate cells in your Kaggle notebook
3. Run each cell sequentially
4. The code will automatically:
   - Load competition data
   - Create features
   - Train XGBoost model
   - Generate predictions
   - Save submission file

**That's it!** No complex setup needed.

### Step 3: Run the Script

Add this cell to run the complete pipeline:

```python
# Run the submission script
!python kaggle_submission.py
```

The script will automatically install XGBoost if needed.

## 📁 File Structure

### On Kaggle
```
/kaggle/working/
├── kaggle_submission.py   # Main submission script
└── submission.csv         # Generated submission file
```

### Project Components

- **Data Loader**: Handles multiple data sources (LME, JPX, US Stock, Forex)
- **Feature Engineer**: Creates 857+ features including technical indicators
- **XGBoost Model**: Optimized for large datasets and time constraints
- **Evaluation Metrics**: Implements competition-specific Sharpe ratio variant

## ✅ Competition Requirements

### Time Limits
- **CPU Notebook**: ≤ 8 hours
- **GPU Notebook**: ≤ 8 hours
- **Forecasting Phase**: ≤ 9 hours

### Environment Constraints
- **Internet Access**: **MUST BE DISABLED** (no external API calls, no pip install)
- **Memory**: Optimized for large datasets
- **GPU**: Utilized for faster computation
- **Output Format**: `.parquet` file required

### Submission Format
- **File Type**: CSV
- **Columns**: 'id', 'prediction'
- **Path**: `/kaggle/working/submission.csv`
- **Validation**: Handles infinite values and missing data

## 🔧 Model Features

### Feature Engineering (857+ Features)

1. **Technical Indicators**
   - RSI, MACD, Bollinger Bands
   - Stochastic Oscillator, Williams %R
   - CCI, ADX, Moving Averages

2. **Time-Series Features**
   - Lag features (1, 2, 3, 5, 10, 20 periods)
   - Rolling statistics (mean, std, min, max, median)
   - Price differences and momentum

3. **Volatility Features**
   - Rolling volatility measures
   - Realized volatility calculations

4. **Time Features**
   - Cyclical encoding of dates
   - Day of week, month, quarter patterns

### Model Configuration

```yaml
xgboost:
  n_estimators: 500      # Reduced for speed
  max_depth: 4           # Reduced for speed
  learning_rate: 0.05    # Increased for speed
  subsample: 0.8
  colsample_bytree: 0.8
  early_stopping_rounds: 20
  random_state: 42
```

## ⚡ Performance Optimization

### Memory Optimization
- **Batch Processing**: Predictions in batches of 10,000
- **Garbage Collection**: Automatic cleanup after each step
- **Memory Monitoring**: Usage tracking and optimization

### Time Optimization
- **Reduced Parameters**: Conservative model settings
- **Early Stopping**: Prevents overfitting and saves time
- **Efficient Features**: Optimized feature engineering pipeline

### Expected Performance
- **Training Time**: ~10 minutes
- **Prediction Time**: ~2 minutes
- **Total Runtime**: ~15 minutes
- **Memory Usage**: < 8GB

## 🔍 Troubleshooting

### Common Issues

1. **Memory Error**
   ```python
   # Reduce batch size in kaggle_submission.py
   batch_size = 5000  # Instead of 10000
   ```

2. **Time Limit Exceeded**
   ```yaml
   # Modify config.yaml
   xgboost:
     n_estimators: 300  # Further reduce
     learning_rate: 0.1  # Further increase
   ```

3. **Too Many Features**
   ```yaml
   # Modify config.yaml
   features:
     rolling_windows: [5, 10, 20]  # Reduce windows
     lag_features: [1, 2, 3]       # Reduce lags
   ```

4. **Data File Not Found**
   ```python
   # Check file paths
   train_path = '/kaggle/input/mitsui-commodity-prediction-challenge/train.csv'
   test_path = '/kaggle/input/mitsui-commodity-prediction-challenge/test.csv'
   ```

### Performance Monitoring

```python
# Check memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Check execution time
from datetime import datetime
start_time = datetime.now()
# ... your code ...
end_time = datetime.now()
print(f"Execution time: {end_time - start_time}")
```

## 📊 Expected Output

### Log Output Example
```
2024-01-01 10:00:00 - INFO - Starting Kaggle submission pipeline
2024-01-01 10:00:05 - INFO - Train data loaded: (100000, 15)
2024-01-01 10:00:10 - INFO - Test data loaded: (25000, 14)
2024-01-01 10:05:00 - INFO - Feature engineering completed. Total features: 857
2024-01-01 10:10:00 - INFO - Model training completed
2024-01-01 10:12:00 - INFO - Predictions completed: (25000,)
2024-01-01 10:12:05 - INFO - Submission saved to /kaggle/working/submission.csv
2024-01-01 10:12:05 - INFO - Pipeline completed successfully in 0:12:05
```

### Submission File Format
```csv
id,prediction
0,0.123456
1,-0.045678
2,0.234567
3,0.089123
4,-0.156789
...
```

## 🎯 Submission Process

### Pre-Submission Checklist

- [ ] All required files uploaded
- [ ] Dependencies installed
- [ ] Script runs without errors
- [ ] Submission file format correct
- [ ] Predictions in reasonable range
- [ ] No infinite or NaN values
- [ ] Runtime within 8-hour limit

### Submission Steps

1. **Run Pipeline**: Execute `kaggle_submission.py`
2. **Verify Output**: Check submission.csv is created
3. **Validate Format**: Ensure correct column names and data types
4. **Submit**: Use Kaggle's submission interface
5. **Monitor**: Check leaderboard for results

### Success Indicators

- ✅ Pipeline completes without errors
- ✅ Submission file generated successfully
- ✅ Valid score on leaderboard
- ✅ Runtime under time limit
- ✅ Memory usage within limits

## 🏆 Tips for Better Performance

1. **Feature Selection**: Use correlation analysis to remove redundant features
2. **Hyperparameter Tuning**: Use Optuna for automated optimization
3. **Ensemble Methods**: Combine multiple models for better predictions
4. **Cross-Validation**: Use time-series CV instead of random splits
5. **Feature Engineering**: Experiment with different technical indicators

## 📞 Support

If you encounter issues:

1. **Check Logs**: Review detailed error messages
2. **Verify Data**: Ensure competition data is accessible
3. **Test Locally**: Use `test_kaggle_pipeline.py` for testing
4. **Adjust Parameters**: Modify configuration based on errors
5. **Monitor Resources**: Track memory and time usage

## 🎉 Success Criteria

Upon completion, you should have:

1. ✅ Successfully run the complete ML pipeline
2. ✅ Generated a properly formatted submission file
3. ✅ Obtained a valid score on the Kaggle leaderboard
4. ✅ Participated in the competition rankings

**Good luck with the competition!** 🚀

---

*This guide provides everything you need to successfully participate in the MITSUI&CO. Commodity Prediction Challenge on Kaggle.* 
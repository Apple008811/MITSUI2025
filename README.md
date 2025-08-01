# MITSUI&CO. Commodity Prediction Challenge

A comprehensive machine learning project for predicting commodity returns using historical data from multiple financial markets.

## 🎯 Competition Overview

This is a Kaggle competition focused on commodity price prediction, aiming to develop a robust model for accurately predicting commodity returns.

### Competition Key Points
- **Objective**: Predict future commodity returns using historical data from LME, JPX, US Stock, and Forex markets
- **Evaluation Metric**: Variant of Sharpe ratio, calculated as mean Spearman rank correlation between predictions and targets divided by standard deviation
- **Data Sources**: London Metal Exchange (LME), Japan Exchange Group (JPX), US Stock, Forex markets
- **Prediction Target**: Price-difference series (time-series differences between two distinct assets' prices)

### Timeline
- **Start Date**: July 24, 2025
- **Entry Deadline**: September 29, 2025
- **Team Merger Deadline**: September 29, 2025
- **Final Submission Deadline**: October 6, 2025
- **Competition End**: January 16, 2026

### Prizes
- 1st Place: $20,000
- 2nd Place: $10,000
- 3rd Place: $10,000
- 4th-15th Place: $5,000

## 🏗️ Project Structure

```
├── data/                   # Data files directory
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── features/         # Feature engineering modules
│   ├── models/           # Model modules
│   └── utils/            # Utility functions
├── config/               # Configuration files
├── requirements.txt      # Python dependencies
├── kaggle_submission_final.py  # Main Kaggle submission script
├── test_kaggle_pipeline.py # Local testing script
└── README.md            # Project documentation
```

## 🚀 Quick Start

### For Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run local test
python test_kaggle_pipeline.py
```

### For Kaggle Submission
See [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) for detailed instructions on running the project on Kaggle.

## 🔧 Key Features

### Data Processing
- Multi-source data loader (LME, JPX, US Stock, Forex)
- Robust missing value handling
- Time-series aware data validation

### Feature Engineering (857+ Features)
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Lag features and rolling statistics
- Volatility and momentum measures
- Time-based features with cyclical encoding

### Machine Learning
- XGBoost model optimized for large datasets
- Memory-efficient training and prediction
- Competition-specific evaluation metrics
- Feature importance analysis

### Performance Optimization
- Batch processing for large datasets
- Automatic garbage collection
- Memory usage monitoring
- Time-optimized model parameters

## 📊 Model Performance

- **Training Time**: ~10 minutes
- **Prediction Time**: ~2 minutes
- **Total Runtime**: ~15 minutes (well within 8-hour limit)
- **Memory Usage**: < 8GB
- **Features**: 857+ engineered features

## 🎯 Usage Instructions

### Local Development
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run local test: `python test_kaggle_pipeline.py`
4. Explore notebooks in `notebooks/` directory

### Kaggle Submission
1. Follow the detailed guide in [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)
2. Upload required files to Kaggle notebook
3. Run `python kaggle_submission_final.py`
4. Submit generated `submission.parquet` file

## 📁 Key Files

- **`kaggle_submission_final.py`**: Main script for Kaggle submission
- **`test_kaggle_pipeline.py`**: Local testing script
- **`config/config.yaml`**: Model and feature configuration
- **`src/`**: Complete source code modules
- **`KAGGLE_GUIDE.md`**: Detailed Kaggle setup and usage guide

## 🔍 Testing

The project includes comprehensive testing:
```bash
# Run complete pipeline test
python test_kaggle_pipeline.py
```

This will test:
- Data loading and preprocessing
- Feature engineering
- Model training
- Prediction generation
- Submission file creation

## 📈 Expected Results

- **Validation Score**: Based on Sharpe ratio variant
- **Submission Format**: Parquet with 'id' and 'prediction' columns
- **File Location**: `/kaggle/working/submission.parquet` (on Kaggle)

## 🛠️ Configuration

Model parameters can be adjusted in `config/config.yaml`:
- XGBoost hyperparameters
- Feature engineering settings
- Training parameters

## 📞 Support

- **Local Issues**: Check `test_kaggle_pipeline.py` output
- **Kaggle Issues**: See [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) troubleshooting section
- **Performance**: Monitor memory and time usage

## 🏆 Success Criteria

Upon successful completion:
- ✅ Pipeline runs without errors
- ✅ Submission file generated correctly
- ✅ Valid score on Kaggle leaderboard
- ✅ Runtime within competition limits

---

**Ready to compete?** Check out [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) for detailed Kaggle setup instructions! 🚀 
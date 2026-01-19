# UK-Gas-Demand-Model
Neural network training pipeline to predict gas demand in the UK.

# UK Gas Demand Forecasting

## Overview
Neural network model for predicting UK gas demand using weather forecasts and time features. Implements feature engineering and MLP regression in a scikit-learn pipeline.

## Features
- **Cyclical time encoding**: Month, day-of-week, week-of-year as sine/cosine features
- **Delta features**: 1-3 day trailing differences for weather variables
- **Neural network**: 2-layer MLP (128→64) with ReLU activation
- **Robust training**: Early stopping, adaptive learning, L2 regularization

## Quick Start
```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Run model
python forecasting_model.py
```

## Input/Output
- **Input**: `train.csv` and `test.csv` with date, weather, and demand columns
- **Output**: `submission.csv` with ID and predicted demand

## Key Parameters
```python
forecast_cols = ['temp_1','temp_night_1','wind_1','wind_night_1','ssrd_ratio_1']
lags = [1, 2, 3]
model = MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True)
```

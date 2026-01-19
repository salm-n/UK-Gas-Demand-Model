import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

# --- Helper functions ---
def add_cyclical_features(df, date_col="date"):
    """
    Adds cyclical features for month, day-of-week, and week-of-year
    to capture seasonality in time series data.
    """
    d = df[date_col]
    df['month_sin'] = np.sin(2 * np.pi * d.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * d.dt.month / 12)
    df['dow_sin'] = np.sin(2 * np.pi * d.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * d.dt.dayofweek / 7)
    df['week_sin'] = np.sin(2 * np.pi * d.dt.isocalendar().week / 52)
    df['week_cos'] = np.cos(2 * np.pi * d.dt.isocalendar().week / 52)
    return df

def add_delta_features(df, forecast_cols, lags):
    """
    Adds trailing delta features for specified forecast columns and lags.
    Delta = current value - lagged value.
    """
    df = df.copy()
    for lag in lags:
        for col in forecast_cols:
            df[f'{col}_delta_{lag}'] = df[col] - df[col].shift(lag)
    return df

def make_nn():
    """
    Creates a neural network regressor pipeline with StandardScaler.
    Architecture: two hidden layers (128, 64), ReLU activation, adaptive learning.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            alpha=0.001,
            learning_rate_init=0.001,
            learning_rate="adaptive",
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            activation="relu",
            solver="adam",
            random_state=99
        ))
    ])

def load_and_prepare(path, forecast_cols, lags, max_lag):
    """
    Loads a CSV file, converts date, adds cyclical features,
    adds trailing delta features, and forward-fills the first max_lag rows.
    Returns the prepared DataFrame.
    """
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = add_cyclical_features(df)
    df = add_delta_features(df, forecast_cols, lags)
    
    # Forward-fill first max_lag rows for delta columns
    for col in [c for c in df.columns if "delta" in c]:
        first_real = df.loc[max_lag, col]
        df.loc[:max_lag-1, col] = first_real
        
    return df

# --- Parameters ---
forecast_cols = ['temp_1','temp_night_1','wind_1','wind_night_1','ssrd_ratio_1']
lags = [1,2,3]
max_lag = max(lags)

# --- Load and prepare data ---
train = load_and_prepare("train.csv", forecast_cols, lags, max_lag)
test  = load_and_prepare("test.csv", forecast_cols, lags, max_lag)

# --- Prepare features and target ---
feature_cols = [c for c in train.columns if c not in ["id", "date", "demand"]]
X_train = train[feature_cols].astype(float)
y_train = train["demand"].astype(float)
X_test  = test[feature_cols].astype(float)

# --- Train and predict ---
nn = make_nn()
nn.fit(X_train, y_train)
test['demand'] = nn.predict(X_test)

# --- Save submission ---
submission = test[['id', 'demand']]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

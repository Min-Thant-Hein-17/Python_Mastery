# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:47:51 2026

@author: ASUS
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# %% 1. Load Data
file_path = "synthetic_ml_metrics_datetime_2026.csv"
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# %% 2. Target Columns
target_cols = ["accuracy", "loss", "precision", "recall", "f1_score"]



# %% 3. Feature Engineering

# There are 8 events/day, choose window size = 8
window_size = 8

def mad(x):
    return np.mean(np.abs(x - np.mean(x)))

for col in target_cols:
    df[f"{col}_roll_mean"] = df[col].rolling(window=window_size).mean()
    df[f"{col}_roll_median"] = df[col].rolling(window=window_size).median()
    df[f"{col}_roll_std"] = df[col].rolling(window=window_size).std()
    df[f"{col}_roll_mad"] = df[col].rolling(window=window_size).apply(mad, raw=True)

extra_cols = [
    "new_data_arrived",
    "retraining_on",
    "retraining_complete",
    "event_order"
]

feature_cols = target_cols + extra_cols + [
    c for c in df.columns if "roll_" in c
]


df = df.dropna().reset_index(drop=True)


# %% 4. Preparing X and y
X_raw = df[feature_cols].copy()
y_raw = df[target_cols].shift(-1)

X_raw = X_raw.iloc[:-1]
y_raw = y_raw.iloc[:-1]



# %% Train/val/test (time-based)

Train -> 70%
Val -> 15%
Test -> 15%


n = len(X_raw)
train_end = int(n * 0.70)
val_end = int(n * 0.85)


X_train_raw = X_raw.iloc[:train_end]
X_val_raw = X_raw.iloc[train_end:val_end]
X_test_raw = X_raw.iloc[val_end:]

y_train_raw = y_raw.iloc[:train_end]
y_val_raw = y_raw.iloc[train_end:val_end]
y_test_raw = y_raw.iloc[val_end:]


# %% 6. Scale features and targets

X_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))

X_train_scaled = X_scaler.fit_transform(X_train_raw)
X_val_scaled = X_scaler.transform(X_val_raw)
X_test_scaled = X_scaler.transform(X_test_raw)

y_train_scaled = y_scaler.fit_transform(y_train_raw)
y_val_scaled = y_scaler.transform(y_val_raw)
y_test_scaled = y_scaler.transform(y_test_raw)


# %% 7. Create sequences

seq_length = 16  # 16 timesteps = 2 days because 8 events

def create_sequences(X, y, seq_length=16):



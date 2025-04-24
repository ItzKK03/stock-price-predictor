import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.indicators import add_indicators

def train_lstm_model():
    TICKER = 'AAPL'
    START_DATE = "2012-01-01"
    END_DATE = "2024-01-01"
    SEQ_LEN = 60

    df = yf.download(TICKER, start=START_DATE, end=END_DATE)
    df = add_indicators(df)
    FEATURES = ['Close', 'MA20', 'MA50', 'RSI', 'Volume']
    df = df[FEATURES].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    joblib.dump(scaler, 'saved_models/scaler.save')

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled_data)):
        X.append(scaled_data[i-SEQ_LEN:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    os.makedirs('saved_models', exist_ok=True)
    checkpoint = ModelCheckpoint('saved_models/lstm_stock_model.h5', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=32,
                        callbacks=[checkpoint, early_stop],
                        verbose=1)

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.savefig("saved_models/loss_curve.png")
    plt.close()

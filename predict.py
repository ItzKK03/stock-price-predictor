import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from utils.indicators import add_indicators

SEQUENCE_LENGTH = 60
FEATURES = ['Close', 'MA20', 'MA50', 'RSI', 'Volume']

def predict_next_day_close(ticker):
    model = load_model('saved_models/lstm_stock_model.h5')
    scaler = joblib.load('saved_models/scaler.save')

    df = yf.download(ticker, period='6mo', interval='1d')
    df = add_indicators(df)
    df = df[FEATURES].dropna()

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError("Not enough data to make prediction.")

    scaled = scaler.transform(df)
    input_seq = np.expand_dims(scaled[-SEQUENCE_LENGTH:], axis=0)

    pred_scaled = model.predict(input_seq)
    padded = np.concatenate([pred_scaled, np.zeros((1, len(FEATURES)-1))], axis=1)
    prediction = scaler.inverse_transform(padded)[0][0]

    return df, prediction

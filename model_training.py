import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

from data_processing import get_indian_stocks, get_stock_data, create_training_data

def train_model():
    """Trains the LSTM model and saves it."""
    tickers = get_indian_stocks()
    all_data = []

    for ticker in tickers:
        data = get_stock_data(ticker)
        if data is not None:
            all_data.append(data)

    if not all_data:
        print("No valid stock data available for training.")
        return

    data = np.concatenate(all_data, axis=0)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X_train, y_train = create_training_data(scaled_data)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=1)

    # Ensure the models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")

    model.save("models/lstm_stock_model.keras")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from src.data_processing import get_indian_stocks, get_stock_data, create_training_data

def train_model():
    """Trains the LSTM model with engineered features and saves it."""
    tickers = get_indian_stocks()
    all_data = []

    for ticker in tickers:
        # get_stock_data now returns data with features (Close, SMA)
        data = get_stock_data(ticker)
        if data is not None:
            all_data.append(data)

    if not all_data:
        print("No valid stock data available for training.")
        return

    # Concatenate all data
    data = np.concatenate(all_data, axis=0)

    # Scale both features (Close and SMA)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create training data with the new features
    X_train, y_train = create_training_data(scaled_data)

    # The data is already in the correct shape: [samples, time steps, features]
    # No need to reshape X_train if create_training_data handles it correctly

    # Define the model, updating the input shape for multiple features
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
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

    print("Model trained and saved successfully with new features!")

if __name__ == "__main__":
    train_model()

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os
import yfinance as yf
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processing import get_indian_stocks, get_stock_data

def get_stock_data_for_chart(ticker, period="6mo"):
    """Fetches historical stock data and returns it as a DataFrame."""
    try:
        stock_df = yf.download(ticker, period=period, interval="1d")
        if stock_df.empty:
            st.error(f"No data found for {ticker}")
            return None
        return stock_df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Load trained model and scaler
MODEL_PATH = os.path.join("models", "lstm_stock_model.keras")
SCALER_PATH = os.path.join("models", "scaler.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler not found. Please train the model first.")
    st.stop()

try:
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mean_squared_error")
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Streamlit UI
st.title("Indian Stock Market Prediction (LSTM)")

tickers = get_indian_stocks()
selected_stock = st.selectbox("Select a stock:", tickers)

if st.button("Predict"):
    # Fetch data for prediction (last 30 days)
    data = get_stock_data(selected_stock, period="31d")

    if data is None or len(data) < 30:
        st.error("Not enough data available! Need at least 30 days of stock prices.")
    else:
        try:
            # Scale the data and prepare the input for the model
            scaled_data = scaler.transform(data)
            test_input = scaled_data[-30:].reshape(1, 30, 1) # 1 feature

            # Make a prediction
            pred_scaled = model.predict(test_input)

            # Inverse transform the prediction
            pred = scaler.inverse_transform(pred_scaled)

            st.success(f"Predicted Stock Price for {selected_stock}: ₹{pred[0][0]:.2f}")

            # Fetch data for the chart
            chart_data = get_stock_data_for_chart(selected_stock)
            if chart_data is not None:
                st.subheader("Historical Stock Prices")
                st.line_chart(chart_data['Close'])

        except Exception as e:
            st.error(f"Prediction error: {e}")

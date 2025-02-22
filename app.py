import streamlit as st
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os

# Load trained model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
model = load_model(os.path.join(BASE_DIR, "lstm_stock_model.keras"), compile=False)
model.compile(optimizer="adam", loss="mean_squared_error")
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Predefined list of Indian stocks (NIFTY 50 & Sensex)
def get_indian_stocks():
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "ITC.NS",
        "TATASTEEL.NS", "AXISBANK.NS", "WIPRO.NS", "BHARTIARTL.NS", "LT.NS",
        "MARUTI.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "TATAMOTORS.NS", "SUNPHARMA.NS",
        "M&M.NS", "ULTRACEMCO.NS", "INDUSINDBK.NS", "POWERGRID.NS", "HCLTECH.NS"
    ]

# Function to fetch stock data
def get_stock_data(ticker):
    try:
        stock = yf.download(ticker, period="31d", interval="1d")
        if stock.empty:
            st.error(f"Stock data for {ticker} not found.")
            return None
        return stock["Close"].dropna().values.reshape(-1, 1)
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Streamlit UI
st.title("Indian Stock Market Prediction (LSTM)")

# Dropdown menu to select a stock
tickers = get_indian_stocks()
selected_stock = st.selectbox("Select a stock:", tickers)

if st.button("Predict"):
    data = get_stock_data(selected_stock)
    
    if data is None or len(data) < 30:
        st.error("Not enough data available! Need at least 30 days of stock prices.")
    else:
        try:
            scaled_data = scaler.transform(data)
            test_input = scaled_data[-30:].reshape(1, 30, 1)

            pred_scaled = model.predict(test_input)
            pred = scaler.inverse_transform(pred_scaled)

            st.success(f"Predicted Stock Price for {selected_stock}: ₹{pred[0][0]:.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
